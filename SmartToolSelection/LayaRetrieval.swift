import Foundation
import PicoDecisions
import PicoDecisionsMLX
import Tokenizers

enum LayaScoring: String, CaseIterable, Identifiable, Sendable {
    case multipleChoice, boolean
    var id: String { rawValue }
    var title: String { self == .multipleChoice ? "Multiple choice" : "Boolean" }
}

enum DecisionPrecision: String, CaseIterable, Identifiable, Sendable {
    case float16, float32
    var id: String { rawValue }
    var title: String { self == .float16 ? "FP16" : "FP32" }
    var layaPrecision: LayaPrecision { self == .float16 ? .float16 : .float32 }
}

/// Serialize GPU work across the retriever and decision model, including loads.
/// Actors alone do not serialize operations across an `await` suspension.
actor DeviceInferenceQueue {
    private var tail: Task<Void, Never>?

    func run<Value: Sendable>(
        _ operation: @escaping @Sendable () async throws -> Value
    ) async throws -> Value {
        let previous = tail
        let task = Task {
            await previous?.value
            try Task.checkCancellation()
            return try await operation()
        }
        tail = Task { _ = try? await task.value }
        return try await withTaskCancellationHandler {
            try await task.value
        } onCancel: {
            task.cancel()
        }
    }
}

/// A score over the full tool catalog. Boolean confidence describes certainty in
/// relevance or irrelevance. Choice scores rank only surviving finalists.
nonisolated struct ToolSearchScore: Sendable {
    let score: Double
    var confidence: Double? = nil
    var inputDiagnostics: DecisionInputDiagnostics? = nil
    var isChoiceProbability = false
    var isFinalist = true
}

nonisolated protocol ToolSearchEngine: Sendable {
    func load(directory: URL, backend: Backend, precision: DecisionPrecision, tools: [Tool]) async throws
    func unload() async
    func scores(for query: String, tools: [Tool], layaScoring: LayaScoring) async throws -> [ToolSearchScore]
}

/// The app loads one backend at a time. DeviceInferenceQueue serializes complete
/// load/search/unload operations even when this actor suspends.
actor CatalogSearchEngine: ToolSearchEngine {
    private let retriever = RetrievalEngine()
    private var laya: LayaModel?
    private var choices: LayaChoiceCatalog?
    private var backend: Backend?

    func load(directory: URL, backend: Backend, precision: DecisionPrecision, tools: [Tool]) async throws {
        await unload()
        try Task.checkCancellation()
        if backend == .laya {
            laya = try await LayaModel.load(
                from: directory, precision: precision.layaPrecision,
                batchSize: 16, inputPolicy: .reject)
            choices = try await LayaChoiceCatalog.load(directory: directory, tools: tools)
        } else {
            try await retriever.load(directory: directory)
            await retriever.buildIndex(routingTexts: tools.map(\.routingText))
        }
        try Task.checkCancellation()
        self.backend = backend
    }

    func unload() async {
        backend = nil
        laya = nil
        choices = nil
        await retriever.unload()
    }

    func scores(for query: String, tools: [Tool], layaScoring: LayaScoring = .multipleChoice) async throws -> [ToolSearchScore] {
        try Task.checkCancellation()
        guard let backend else {
            throw DecisionError.invalidConfiguration("Load a search model first.")
        }
        if backend == .laya {
            guard let laya else { throw DecisionError.invalidConfiguration("Load Laya first.") }
            if layaScoring == .multipleChoice {
                guard let choices, choices.tools == tools else {
                    throw DecisionError.invalidConfiguration("Load the Laya choice catalog first.")
                }
                return try await choices.scores(query: query, model: laya)
            }
            return try await layaToolScores(query: query, tools: tools, model: laya)
        }
        let scores = await retriever.scores(for: query)
        try Task.checkCancellation()
        return scores.map { ToolSearchScore(score: Double($0)) }
    }
}

/// Bound choice groups using the checkpoint's actual tokenizer and prefix budget.
/// The first round includes full capabilities. Later rounds compare surviving tool
/// labels, retaining five per group until all finalists fit in one shared choice.
/// Final probabilities belong to that shared set, including an explicit no match.
nonisolated struct LayaChoiceCatalog: Sendable {
    static let instructions = "Select the tool most relevant to the request. Choose none if no tool is relevant."
    static let noMatch = DecisionOption(id: "none", description: "None of these tools is relevant to the request.")
    let tools: [Tool]
    private let initialGroups: [[DecisionOption]]
    private let labels: [String: DecisionOption]
    private let labelCosts: [String: Int]
    private let optionBudget: Int

    static func load(directory: URL, tools: [Tool]) async throws -> Self {
        struct Configuration: Decodable {
            let head_max_len: Int
        }
        let configuration = try JSONDecoder().decode(Configuration.self,
            from: Data(contentsOf: directory.appending(path: "rl_agent_config.json")))
        let tokenizer = try await AutoTokenizer.from(modelFolder: directory.appending(path: "tokenizer"))
        return try Self(tools: tools, prefixBudget: configuration.head_max_len) {
            tokenizer.encode(text: $0, addSpecialTokens: false).count
        }
    }

    init(tools: [Tool], prefixBudget: Int = 256, tokenCount: (String) -> Int) throws {
        guard Set(tools.map(\.id)).count == tools.count, !tools.contains(where: { $0.id == Self.noMatch.id }) else {
            throw DecisionError.invalidRequest("Choice tools need unique identifiers distinct from no match.")
        }
        func cost(_ option: DecisionOption) throws -> Int {
            let text = option.description.isEmpty ? option.id : "\(option.id): \(option.description)"
            let tokens = tokenCount(" " + text)
            guard tokens <= 48 else {
                throw DecisionError.capacityExceeded("Choice text for \(option.id) exceeds Laya's 48-token option limit.")
            }
            return 1 + tokens // The option's mask marker.
        }
        let budget = prefixBudget - max(16, tokenCount("choice question: " + Self.instructions)) - (try cost(Self.noMatch))
        guard budget > 0 else { throw DecisionError.invalidConfiguration("The choice prefix budget is too small.") }
        let options = tools.map {
            DecisionOption(id: $0.id, description: $0.description.trimmingCharacters(in: .whitespacesAndNewlines))
        }
        let costs = try Dictionary(uniqueKeysWithValues: options.map { ($0.id, try cost($0)) })
        let labels = tools.map { DecisionOption(id: $0.id, description: "") }
        self.tools = tools
        self.optionBudget = budget
        self.labels = Dictionary(uniqueKeysWithValues: labels.map { ($0.id, $0) })
        self.labelCosts = try Dictionary(uniqueKeysWithValues: labels.map { ($0.id, try cost($0)) })
        self.initialGroups = try Self.groups(options, costs: costs, budget: budget)
    }

    private static func groups(_ options: [DecisionOption], costs: [String: Int], budget: Int) throws -> [[DecisionOption]] {
        var groups: [[DecisionOption]] = []
        var current: [DecisionOption] = []
        var used = 0
        for option in options {
            guard let cost = costs[option.id], cost <= budget else {
                throw DecisionError.capacityExceeded("Choice text for \(option.id) does not fit the checkpoint prefix budget.")
            }
            if current.count == 20 || used + cost > budget {
                groups.append(current)
                current = []
                used = 0
            }
            current.append(option)
            used += cost
        }
        if !current.isEmpty { groups.append(current) }
        return groups
    }

    func request(query: String) -> DecisionRequest { request(query: query, groups: initialGroups) }

    private func request(query: String, groups: [[DecisionOption]]) -> DecisionRequest {
        DecisionRequest(state: query, questions: groups.enumerated().map { index, group in
            DecisionQuestion(id: "group_\(index)", instructions: Self.instructions,
                             kind: .choice(options: group + [Self.noMatch]))
        })
    }

    func scores(query: String, model: any DecisionModel) async throws -> [ToolSearchScore] {
        try Task.checkCancellation()
        guard !tools.isEmpty else { return [] }
        var groups = initialGroups
        var previousCount = tools.count
        var comparingLabels = false
        var diagnostics: [String: DecisionInputDiagnostics] = [:]
        while true {
            let request = request(query: query, groups: groups)
            try request.validate()
            let results = try await model.predict(request)
            try Task.checkCancellation()
            let answers = try Self.validated(results, request: request)
            var survivors: [String] = []
            for (question, answer) in zip(request.questions, answers) {
                guard case .choice(let options) = question.kind,
                      case .choice(_, let probabilities) = answer.answer else {
                    throw DecisionError.invalidRequest("Expected a Laya choice answer.")
                }
                let byID = Dictionary(uniqueKeysWithValues: probabilities.map { ($0.optionID, $0.probability) })
                let candidates = options.filter { $0.id != Self.noMatch.id }
                for candidate in candidates {
                    if let input = answer.inputDiagnostics { diagnostics[candidate.id] = input }
                }
                if groups.count == 1 {
                    return tools.map {
                        ToolSearchScore(score: byID[$0.id] ?? 0, inputDiagnostics: diagnostics[$0.id],
                                        isChoiceProbability: true, isFinalist: byID[$0.id] != nil)
                    }
                }
                survivors += candidates.enumerated().sorted {
                    let lhs = byID[$0.element.id]!, rhs = byID[$1.element.id]!
                    return lhs == rhs ? $0.offset < $1.offset : lhs > rhs
                }.prefix(5).map { $0.element.id }
            }
            guard !comparingLabels || survivors.count < previousCount else {
                throw DecisionError.capacityExceeded("The choice budget cannot reduce this catalog to a shared final ranking.")
            }
            // Catalog order keeps tie handling deterministic across rounds.
            let retained = Set(survivors)
            let options = tools.compactMap { retained.contains($0.id) ? labels[$0.id] : nil }
            groups = try Self.groups(options, costs: labelCosts, budget: optionBudget)
            previousCount = survivors.count
            comparingLabels = true
        }
    }

    private static func validated(_ results: [DecisionResult], request: DecisionRequest) throws -> [DecisionResult] {
        guard results.count == request.questions.count, Set(results.map(\.id)).count == results.count else {
            throw DecisionError.invalidRequest("Laya must return exactly one answer per choice group.")
        }
        let byID = Dictionary(uniqueKeysWithValues: results.map { ($0.id, $0) })
        return try request.questions.map { question in
            guard let result = byID[question.id], case .choice(let options) = question.kind,
                  case .choice(let selectedID, let probabilities) = result.answer else {
                throw DecisionError.invalidRequest("Laya returned an unexpected choice group or answer kind.")
            }
            let identifiers = Set(options.map(\.id))
            guard identifiers.contains(selectedID), probabilities.count == identifiers.count,
                  Set(probabilities.map(\.optionID)) == identifiers else {
                throw DecisionError.invalidRequest("Laya choice probabilities must cover every supplied option exactly once.")
            }
            guard probabilities.allSatisfy({ $0.probability.isFinite && (0...1).contains($0.probability) }),
                  result.confidence.map({ $0.isFinite && (0...1).contains($0) }) != false else {
                throw DecisionError.nonFiniteOutput
            }
            guard abs(probabilities.reduce(0) { $0 + $1.probability } - 1) <= 1e-5 else {
                throw DecisionError.invalidRequest("Laya choice probabilities must sum to one.")
            }
            guard result.inputDiagnostics?.wasTruncated != true else {
                throw DecisionError.capacityExceeded("Laya shortened a choice prompt; reduce the group budget before ranking.")
            }
            return result
        }
    }
}

/// Separate boolean questions let several tools be relevant without competing
/// in a shared choice distribution. Tool capabilities stay in the instructions;
/// the shared state contains only the user's request.
nonisolated func layaRelevanceRequest(query: String, tools: [Tool]) -> DecisionRequest {
    DecisionRequest(state: query, questions: tools.map { tool in
        let description = tool.description.trimmingCharacters(in: .whitespacesAndNewlines)
        let capability = description.isEmpty ? tool.name : description
        return DecisionQuestion(
            id: tool.id,
            instructions: "Does the user's request require the following capability: \(capability)?",
            kind: .boolean)
    })
}

nonisolated func layaToolScores(query: String, tools: [Tool], model: any DecisionModel) async throws -> [ToolSearchScore] {
    try Task.checkCancellation()
    guard !tools.isEmpty else { return [] }
    let request = layaRelevanceRequest(query: query, tools: tools)
    try request.validate()
    let results = try await model.predict(request)
    try Task.checkCancellation()
    guard results.count == tools.count else {
        throw DecisionError.invalidRequest("Laya must return one relevance score for every tool.")
    }
    var byID: [String: ToolSearchScore] = [:]
    let identifiers = Set(tools.map(\.id))
    for result in results {
        guard identifiers.contains(result.id), byID[result.id] == nil,
              case .boolean(let probability) = result.answer else {
            throw DecisionError.invalidRequest("Laya returned an unexpected or duplicate tool relevance answer.")
        }
        guard probability.isFinite, (0...1).contains(probability),
              result.confidence.map({ $0.isFinite && (0...1).contains($0) }) != false else {
            throw DecisionError.nonFiniteOutput
        }
        byID[result.id] = ToolSearchScore(
            score: probability, confidence: result.confidence,
            inputDiagnostics: result.inputDiagnostics)
    }
    return try tools.map { tool in
        guard let score = byID[tool.id] else {
            throw DecisionError.invalidRequest("Laya omitted tool \(tool.id).")
        }
        return score
    }
}

/// Stable catalog order breaks ties. Ranking uses full-precision relevance,
/// never confidence; even confidently irrelevant tools have high confidence.
nonisolated func rankedTools(_ tools: [Tool], scores: [ToolSearchScore], limit: Int = 5) throws -> [SearchResult] {
    guard scores.count == tools.count, scores.allSatisfy({ $0.score.isFinite }) else {
        throw DecisionError.invalidRequest("Search scores must be finite and cover the complete catalog.")
    }
    return tools.indices.filter { scores[$0].isFinalist }.sorted {
        scores[$0].score == scores[$1].score ? $0 < $1 : scores[$0].score > scores[$1].score
    }
    .prefix(min(max(0, limit), 5))
    .enumerated().map { rank, index in
        SearchResult(tool: tools[index], score: scores[index].score, rank: rank + 1,
                     confidence: scores[index].confidence, inputDiagnostics: scores[index].inputDiagnostics,
                     isChoiceProbability: scores[index].isChoiceProbability)
    }
}
