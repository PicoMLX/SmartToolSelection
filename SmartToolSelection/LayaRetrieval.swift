import Foundation
import PicoDecisions
import PicoDecisionsMLX

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

/// A score over the full tool catalog. Confidence describes certainty in either
/// relevance or irrelevance; only score determines the shortlist order.
nonisolated struct ToolSearchScore: Sendable {
    let score: Double
    var confidence: Double? = nil
    var inputDiagnostics: DecisionInputDiagnostics? = nil
}

nonisolated protocol ToolSearchEngine: Sendable {
    func load(directory: URL, backend: Backend, precision: DecisionPrecision, tools: [Tool]) async throws
    func unload() async
    func scores(for query: String, tools: [Tool]) async throws -> [ToolSearchScore]
}

/// The app loads one backend at a time. DeviceInferenceQueue serializes complete
/// load/search/unload operations even when this actor suspends.
actor CatalogSearchEngine: ToolSearchEngine {
    private let retriever = RetrievalEngine()
    private var laya: LayaModel?
    private var backend: Backend?

    func load(directory: URL, backend: Backend, precision: DecisionPrecision, tools: [Tool]) async throws {
        await unload()
        try Task.checkCancellation()
        if backend == .laya {
            laya = try await LayaModel.load(
                from: directory, precision: precision.layaPrecision,
                batchSize: 16, inputPolicy: .reject)
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
        await retriever.unload()
    }

    func scores(for query: String, tools: [Tool]) async throws -> [ToolSearchScore] {
        try Task.checkCancellation()
        guard let backend else {
            throw DecisionError.invalidConfiguration("Load a search model first.")
        }
        if backend == .laya {
            guard let laya else { throw DecisionError.invalidConfiguration("Load Laya first.") }
            return try await layaToolScores(query: query, tools: tools, model: laya)
        }
        let scores = await retriever.scores(for: query)
        try Task.checkCancellation()
        return scores.map { ToolSearchScore(score: Double($0)) }
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
    return tools.indices.sorted {
        scores[$0].score == scores[$1].score ? $0 < $1 : scores[$0].score > scores[$1].score
    }
    .prefix(min(max(0, limit), 5))
    .enumerated().map { rank, index in
        SearchResult(tool: tools[index], score: scores[index].score, rank: rank + 1,
                     confidence: scores[index].confidence, inputDiagnostics: scores[index].inputDiagnostics)
    }
}
