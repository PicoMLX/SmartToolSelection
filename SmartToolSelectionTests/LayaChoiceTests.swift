import Foundation
import PicoDecisions
import PicoDecisionsMLX
import Testing

@testable import SmartToolSelection

struct LayaChoiceTests {
    @MainActor
    @Test("Choice groups cover the full catalog and preserve descriptions")
    func fullCatalog() throws {
        let tools = ToolCatalog.load()
        let catalog = try LayaChoiceCatalog(tools: tools) { $0.split(separator: " ").count }
        let request = catalog.request(query: "Find chairs")
        try request.validate()
        #expect(request.state == "Find chairs")
        var options: [DecisionOption] = []
        for question in request.questions {
            guard case .choice(let choices) = question.kind else { Issue.record("Expected choice"); continue }
            #expect(choices.count <= 21)
            #expect(choices.last == LayaChoiceCatalog.noMatch)
            options += choices.dropLast()
        }
        #expect(options.map(\.id) == tools.map(\.id))
        #expect(options.map(\.description) == tools.map(\.description))
    }

    @Test("A shared final choice ranks survivors without mixing group probabilities")
    func sharedRanking() async throws {
        let tools = Self.tools(41)
        let catalog = try LayaChoiceCatalog(tools: tools) { _ in 1 }
        let weights = Dictionary(uniqueKeysWithValues: tools.enumerated().map { ($0.element.id, Double($0.offset + 1)) })
        let predictor = ChoicePredictor(weights: weights)
        let scores = try await catalog.scores(query: "Find a tool", model: predictor)
        #expect(scores.count == tools.count)
        #expect(scores.allSatisfy { $0.confidence == nil && $0.isChoiceProbability })
        let ranked = try rankedTools(tools, scores: scores)
        #expect(ranked.map(\.id) == tools.suffix(5).reversed().map(\.id))
        #expect(ranked.allSatisfy { $0.isChoiceProbability })
        let requests = await predictor.requests
        #expect(requests.map { $0.questions.count } == [3, 1])
        guard case .choice(let final) = requests.last!.questions[0].kind else { return }
        #expect(final.count == 12) // 5 + 5 + 1 finalists, plus no match.
        #expect(final.dropLast().allSatisfy { $0.description.isEmpty })
        let denominator = 1 + final.dropLast().reduce(0.0) { $0 + weights[$1.id]! }
        #expect(abs(ranked[0].score - 41 / denominator) < 1e-12)
    }

    @Test("Zero-probability ties never restore tools eliminated in earlier rounds")
    func zeroFinalProbabilities() async throws {
        let tools = Self.tools(41)
        let catalog = try LayaChoiceCatalog(tools: tools) { _ in 1 }
        let model = ChoicePredictor(weights: Dictionary(uniqueKeysWithValues: tools.map { ($0.id, 1.0) }), zeroFinal: true)
        let scores = try await catalog.scores(query: "No matching tool", model: model)
        let ranked = try rankedTools(tools, scores: scores)
        #expect(ranked.count == 5)
        #expect(ranked.map(\.id) == tools.prefix(5).map(\.id))
        #expect(scores.filter(\.isFinalist).count == 11)
        #expect(scores.allSatisfy { $0.score == 0 })
    }

    @Test("Small catalogs use one choice round; empty catalogs skip inference")
    func smallAndEmpty() async throws {
        let model = ChoicePredictor()
        let small = try LayaChoiceCatalog(tools: Self.tools(3)) { _ in 1 }
        #expect(try await small.scores(query: "Find a tool", model: model).count == 3)
        let empty = try LayaChoiceCatalog(tools: []) { _ in 1 }
        #expect(try await empty.scores(query: "Hello", model: model).isEmpty)
        #expect(await model.requests.count == 1)
    }

    @Test("Group sizing reserves instructions, no match, and every option marker")
    func tokenBudget() async throws {
        let catalog = try LayaChoiceCatalog(tools: Self.tools(8), prefixBudget: 32) {
            $0.contains("Capability") ? 8 : 1
        }
        let request = catalog.request(query: "Find a tool")
        #expect(request.questions.count == 8) // Full descriptions fit one at a time.
        let predictor = ChoicePredictor()
        #expect(try await catalog.scores(query: "Find a tool", model: predictor).count == 8)
        #expect(await predictor.requests.map { $0.questions.count } == [8, 2, 1])
    }

    @Test("Capacity and malformed catalog inputs are rejected")
    func capacity() throws {
        #expect(throws: DecisionError.self) { try LayaChoiceCatalog(tools: Self.tools(1)) { _ in 49 } }
        #expect(throws: DecisionError.self) { try LayaChoiceCatalog(tools: Self.tools(1), prefixBudget: 10) { _ in 1 } }
        #expect(throws: DecisionError.self) { try LayaChoiceCatalog(tools: Self.tools(1) + Self.tools(1)) { _ in 1 } }
    }

    @Test("Choice output validation rejects missing, duplicate, nonfinite, and shortened answers",
          arguments: ["missing", "question", "kind", "option", "duplicate", "nan", "outside", "sum", "selected", "confidence", "truncated"])
    func malformed(kind: String) async throws {
        let catalog = try LayaChoiceCatalog(tools: Self.tools(3)) { _ in 1 }
        await #expect(throws: DecisionError.self) {
            try await catalog.scores(query: "Find a tool", model: ChoicePredictor(malformed: kind))
        }
    }

    @Test("Cancellation before inference skips the model")
    func cancellation() async throws {
        let catalog = try LayaChoiceCatalog(tools: Self.tools(3)) { _ in 1 }
        let model = ChoicePredictor()
        let signal = AsyncStream<Void>.makeStream()
        let task = Task {
            for await _ in signal.stream { break }
            return try await catalog.scores(query: "Find a tool", model: model)
        }
        task.cancel()
        signal.continuation.finish()
        await #expect(throws: CancellationError.self) { try await task.value }
        #expect(await model.requests.isEmpty)
    }

    nonisolated static func tools(_ count: Int) -> [Tool] {
        (0..<count).map { Tool(name: "tool_\($0)", description: "Capability \($0)", domain: "test", parameters: [], keywords: []) }
    }
}

private actor ChoicePredictor: DecisionModel {
    private(set) var requests: [DecisionRequest] = []
    let weights: [String: Double]
    let malformed: String?
    let zeroFinal: Bool
    init(weights: [String: Double] = [:], malformed: String? = nil, zeroFinal: Bool = false) {
        self.weights = weights; self.malformed = malformed; self.zeroFinal = zeroFinal
    }
    func predict(_ request: DecisionRequest) throws -> [DecisionResult] {
        requests.append(request)
        if malformed == "missing" { return [] }
        return request.questions.reversed().map { question in
            guard case .choice(let options) = question.kind else { preconditionFailure() }
            let values = options.map { option in
                if zeroFinal && requests.count > 1 { return option.id == "none" ? 1.0 : 0.0 }
                return weights[option.id] ?? 1
            }
            let total = values.reduce(0, +)
            var probabilities = zip(options, values).map { OptionProbability(optionID: $0.0.id, probability: $0.1 / total) }
            switch malformed {
            case "option": probabilities[0] = .init(optionID: "unknown", probability: probabilities[0].probability)
            case "duplicate": probabilities[1] = probabilities[0]
            case "nan": probabilities[0] = .init(optionID: options[0].id, probability: .nan)
            case "outside": probabilities[0] = .init(optionID: options[0].id, probability: 1.1)
            case "sum": probabilities[0] = .init(optionID: options[0].id, probability: 0)
            default: break
            }
            let selected = probabilities.max { $0.probability < $1.probability }!.optionID
            let diagnostics: DecisionInputDiagnostics? = malformed == "truncated"
                ? .init(state: .init(originalTokenCount: 2, retainedTokenCount: 1), instructions: .init(originalTokenCount: 2, retainedTokenCount: 2), options: []) : nil
            return DecisionResult(id: malformed == "question" ? "unknown" : question.id,
                answer: malformed == "kind" ? .boolean(probabilityTrue: 0.5)
                    : .choice(selectedID: malformed == "selected" ? "unknown" : selected, probabilities: probabilities),
                confidence: malformed == "confidence" ? .infinity : 0.7, inputDiagnostics: diagnostics)
        }
    }
}

/// Run separately from other GPU suites with a cached checkpoint. The cases are
/// development smoke examples, not a held-out accuracy benchmark.
@Suite(.serialized, .enabled(if: ProcessInfo.processInfo.environment["SMART_TOOL_SELECTION_LAYA_BENCHMARK"] != nil))
struct LayaChoiceBenchmarkTests {
    @MainActor
    @Test func compareScoringMethods() async throws {
        let directory = URL(filePath: ProcessInfo.processInfo.environment["SMART_TOOL_SELECTION_LAYA_BENCHMARK"]!)
        let tools = ToolCatalog.load()
        let choices = try await LayaChoiceCatalog.load(directory: directory, tools: tools)
        let laya = try await LayaModel.load(from: directory, precision: .float16, batchSize: 16)
        let model = TracedDecisionModel(model: laya)
        let cases: [(String, [String])] = [
            ("show me cheap blue outdoor chairs", ["ecommerce|search_products"]),
            ("Find my order; do not cancel it", ["ecommerce|track_shipment"]),
            ("Find a flight, book it, and add it to my calendar", ["travel|search_flights", "travel|book_flight"]),
            ("spin up a new postgres database on aws", ["devops|provision_resource"]),
            ("find me a business class flight to tokyo", ["travel|search_flights"]),
            ("there's a charge I didn't make, dispute it", ["finance|dispute_transaction"]),
            ("book a telehealth visit with a dermatologist", ["healthcare|book_appointment"]),
            ("bump this ticket up to a manager", ["support|escalate_ticket"]),
            ("request a week of vacation", ["workplace|submit_pto_request"]),
            ("where is my delivery right now", ["ecommerce|track_shipment"])
        ]
        var report: [[String: Any]] = []
        for (query, expected) in cases {
            for method in LayaScoring.allCases { _ = try await score(method, query: query, tools: tools, choices: choices, model: model) }
            for round in 0..<3 {
                let methods = round.isMultiple(of: 2) ? LayaScoring.allCases : LayaScoring.allCases.reversed().map { $0 }
                for method in methods {
                    await model.reset()
                    let started = ContinuousClock.now
                    let scores = try await score(method, query: query, tools: tools, choices: choices, model: model)
                    let ranked = try rankedTools(tools, scores: scores)
                    let duration = started.duration(to: .now).components
                    let milliseconds = Double(duration.seconds) * 1000 + Double(duration.attoseconds) / 1e15
                    let trace = await model.trace
                    #expect(scores.count == 151)
                    #expect(scores.allSatisfy { $0.score.isFinite && (0...1).contains($0.score) })
                    #expect(ranked.count == 5)
                    #expect(trace.truncated == 0)
                    let ids = ranked.map(\.id)
                    let entry: [String: Any] = ["method": method.rawValue, "query": query, "round": round,
                        "milliseconds": milliseconds, "topFive": ids, "expected": expected,
                        "expectedHits": expected.filter { ids.contains($0) }.count,
                        "roundQuestionCounts": trace.questions, "truncated": trace.truncated,
                        "totalInputTokens": trace.tokens]
                    report.append(entry)
                    print("LAYA_CHOICE_BENCHMARK \(String(data: try JSONSerialization.data(withJSONObject: entry, options: [.sortedKeys]), encoding: .utf8)!)")
                }
            }
        }
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            .write(to: URL(filePath: "/tmp/smarttool-laya-choice-benchmark.json"))
    }

    private func score(_ method: LayaScoring, query: String, tools: [Tool], choices: LayaChoiceCatalog,
                       model: any DecisionModel) async throws -> [ToolSearchScore] {
        if method == .multipleChoice { return try await choices.scores(query: query, model: model) }
        return try await layaToolScores(query: query, tools: tools, model: model)
    }
}

private actor TracedDecisionModel: DecisionModel {
    struct Trace: Sendable { var questions: [Int] = []; var tokens = 0; var truncated = 0 }
    let model: any DecisionModel
    private(set) var trace = Trace()
    init(model: any DecisionModel) { self.model = model }
    func reset() { trace = Trace() }
    func predict(_ request: DecisionRequest) async throws -> [DecisionResult] {
        let results = try await model.predict(request)
        trace.questions.append(request.questions.count)
        trace.tokens += results.reduce(0) { $0 + ($1.inputTokenCount ?? 0) }
        trace.truncated += results.filter { $0.inputDiagnostics?.wasTruncated == true }.count
        return results
    }
}
