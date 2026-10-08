import Foundation
import PicoDecisions
import PicoDecisionsMLX
import Testing

@testable import SmartToolSelection

struct LayaRetrievalTests {
    @MainActor
    @Test("Laya evaluates the entire catalog with independent boolean questions")
    func fullCatalogRequest() throws {
        let tools = ToolCatalog.load()
        let query = "Find a flight, book it, and add it to my calendar."
        let request = layaRelevanceRequest(query: query, tools: tools)
        try request.validate()
        #expect(request.state == query)
        #expect(request.questions.count == 151)
        #expect(request.questions.map(\.id) == tools.map(\.id))
        for (question, tool) in zip(request.questions, tools) {
            #expect(question.kind == .boolean)
            #expect(question.instructions.contains(tool.description))
        }
    }

    @Test("Tools without descriptions retain their name as the relevance criterion")
    func nameFallback() {
        let tool = Tool(name: "search_products", description: "  ", domain: "test", parameters: [], keywords: [])
        let request = layaRelevanceRequest(query: "Find chairs", tools: [tool])
        #expect(request.questions.first?.instructions.contains("search_products") == true)
    }

    @Test("Rank by relevance, preserving confidence and diagnostics when answers arrive out of order")
    func relevanceRanking() async throws {
        let tools = Self.tools()
        let probabilities = [0.98, 0.02, 0.55, 0.9, 0.8, 0.7]
        let diagnostics = DecisionInputDiagnostics(
            state: .init(originalTokenCount: 12, retainedTokenCount: 12),
            instructions: .init(originalTokenCount: 400, retainedTokenCount: 200), options: [])
        let answers = tools.enumerated().map { index, tool in
            DecisionResult(id: tool.id, answer: .boolean(probabilityTrue: probabilities[index]),
                           confidence: max(probabilities[index], 1 - probabilities[index]),
                           inputDiagnostics: diagnostics)
        }
        let predictor = FakeDecisionModel(results: Array(answers.reversed()))
        let scores = try await layaToolScores(query: "Book a flight", tools: tools, model: predictor)
        let ranked = try rankedTools(tools, scores: scores)
        #expect(scores.map(\.score) == probabilities)
        #expect(ranked.map(\.id) == [0, 3, 4, 5, 2].map { tools[$0].id })
        #expect(ranked.map(\.rank) == [1, 2, 3, 4, 5])
        #expect(!ranked.contains { $0.id == tools[1].id }) // Confidence 0.98, relevance 0.02.
        #expect(ranked[0].confidence == 0.98)
        #expect(ranked.allSatisfy { $0.inputDiagnostics == diagnostics })
        #expect(await predictor.requests.count == 1)
    }

    @Test("Ranking preserves Double precision and uses catalog order on exact ties")
    func precisionAndTies() throws {
        let tools = Self.tools()
        let scores = [0.9, 0.9000000001, 0.9, 0.1, 0.1, 0.1].map { ToolSearchScore(score: $0) }
        #expect(try rankedTools(tools, scores: scores).map(\.id) == [1, 0, 2, 3, 4].map { tools[$0].id })
        #expect(try rankedTools(tools, scores: scores, limit: 2).count == 2)
        #expect(try rankedTools(tools, scores: scores, limit: 0).isEmpty)
    }

    @Test("Malformed or incomplete relevance results fail instead of silently dropping tools",
          arguments: ["missing", "duplicate", "unknown", "choice", "nan", "outside", "confidence"])
    func invalidAnswers(kind: String) async {
        let tools = Array(Self.tools().prefix(2))
        var answers = tools.map { DecisionResult(id: $0.id, answer: .boolean(probabilityTrue: 0.7)) }
        switch kind {
        case "missing": answers.removeLast()
        case "duplicate": answers[1] = answers[0]
        case "unknown": answers[1] = .init(id: "unknown", answer: .boolean(probabilityTrue: 0.7))
        case "choice": answers[1] = .init(id: tools[1].id, answer: .choice(selectedID: "a", probabilities: []))
        case "nan": answers[1] = .init(id: tools[1].id, answer: .boolean(probabilityTrue: .nan))
        case "outside": answers[1] = .init(id: tools[1].id, answer: .boolean(probabilityTrue: 1.1))
        default: answers[1] = .init(id: tools[1].id, answer: .boolean(probabilityTrue: 0.7), confidence: -.infinity)
        }
        let model = FakeDecisionModel(results: answers)
        await #expect(throws: DecisionError.self) {
            try await layaToolScores(query: "Find a flight", tools: tools, model: model)
        }
    }

    @Test("An empty catalog skips inference and mismatched score arrays are rejected")
    func emptyAndMissingScores() async throws {
        let predictor = FakeDecisionModel(results: [])
        #expect(try await layaToolScores(query: "Hello", tools: [], model: predictor).isEmpty)
        #expect(await predictor.requests.isEmpty)
        #expect(throws: DecisionError.self) { try rankedTools(Self.tools(), scores: []) }
    }

    @MainActor
    @Test("Laya searches all tools and returns the same five-card result contract")
    func appFullCatalog() async throws {
        let engine = FakeSearchEngine()
        let model = Self.app(engine: engine)
        model.backend = .laya
        await model.loadIfNeeded().value
        #expect(model.status == .ready)
        await model.search("Find a flight")
        #expect(await engine.loadedBackends == [.laya])
        #expect(await engine.scoredToolIDs == [model.tools.map(\.id)])
        #expect(model.results.count == 5)
        #expect(model.results.first?.id == model.tools.last?.id)
        #expect(model.results.first?.confidence != nil)
        #expect(model.loadLatencyMs != nil)
        #expect(!model.isSearching)
    }

    @MainActor
    @Test("Laya scoring can switch between choice and boolean without reloading weights")
    func scoringSwitch() async {
        let engine = FakeSearchEngine()
        let model = Self.app(engine: engine)
        model.backend = .laya
        await model.loadIfNeeded().value
        await model.search("Find a flight")
        model.layaScoring = .boolean
        await model.search("Find a flight")
        #expect(await engine.scoredMethods == [.multipleChoice, .boolean])
        #expect(await engine.loadedBackends == [.laya])
        #expect(model.results.count == 5)
    }

    @MainActor
    @Test("Clearing or editing the request cancels the full-catalog search without restoring stale results",
          arguments: ["clear", "edit"])
    func cancelObsoleteSearch(action: String) async throws {
        let gate = SearchGate()
        let engine = FakeSearchEngine(firstSearchGate: gate)
        let model = Self.app(engine: engine)
        model.backend = .laya
        await model.loadIfNeeded().value
        let stale = Task { await model.search("Original request") }
        await gate.waitUntilEntered()
        #expect(model.isSearching)
        if action == "clear" { model.clearResults() } else { model.prepareSearch("New request") }
        await gate.release()
        await stale.value
        #expect(model.results.isEmpty)
        #expect(!model.isSearching)
        #expect(model.lastLatencyMs == 0)
        #expect(await engine.firstSearchCancelled)
        await model.search("Replacement request")
        #expect(model.results.count == 5)
    }

    @MainActor
    @Test("Switching backend during Laya inference publishes only the replacement backend")
    func backendSwitch() async throws {
        let gate = SearchGate()
        let engine = FakeSearchEngine(firstSearchGate: gate)
        let model = Self.app(engine: engine)
        model.backend = .laya
        await model.loadIfNeeded().value
        let stale = Task { await model.search("Find a flight") }
        await gate.waitUntilEntered()
        let reload = model.reload(backend: .embedding, quant: .int4)
        #expect(model.results.isEmpty)
        await gate.release()
        await stale.value
        await reload.value
        #expect(model.backend == .embedding)
        #expect(model.status == .ready)
        #expect(model.results.count == 5)
        #expect(model.results.allSatisfy { $0.confidence == nil })
        #expect(await engine.loadedBackends == [.laya, .embedding])
    }

    @MainActor
    @Test("A prediction failure remains recoverable without reloading the model")
    func recoverSearchError() async {
        let engine = FakeSearchEngine(failsFirstSearch: true)
        let model = Self.app(engine: engine)
        await model.loadIfNeeded().value
        await model.search("Too long")
        #expect(model.searchError != nil)
        #expect(model.status == .ready)
        await model.search("Short request")
        #expect(model.searchError == nil)
        #expect(model.results.count == 5)
        #expect(await engine.loadedBackends.count == 1)
    }

    @MainActor
    @Test("An obsolete load cannot overwrite the replacement backend's status")
    func staleLoad() async {
        let gate = SearchGate()
        let engine = FakeSearchEngine(firstLoadGate: gate)
        let model = Self.app(engine: engine)
        model.backend = .laya
        let old = model.loadIfNeeded()
        await gate.waitUntilEntered()
        let replacement = model.reload(backend: .colbert, quant: .int8)
        await gate.release()
        await old.value
        await replacement.value
        #expect(model.status == .ready)
        #expect(model.backend == .colbert)
        await model.search("Replacement")
        #expect(model.results.allSatisfy { $0.confidence == nil })
    }

    @Test("Queued GPU operations stay serialized and a cancelled caller does not execute")
    func queueCancellation() async throws {
        let queue = DeviceInferenceQueue()
        let gate = SearchGate()
        let events = SearchEvents()
        let first = Task { try await queue.run { await gate.wait(); await events.append(1) } }
        await gate.waitUntilEntered()
        let second = Task { try await queue.run { await events.append(2) } }
        second.cancel()
        await gate.release()
        try await first.value
        await #expect(throws: CancellationError.self) { try await second.value }
        #expect(await events.values == [1])
        #expect(try await queue.run { 42 } == 42)
    }

    nonisolated static func tools() -> [Tool] {
        (0..<6).map { Tool(name: "tool_\($0)", description: "Capability \($0)", domain: "test",
                          parameters: [], keywords: []) }
    }

    @MainActor
    private static func app(engine: FakeSearchEngine) -> AppModel {
        AppModel(tools: tools(), engine: engine, download: { _, _, _ in URL(filePath: "/unused-model") })
    }
}

private actor FakeDecisionModel: DecisionModel {
    let results: [DecisionResult]
    private(set) var requests: [DecisionRequest] = []
    init(results: [DecisionResult]) { self.results = results }
    func predict(_ request: DecisionRequest) -> [DecisionResult] {
        requests.append(request)
        return results
    }
}

private actor FakeSearchEngine: ToolSearchEngine {
    let firstSearchGate: SearchGate?
    let firstLoadGate: SearchGate?
    let failsFirstSearch: Bool
    private var backend: Backend?
    private(set) var loadedBackends: [Backend] = []
    private(set) var scoredToolIDs: [[String]] = []
    private(set) var scoredMethods: [LayaScoring] = []
    private(set) var firstSearchCancelled = false

    init(firstSearchGate: SearchGate? = nil, firstLoadGate: SearchGate? = nil, failsFirstSearch: Bool = false) {
        self.firstSearchGate = firstSearchGate
        self.firstLoadGate = firstLoadGate
        self.failsFirstSearch = failsFirstSearch
    }

    func load(directory: URL, backend: Backend, precision: DecisionPrecision, tools: [Tool]) async throws {
        let first = loadedBackends.isEmpty
        loadedBackends.append(backend)
        if first, let firstLoadGate { await firstLoadGate.wait() }
        try Task.checkCancellation()
        self.backend = backend
    }
    func unload() { backend = nil }
    func scores(for query: String, tools: [Tool], layaScoring: LayaScoring) async throws -> [ToolSearchScore] {
        let first = scoredToolIDs.isEmpty
        scoredToolIDs.append(tools.map(\.id))
        scoredMethods.append(layaScoring)
        if first, let firstSearchGate {
            await firstSearchGate.wait()
            firstSearchCancelled = Task.isCancelled
        }
        if first && failsFirstSearch { throw DecisionError.capacityExceeded("Test capacity failure") }
        return tools.indices.map {
            let relevance = Double($0 + 1) / Double(tools.count + 1)
            return ToolSearchScore(score: relevance,
                confidence: backend == .laya ? max(relevance, 1 - relevance) : nil)
        }
    }
}

private actor SearchSignal {
    private var signalled = false
    private var waiters: [CheckedContinuation<Void, Never>] = []
    func signal() {
        signalled = true
        let pending = waiters
        waiters.removeAll()
        for waiter in pending { waiter.resume() }
    }
    func wait() async {
        if signalled { return }
        await withCheckedContinuation { waiters.append($0) }
    }
}

private actor SearchGate {
    private let entered = SearchSignal()
    private let released = SearchSignal()
    func wait() async { await entered.signal(); await released.wait() }
    func waitUntilEntered() async { await entered.wait() }
    func release() async { await released.signal() }
}

private actor SearchEvents {
    private(set) var values: [Int] = []
    func append(_ value: Int) { values.append(value) }
}

/// Opt in using an already-downloaded checkpoint; this test never downloads weights.
@Suite(.serialized, .enabled(if: ProcessInfo.processInfo.environment["SMART_TOOL_SELECTION_LAYA_MODEL"] != nil))
struct LayaCatalogSmokeTests {
    @MainActor
    @Test("The real Laya backend scores all 151 tools and returns finite relevance/confidence")
    func fullCatalogInference() async throws {
        let directory = URL(filePath: ProcessInfo.processInfo.environment["SMART_TOOL_SELECTION_LAYA_MODEL"]!)
        let engine = CatalogSearchEngine()
        let tools = ToolCatalog.load()
        print("LAYA_CACHE_ROOT \(URL.applicationSupportDirectory.appending(path: "SmartToolSelection/models").path)")
        try await engine.load(directory: directory, backend: .laya, precision: .float16, tools: tools)
        for query in ["show me cheap blue outdoor chairs", "Find my order; do not cancel it", "Find a flight, book it, and add it to my calendar"] {
            let started = ContinuousClock.now
            let scores = try await engine.scores(for: query, tools: tools, layaScoring: .boolean)
            let duration = started.duration(to: .now).components
            let milliseconds = Double(duration.seconds) * 1_000 + Double(duration.attoseconds) / 1e15
            #expect(scores.count == 151)
            #expect(scores.allSatisfy { $0.score.isFinite && (0...1).contains($0.score) })
            #expect(scores.allSatisfy { $0.confidence == max($0.score, 1 - $0.score) })
            let ranked = try rankedTools(tools, scores: scores)
            #expect(ranked.count == 5)
            print("LAYA_CATALOG \(query) | \(milliseconds) ms | \(ranked.map { "\($0.tool.name)=\($0.score)" }.joined(separator: ", ")) | truncated=\(scores.filter { $0.inputDiagnostics?.wasTruncated == true }.count)")
        }
        await engine.unload()
    }
}
