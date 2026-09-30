import Foundation
import PicoDecisions
import Testing

@testable import SmartToolSelection

struct DecisionRoutingTests {
    @MainActor
    @Test("Decision candidates preserve identity, scores, and full capabilities without retrieval expansion")
    func candidatesUseCapabilityDescriptions() throws {
        let tool = Tool(
            name: "cancel_order", description: "Cancel an order only before it ships.",
            domain: "ecommerce",
            parameters: [
                ToolParameter(name: "order_id", type: "string", description: "Order identifier",
                              enumValues: nil, required: true, itemsType: nil, itemsEnum: nil),
                ToolParameter(name: "reason", type: "string", description: "Optional reason",
                              enumValues: ["duplicate"], required: false, itemsType: nil, itemsEnum: nil)
            ], keywords: ["refund", "duplicate purchase"])
        let candidates = decisionCandidates(from: [SearchResult(tool: tool, score: 0.625, rank: 1)])
        let candidate = try #require(candidates.first)

        #expect(candidate.id == "ecommerce|cancel_order")
        #expect(candidate.retrievalScore == 0.625)
        #expect(candidate.description == "cancel_order: Cancel an order only before it ships. Required arguments: order_id.")
        #expect(!candidate.description.contains("refund"))
        #expect(!candidate.description.contains("duplicate"))
        #expect(!candidate.description.contains("reason"))
    }

    @MainActor
    @Test("Changing acceptance gates retains the raw recommendation without another inference")
    func policyAbstentionRetainsRecommendation() async throws {
        let predictor = RoutingPredictor(responses: [.init(probability: 0.6)])
        let model = try await loadedModel(predictor: predictor)
        let candidates = [result()]
        await model.route(query: "Cancel this order", candidates: candidates)
        let raw = try #require(model.recommendation)

        model.minimumProbability = 0.8
        model.minimumMargin = 0.3
        let abstention = try #require(model.disposition)
        #expect(abstention.status == .abstained)
        #expect(abstention.reasons == [.belowMinimumProbability, .belowMinimumMargin])
        #expect(abstention.selection == raw.selection)
        #expect(model.recommendation?.selection.selectedCandidateID == candidates[0].id)

        model.minimumProbability = 0.5
        model.minimumMargin = 0.1
        #expect(model.disposition?.status == .acceptedTool)
        #expect(model.recommendation?.selection == raw.selection)
        #expect(await predictor.requestCount == 1)
    }

    @MainActor
    @Test("Accepted no match remains distinct from an abstained no-match recommendation")
    func noMatchAndAbstentionAreDistinct() async throws {
        let predictor = RoutingPredictor(responses: [.init(probability: 0.9, choosesNoMatch: true)])
        let model = try await loadedModel(predictor: predictor)
        await model.route(query: "Do nothing", candidates: [result()])
        let raw = try #require(model.recommendation)
        #expect(raw.selection.selectedCandidateID == nil)
        #expect(model.disposition?.status == .acceptedNoMatch)

        model.minimumProbability = 0.95
        #expect(model.disposition?.status == .abstained)
        #expect(model.disposition?.reasons == [.belowMinimumProbability])
        #expect(model.disposition?.selection == raw.selection)
        #expect(await predictor.requestCount == 1)
    }

    @MainActor
    @Test("Truncation details survive selection, recommendation, and acceptance policy")
    func diagnosticsPropagate() async throws {
        let candidate = result()
        let diagnostics = DecisionInputDiagnostics(
            state: .init(originalTokenCount: 12, retainedTokenCount: 12),
            instructions: .init(originalTokenCount: 40, retainedTokenCount: 25),
            options: [
                .init(optionID: candidate.id,
                      tokens: .init(originalTokenCount: 80, retainedTokenCount: 32)),
                .init(optionID: ToolDecisionSelector.noMatchID,
                      tokens: .init(originalTokenCount: 20, retainedTokenCount: 20))
            ])
        let predictor = RoutingPredictor(responses: [.init(probability: 0.9, diagnostics: diagnostics)])
        let model = try await loadedModel(predictor: predictor)
        await model.route(query: "Cancel this order", candidates: [candidate])

        let recommendation = try #require(model.recommendation)
        #expect(recommendation.selection.inputTokenCount == 123)
        #expect(recommendation.selection.inputDiagnostics == diagnostics)
        #expect(model.disposition?.selection.inputDiagnostics == diagnostics)
        #expect(model.recommendation?.selection.inputDiagnostics?.wasTruncated == true)
    }

    @MainActor
    @Test("A stale completion cannot restore a recommendation after invalidation",
          arguments: ["new query", "same query", "new precision"])
    func invalidationDropsStaleCompletion(change: String) async throws {
        let gate = RoutingGate()
        let predictor = RoutingPredictor(
            responses: [.init(probability: 0.6), .init(probability: 0.95)], firstPredictionGate: gate)
        let model = try await loadedModel(predictor: predictor)
        let candidates = [result()]
        let stale = Task { await model.route(query: "Original request", candidates: candidates) }
        await gate.waitUntilEntered()
        #expect(model.isSelecting)

        if change == "new precision" {
            model.precision = .float32
            model.invalidateConfiguration()
        } else {
            model.invalidateSearch()
        }
        #expect(model.recommendation == nil)
        await gate.release()
        await stale.value
        #expect(model.recommendation == nil)
        #expect(!model.isSelecting)

        if change == "new precision" { await model.loadIfNeeded() }
        let currentQuery = change == "new query" ? "Replacement request" : "Original request"
        await model.route(query: currentQuery, candidates: candidates)
        let current = try #require(model.recommendation)
        #expect(current.query == currentQuery)
        #expect(current.selection.candidateProbabilities.first?.probability == 0.95)
        #expect(current.precision == (change == "new precision" ? .float32 : .float16))
        #expect(await predictor.requestCount == 2)
    }

    @MainActor
    @Test("A failed request clears its recommendation and permits the next request")
    func inferenceFailureCanRecover() async throws {
        let predictor = RoutingPredictor(responses: [
            .init(probability: 0.9), .init(probability: 0.9, fails: true), .init(probability: 0.95)
        ])
        let model = try await loadedModel(predictor: predictor)
        await model.route(query: "Initial request", candidates: [result()])
        #expect(model.recommendation != nil)
        await model.route(query: "Failing request", candidates: [result()])

        #expect(model.recommendation == nil)
        #expect(model.routingError != nil)
        #expect(model.status == .ready)
        #expect(!model.isSelecting)

        model.invalidateSearch()
        await model.route(query: "Replacement request", candidates: [result()])
        #expect(model.recommendation?.query == "Replacement request")
        #expect(model.routingError == nil)
        #expect(model.status == .ready)
    }

    @MainActor
    @Test("Load failures report failed status and do not enable inference")
    func loadFailureKeepsRecommendationEmpty() async {
        let predictor = RoutingPredictor(responses: [.init(probability: 0.9)])
        let model = DecisionRoutingModel(
            queue: DeviceInferenceQueue(), engine: RoutingEngine(predictor: predictor, failsLoad: true),
            download: { _ in URL(fileURLWithPath: "/unused-model") })
        model.enabled = true
        await model.loadIfNeeded()

        if case .failed = model.status {} else { Issue.record("Expected load failure") }
        await model.route(query: "Cancel this order", candidates: [result()])
        #expect(model.recommendation == nil)
        #expect(await predictor.requestCount == 0)
    }

    @MainActor
    @Test("Empty candidates return no match without running the prediction model")
    func emptyCandidatesSkipInference() async throws {
        let predictor = RoutingPredictor(responses: [.init(probability: 0.9)])
        let model = try await loadedModel(predictor: predictor)
        await model.route(query: "No retrieved tools", candidates: [])

        let selection = try #require(model.recommendation?.selection)
        #expect(selection.selectedCandidateID == nil)
        #expect(selection.candidateProbabilities.isEmpty)
        #expect(selection.noMatchProbability == nil)
        #expect(selection.inputDiagnostics == nil)
        #expect(model.disposition?.status == .acceptedNoMatch)
        #expect(await predictor.requestCount == 0)
    }

    @Test("Device inference operations remain ordered across a suspension")
    func deviceQueueSerializesSuspendedOperations() async throws {
        let queue = DeviceInferenceQueue()
        let firstGate = RoutingGate()
        let secondArrival = RoutingSignal()
        let events = RoutingEvents()
        let first = Task {
            try await queue.run {
                await events.append("first started")
                await firstGate.wait()
                await events.append("first finished")
                return 1
            }
        }
        await firstGate.waitUntilEntered()
        let second = Task {
            await secondArrival.signal()
            return try await queue.run {
                await events.append("second started")
                return 2
            }
        }
        await secondArrival.wait()
        await firstGate.release()

        #expect(try await first.value == 1)
        #expect(try await second.value == 2)
        #expect(await events.values == ["first started", "first finished", "second started"])
    }

    @Test("Cancelling a waiting inference caller prevents its operation from executing")
    func deviceQueuePropagatesQueuedCancellation() async throws {
        let queue = DeviceInferenceQueue()
        let firstGate = RoutingGate()
        let secondArrival = RoutingSignal()
        let events = RoutingEvents()
        let first = Task {
            try await queue.run { await firstGate.wait() }
        }
        await firstGate.waitUntilEntered()
        let queued = Task {
            await secondArrival.signal()
            try await queue.run { await events.append("cancelled operation executed") }
        }
        await secondArrival.wait()
        queued.cancel()
        await firstGate.release()
        try await first.value
        await #expect(throws: CancellationError.self) { try await queued.value }
        #expect(await events.values.isEmpty)

        // A cancelled tail must still allow future inference to progress.
        let recovered = try await queue.run { 42 }
        #expect(recovered == 42)
    }

    @MainActor
    private func result() -> SearchResult {
        SearchResult(tool: Tool(name: "cancel_order", description: "Cancel an unshipped order.",
                                domain: "ecommerce", parameters: [], keywords: []), score: 0.75, rank: 1)
    }

    @MainActor
    private func loadedModel(predictor: RoutingPredictor) async throws -> DecisionRoutingModel {
        let model = DecisionRoutingModel(
            queue: DeviceInferenceQueue(), engine: RoutingEngine(predictor: predictor),
            download: { _ in URL(fileURLWithPath: "/unused-model") })
        model.enabled = true
        await model.loadIfNeeded()
        try #require(model.status == .ready)
        return model
    }
}

private enum RoutingFailure: Error { case load, prediction }

private actor RoutingEngine: ToolRoutingDecisionEngine {
    let predictor: RoutingPredictor
    let failsLoad: Bool

    init(predictor: RoutingPredictor, failsLoad: Bool = false) {
        self.predictor = predictor
        self.failsLoad = failsLoad
    }

    func load(directory: URL, precision: DecisionPrecision) throws {
        if failsLoad { throw RoutingFailure.load }
    }

    func unload() {}

    func select(query: String, candidates: [ToolDecisionCandidate]) async throws -> ToolDecisionSelection {
        try await ToolDecisionSelector(model: predictor, maximumCandidates: 5)
            .select(query: query, candidates: candidates)
    }
}

private actor RoutingPredictor: DecisionModel {
    struct Response: Sendable {
        var probability: Double
        var choosesNoMatch = false
        var diagnostics: DecisionInputDiagnostics? = nil
        var fails = false
    }

    private let responses: [Response]
    private let firstPredictionGate: RoutingGate?
    private(set) var requestCount = 0

    init(responses: [Response], firstPredictionGate: RoutingGate? = nil) {
        self.responses = responses
        self.firstPredictionGate = firstPredictionGate
    }

    func predict(_ request: DecisionRequest) async throws -> [DecisionResult] {
        let index = requestCount
        requestCount += 1
        let response = responses[min(index, responses.count - 1)]
        if index == 0, let firstPredictionGate { await firstPredictionGate.wait() }
        if response.fails { throw RoutingFailure.prediction }
        guard let question = request.questions.first,
              case .choice(let options) = question.kind,
              let firstOption = options.first else { throw RoutingFailure.prediction }
        let selectedID = response.choosesNoMatch ? ToolDecisionSelector.noMatchID : firstOption.id
        let probabilities = options.map { option in
            OptionProbability(optionID: option.id, probability: option.id == selectedID
                ? response.probability : (1 - response.probability) / Double(options.count - 1))
        }
        return [DecisionResult(
            id: question.id, answer: .choice(selectedID: selectedID, probabilities: probabilities),
            confidence: 0.7, actProbability: 0.9, inputTokenCount: 123,
            inputDiagnostics: response.diagnostics)]
    }
}

/// One-shot signals keep suspended operations deterministic without timers.
private actor RoutingSignal {
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

private actor RoutingGate {
    private let entered = RoutingSignal()
    private let released = RoutingSignal()

    func wait() async {
        await entered.signal()
        await released.wait()
    }

    func waitUntilEntered() async { await entered.wait() }
    func release() async { await released.signal() }
}

private actor RoutingEvents {
    private(set) var values: [String] = []
    func append(_ value: String) { values.append(value) }
}
