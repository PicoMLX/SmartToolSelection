import Foundation
import Observation
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

nonisolated protocol ToolRoutingDecisionEngine: Sendable {
    func load(directory: URL, precision: DecisionPrecision) async throws
    func unload() async
    func select(query: String, candidates: [ToolDecisionCandidate]) async throws -> ToolDecisionSelection
}

actor LayaRoutingEngine: ToolRoutingDecisionEngine {
    private var model: LayaModel?

    func load(directory: URL, precision: DecisionPrecision) async throws {
        // Release the old precision before allocating the replacement.
        model = nil
        model = try await LayaModel.load(
            from: directory, precision: precision.layaPrecision,
            batchSize: 1, inputPolicy: .reject)
    }

    func unload() { model = nil }

    func select(query: String, candidates: [ToolDecisionCandidate]) async throws -> ToolDecisionSelection {
        guard let model else { throw DecisionError.invalidConfiguration("Load the decision model first.") }
        return try await ToolDecisionSelector(model: model, maximumCandidates: 5)
            .select(query: query, candidates: candidates)
    }
}

struct RoutingRecommendation: Sendable {
    let query: String
    let candidates: [SearchResult]
    let selection: ToolDecisionSelection
    let latencyMilliseconds: Double
    let precision: DecisionPrecision
}

/// Keep retrieval vocabulary expansion out of the bounded decision prompt.
/// Required argument names describe capabilities; routing does not fill them in.
func decisionCandidates(from results: [SearchResult]) -> [ToolDecisionCandidate] {
    results.map { result in
        let tool = result.tool
        let required = tool.parameters.filter(\.required).map(\.name)
        var description = "\(tool.name): \(tool.description)"
        if !required.isEmpty {
            description += " Required arguments: " + required.joined(separator: ", ") + "."
        }
        return ToolDecisionCandidate(
            id: tool.id, description: description, retrievalScore: Double(result.score))
    }
}

@MainActor
@Observable
final class DecisionRoutingModel {
    enum Status: Equatable {
        case idle
        case loading(String)
        case ready
        case failed(String)
    }

    var enabled = false
    var precision: DecisionPrecision = .float16
    // Start with both gates disabled. Device experiments determine useful values.
    var minimumProbability = 0.0
    var minimumMargin = 0.0
    private(set) var status: Status = .idle
    private(set) var isSelecting = false
    private(set) var routingError: String?
    private(set) var loadLatencyMilliseconds: Double?
    private(set) var recommendation: RoutingRecommendation?

    @ObservationIgnored private let engine: any ToolRoutingDecisionEngine
    @ObservationIgnored private let queue: DeviceInferenceQueue
    @ObservationIgnored private let download: @Sendable (@escaping @Sendable (Double) -> Void) async throws -> URL
    @ObservationIgnored private var loadedPrecision: DecisionPrecision?
    @ObservationIgnored private var configurationGeneration = UUID()
    @ObservationIgnored private var searchGeneration = UUID()

    init(
        queue: DeviceInferenceQueue,
        engine: (any ToolRoutingDecisionEngine)? = nil,
        download: @escaping @Sendable (@escaping @Sendable (Double) -> Void) async throws -> URL = {
            try await ModelDownloader.downloadLaya(progress: $0)
        }
    ) {
        self.queue = queue
        self.engine = engine ?? LayaRoutingEngine()
        self.download = download
    }

    var disposition: ToolDecisionDisposition? {
        guard let recommendation,
              let policy = try? ToolDecisionAcceptancePolicy(
                minimumProbability: minimumProbability, minimumMargin: minimumMargin)
        else { return nil }
        return policy.evaluate(recommendation.selection)
    }

    func invalidateSearch() {
        searchGeneration = UUID()
        recommendation = nil
        routingError = nil
        isSelecting = false
    }

    func invalidateConfiguration() {
        configurationGeneration = UUID()
        invalidateSearch()
    }

    func loadIfNeeded() async {
        let generation = configurationGeneration
        let requestedPrecision = precision
        let engine = self.engine
        if !enabled {
            loadedPrecision = nil
            loadLatencyMilliseconds = nil
            status = .idle
            _ = try? await queue.run { await engine.unload() }
            return
        }
        if loadedPrecision == requestedPrecision, case .ready = status { return }
        status = .loading("Downloading Laya multilingual…")
        do {
            let directory = try await download { [weak self] fraction in
                Task { @MainActor [weak self] in
                    guard let self, generation == self.configurationGeneration,
                          case .loading(let message) = self.status,
                          message.hasPrefix("Downloading") else { return }
                    let percent = Int(max(0, min(1, fraction)) * 100)
                    self.status = .loading("Downloading Laya multilingual… \(percent)%")
                }
            }
            try Task.checkCancellation()
            guard generation == configurationGeneration, enabled else { return }
            status = .loading("Loading Laya multilingual (\(requestedPrecision.title))…")
            let milliseconds = try await queue.run {
                let clock = ContinuousClock()
                let started = clock.now
                try await engine.load(directory: directory, precision: requestedPrecision)
                return Self.milliseconds(started.duration(to: clock.now))
            }
            try Task.checkCancellation()
            guard generation == configurationGeneration, enabled else { return }
            loadLatencyMilliseconds = milliseconds
            loadedPrecision = requestedPrecision
            status = .ready
        } catch is CancellationError {
            // A new configuration owns the status, or a later retry will reload.
            if generation == configurationGeneration { status = .idle }
        } catch {
            guard generation == configurationGeneration else { return }
            loadedPrecision = nil
            status = .failed(error.localizedDescription)
        }
    }

    func route(query: String, candidates: [SearchResult]) async {
        guard enabled, loadedPrecision == precision, case .ready = status else { return }
        let search = searchGeneration
        let configuration = configurationGeneration
        let requestedPrecision = precision
        let engine = self.engine
        let options = decisionCandidates(from: candidates)
        recommendation = nil
        routingError = nil
        isSelecting = true
        defer {
            if search == searchGeneration, configuration == configurationGeneration { isSelecting = false }
        }
        do {
            // Measure completed inference only; queue waiting is not model latency.
            let (selection, milliseconds) = try await queue.run {
                let clock = ContinuousClock()
                let started = clock.now
                let selection = try await engine.select(query: query, candidates: options)
                return (selection, Self.milliseconds(started.duration(to: clock.now)))
            }
            try Task.checkCancellation()
            guard search == searchGeneration, configuration == configurationGeneration, enabled else { return }
            recommendation = RoutingRecommendation(
                query: query, candidates: candidates, selection: selection,
                latencyMilliseconds: milliseconds, precision: requestedPrecision)
        } catch is CancellationError {
        } catch {
            guard search == searchGeneration, configuration == configurationGeneration else { return }
            routingError = error.localizedDescription
        }
    }

    nonisolated private static func milliseconds(_ duration: Duration) -> Double {
        let parts = duration.components
        return Double(parts.seconds) * 1_000 + Double(parts.attoseconds) / 1e15
    }
}
