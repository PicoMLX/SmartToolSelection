//  Retrieval.swift
//  On-device semantic tool selection with a LiquidAI LFM2.5 retriever, running
//  natively via MLX (mlx-swift-lm's MLXEmbedders). A Swift port of the reference
//  server's search.py:
//
//    * embedding — LFM2.5-Embedding-350M, one CLS vector per tool, cosine ranking,
//      with the model's asymmetric "query: " / "document: " prompts.
//    * colbert   — LFM2.5-ColBERT-350M, one vector per token, MaxSim late interaction,
//      with "[Q] " / "[D] " markers.
//
//  The index is built once when a model loads; typing a request encodes the query
//  and ranks the 151 tools. Everything runs in-process — no server.

import Foundation
import MLX
import MLXEmbedders
import MLXLMCommon
import Observation
import Tokenizers

// MARK: - Public types

enum Backend: String, CaseIterable, Identifiable, Sendable {
    case embedding
    case colbert
    var id: String { rawValue }
    var title: String { self == .embedding ? "Embedding" : "ColBERT" }
    var modelName: String {
        self == .colbert ? "LFM2.5-ColBERT-350M" : "LFM2.5-Embedding-350M"
    }
}

enum Quant: String, CaseIterable, Identifiable, Sendable {
    case bf16, int8 = "8bit", int4 = "4bit"
    var id: String { rawValue }
    var suffix: String { self == .bf16 ? "bf16" : rawValue }
    var title: String { self == .bf16 ? "bf16" : (self == .int8 ? "int8" : "int4") }
}

/// One ranked tool returned to the UI.
struct SearchResult: Identifiable, Sendable, Hashable {
    let tool: Tool
    let score: Float
    let rank: Int
    var id: String { tool.id }
}

/// Where the converted model directories live on disk (the demo loads locally;
/// the library also supports `mlx-community` registry ids).
let modelsRoot = URL(
    fileURLWithPath: "/Users/ronaldmannak/Developer/Projects/Models/mlx-models")

func modelDirectory(backend: Backend, quant: Quant) -> URL {
    modelsRoot.appending(component: "\(backend.modelName)-\(quant.suffix)")
}

// MARK: - Retrieval engine (actor — keeps the non-Sendable model off the main thread)

actor RetrievalEngine {
    private var model: LFM2BidirectionalModel?
    private var tokenizer: (any Tokenizers.Tokenizer)?
    private var config: LFM2BidirectionalConfiguration?

    private var docVectors: [[Float]] = []  // embedding: (nTools, hidden)
    private var docMultiVectors: [[[Float]]] = []  // colbert: per-tool (L, projDim)

    private var bosId: Int { 1 }  // LFM2.5 bos_token_id; CLS == BOS at position 0
    private var padId: Int32 { 0 }  // LFM2.5 pad_token_id; ColBERT query-augmentation token
    private var head: LFM2BidirectionalConfiguration.MLXHead.Kind {
        config?.mlx.head ?? .embedding
    }

    /// Load a converted model directory (config.json + model.safetensors + tokenizer.json).
    /// Handles bf16 and pre-quantized (int4/int8) checkpoints via the data-driven loader.
    func load(directory: URL) async throws {
        let configData = try Data(contentsOf: directory.appending(component: "config.json"))
        let base = try JSONDecoder.json5().decode(BaseConfiguration.self, from: configData)
        let cfg = try JSONDecoder.json5().decode(
            LFM2BidirectionalConfiguration.self, from: configData)
        let m = LFM2BidirectionalModel(cfg)
        try loadWeights(
            modelDirectory: directory, model: m,
            perLayerQuantization: base.perLayerQuantization)
        eval(m)

        let tok = try await AutoTokenizer.from(modelFolder: directory)

        self.config = cfg
        self.model = m
        self.tokenizer = tok
        self.docVectors = []
        self.docMultiVectors = []
    }

    /// Encode every tool's routing text into the index (documents).
    func buildIndex(routingTexts: [String]) {
        guard let config else { return }
        switch head {
        case .embedding:
            let prefix = config.mlx.prompts?["document"] ?? "document: "
            docVectors = routingTexts.map { encodeEmbedding(prefix + $0) }
        case .colbert:
            let prefix = config.mlx.documentPrefix ?? "[D] "
            // Documents are not augmented — encode at natural length.
            docMultiVectors = routingTexts.map {
                encodeColbert(prefix + $0, padToQueryLength: false)
            }
        }
    }

    /// Score the query against every indexed tool; returns one score per tool (tool order).
    func scores(for query: String) -> [Float] {
        guard let config else { return [] }
        switch head {
        case .embedding:
            let prefix = config.mlx.prompts?["query"] ?? "query: "
            let q = encodeEmbedding(prefix + query)
            return docVectors.map { dot($0, q) }  // cosine (vectors normalized)
        case .colbert:
            let prefix = config.mlx.queryPrefix ?? "[Q] "
            let q = encodeColbert(prefix + query)
            return docMultiVectors.map { maxSim(query: q, document: $0) }
        }
    }

    // MARK: Encoding

    private func tokenIds(_ text: String) -> [Int32] {
        var ids = tokenizer?.encode(text: text) ?? []
        if ids.first != bosId { ids.insert(bosId, at: 0) }  // ensure CLS/BOS leads
        return ids.map { Int32($0) }
    }

    /// CLS-pooled, L2-normalized sentence vector.
    private func encodeEmbedding(_ text: String) -> [Float] {
        guard let model else { return [] }
        let ids = MLXArray(tokenIds(text)).reshaped(1, -1)
        let out = model(ids)
        let pooled = Pooling(strategy: .cls)(out, normalize: true)  // (1, hidden)
        pooled.eval()
        return pooled.reshaped(-1).asArray(Float.self)
    }

    /// Per-token, L2-normalized projection vectors: a (L, projDim) matrix.
    ///
    /// `padToQueryLength` enables ColBERT **query augmentation** (on by default, the
    /// trained query behavior): the input is truncated/padded to `query_length` (32)
    /// with pad tokens. Those positions are masked as attention *keys* (attend-to =
    /// off) but still emit query vectors that are scored in MaxSim — extra "soft
    /// expansion" slots that the model contextualizes toward relevant terms, raising
    /// recall and pushing scores up. Documents pass `false` (encoded at natural
    /// length); a long query is truncated to `query_length`. Flip the default to
    /// `false` to see the un-augmented behavior (lower scores, same ranking).
    private func encodeColbert(_ text: String, padToQueryLength: Bool = true) -> [[Float]] {
        guard let model else { return [] }
        var ids = tokenIds(text)
        var attentionMask: MLXArray? = nil

        if padToQueryLength, let queryLength = config?.mlx.queryLength {
            if ids.count > queryLength { ids = Array(ids.prefix(queryLength)) }  // truncate
            var mask = [Int32](repeating: 1, count: ids.count)
            while ids.count < queryLength {  // augmentation positions: pad id, attend-to off
                ids.append(padId)
                mask.append(0)
            }
            attentionMask = MLXArray(mask).reshaped(1, -1)
        }

        // Raw per-token vectors (the encoder leaves masked positions un-zeroed so the
        // augmentation tokens survive), then per-token L2 normalization.
        let out = model(
            MLXArray(ids).reshaped(1, -1),
            positionIds: nil, tokenTypeIds: nil, attentionMask: attentionMask)
        let pooled = Pooling(strategy: .none)(out, normalize: true)  // (1, L, projDim)
        pooled.eval()
        let l = pooled.dim(1)
        let d = pooled.dim(2)
        let flat = pooled.reshaped(l, d)
        return (0 ..< l).map { flat[$0].asArray(Float.self) }
    }
}

// MARK: - Scoring helpers (plain arrays; cheap for 151 tools)

private func dot(_ a: [Float], _ b: [Float]) -> Float {
    var s: Float = 0
    let n = min(a.count, b.count)
    var i = 0
    while i < n {
        s += a[i] * b[i]
        i += 1
    }
    return s
}

/// MaxSim late interaction: for each query token, the best dot product over the
/// document's tokens; summed and normalized by the query-token count.
private func maxSim(query: [[Float]], document: [[Float]]) -> Float {
    guard !query.isEmpty, !document.isEmpty else { return 0 }
    var total: Float = 0
    for q in query {
        var best = -Float.greatestFiniteMagnitude
        for d in document {
            let s = dot(q, d)
            if s > best { best = s }
        }
        total += best
    }
    return total / Float(query.count)
}

// MARK: - App model (MainActor — drives the UI)

@MainActor
@Observable
final class AppModel {
    enum Status: Equatable {
        case idle
        case loading(String)
        case ready
        case failed(String)
    }

    let tools: [Tool] = ToolCatalog.load()
    private(set) var status: Status = .idle
    private(set) var results: [SearchResult] = []
    private(set) var lastQuery: String = ""
    private(set) var lastLatencyMs: Int = 0

    var backend: Backend = .embedding
    var quant: Quant = .bf16

    private var engine = RetrievalEngine()
    private var loadedKey: String?

    var domains: [String] {
        Array(Set(tools.map(\.domain))).filter { !$0.isEmpty }.sorted()
    }

    /// Load the selected model + build the tool index. Idempotent per backend/quant.
    func loadIfNeeded() async {
        let key = "\(backend.rawValue)-\(quant.suffix)"
        if loadedKey == key, case .ready = status { return }
        let dir = modelDirectory(backend: backend, quant: quant)
        status = .loading("Loading \(backend.modelName) (\(quant.title))…")
        do {
            try await engine.load(directory: dir)
            status = .loading("Indexing \(tools.count) tools…")
            await engine.buildIndex(routingTexts: tools.map(\.routingText))
            loadedKey = key
            status = .ready
            if !lastQuery.isEmpty { await search(lastQuery) }
        } catch {
            status = .failed("\(error.localizedDescription)\n\(dir.path)")
        }
    }

    /// Switch backend/quant and reload.
    func reload(backend: Backend, quant: Quant) async {
        self.backend = backend
        self.quant = quant
        loadedKey = nil
        results = []
        await loadIfNeeded()
    }

    func clearResults() {
        results = []
        lastQuery = ""
    }

    func search(_ query: String, k: Int = 5) async {
        lastQuery = query
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            results = []
            return
        }
        guard case .ready = status else { return }
        let started = Date()
        let scores = await engine.scores(for: trimmed)
        guard scores.count == tools.count else { return }
        let ranked =
            zip(tools, scores)
            .sorted { $0.1 > $1.1 }
            .prefix(k)
            .enumerated()
            .map { SearchResult(tool: $0.element.0, score: $0.element.1, rank: $0.offset + 1) }
        results = Array(ranked)
        lastLatencyMs = Int(Date().timeIntervalSince(started) * 1000)
    }
}
