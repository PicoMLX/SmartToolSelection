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

import Accelerate
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

/// The six converted models are published on the Hugging Face Hub under the
/// `ronaldmannak` account; they're downloaded and cached on first use.
func modelRepoId(backend: Backend, quant: Quant) -> String {
    "ronaldmannak/\(backend.modelName)-\(quant.suffix)"
}

// MARK: - Model download (Hugging Face Hub)

/// Downloads the converted model files from a public HF repo into the app's
/// Application Support cache and returns the local directory. Files already present
/// are skipped, so subsequent launches don't re-download.
enum ModelDownloader {
    private static let smallFiles = [
        "config.json", "config_sentence_transformers.json", "tokenizer.json",
        "tokenizer_config.json", "special_tokens_map.json",
    ]
    private static let weightsFile = "model.safetensors"

    static func download(
        repoId: String, progress: @escaping @Sendable (Double) -> Void
    ) async throws -> URL {
        let directory = try cacheDirectory(for: repoId)
        var didDownload = false
        for file in smallFiles {
            if try await fetch(repoId: repoId, file: file, into: directory, delegate: nil) {
                didDownload = true
            }
        }
        // The weights are ~99% of the bytes — report their byte progress directly.
        if try await fetch(
            repoId: repoId, file: weightsFile, into: directory,
            delegate: DownloadProgress(progress))
        {
            didDownload = true
        }
        // Tell users exactly where the model lives on disk.
        print(
            didDownload
                ? "✅ Downloaded \(repoId) to: \(directory.path)"
                : "✅ Using cached \(repoId) at: \(directory.path)")
        return directory
    }

    private static func cacheDirectory(for repoId: String) throws -> URL {
        var dir = URL.applicationSupportDirectory
            .appending(path: "SmartToolSelection/models")
            .appending(path: repoId)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try? dir.setResourceValues(values)
        return dir
    }

    /// Returns `true` if the file was fetched from the network, `false` if it was
    /// already cached on disk.
    @discardableResult
    private static func fetch(
        repoId: String, file: String, into directory: URL, delegate: URLSessionTaskDelegate?
    ) async throws -> Bool {
        let destination = directory.appending(component: file)
        if FileManager.default.fileExists(atPath: destination.path) { return false }
        guard let url = URL(string: "https://huggingface.co/\(repoId)/resolve/main/\(file)") else {
            throw URLError(.badURL)
        }
        let (temp, response) = try await URLSession.shared.download(from: url, delegate: delegate)
        if let http = response as? HTTPURLResponse, !(200 ..< 300).contains(http.statusCode) {
            throw URLError(.badServerResponse)
        }
        try? FileManager.default.removeItem(at: destination)
        try FileManager.default.moveItem(at: temp, to: destination)
        return true
    }
}

/// Forwards a download task's byte progress (KVO on `task.progress`).
private final class DownloadProgress: NSObject, URLSessionTaskDelegate, @unchecked Sendable {
    private let report: @Sendable (Double) -> Void
    private var observation: NSKeyValueObservation?
    init(_ report: @escaping @Sendable (Double) -> Void) { self.report = report }
    func urlSession(_ session: URLSession, didCreateTask task: URLSessionTask) {
        observation = task.progress.observe(\.fractionCompleted, options: [.new]) {
            [report] progress, _ in
            report(progress.fractionCompleted)
        }
    }
}

// MARK: - Retrieval engine (actor — keeps the non-Sendable model off the main thread)

actor RetrievalEngine {
    private var model: LFM2BidirectionalModel?
    private var tokenizer: (any Tokenizers.Tokenizer)?
    private var config: LFM2BidirectionalConfiguration?

    // Index stored as flat, row-major Float buffers so scoring is a single BLAS call
    // (cblas_sgemv / cblas_sgemm) instead of a Swift dot-product loop.
    private var docEmbeddings: [Float] = []  // embedding: flat (nTools × hidden)
    private var embeddingDim = 0
    private var docMatrices: [[Float]] = []  // colbert: per-tool flat (Lᵈ × projDim)
    private var docRows: [Int] = []  // colbert: per-tool token count Lᵈ
    private var projDim = 0
    private var skiplistIds: Set<Int32> = []  // colbert: punctuation tokens dropped from doc scoring

    private var bosId: Int { 1 }  // LFM2.5 bos_token_id; CLS == BOS at position 0
    // ColBERT query-augmentation token = the tokenizer's pad token, which for
    // LFM2.5-ColBERT is `<|im_end|>` (eos, id 7), NOT `<|pad|>` (id 0). The conv is
    // (correctly) left unmasked, so augmentation tokens flow through the model: id 0
    // yields poor expansion vectors (~60%), id 7 matches the reference (~83%).
    private var padId: Int32 { Int32(tokenizer?.eosTokenId ?? 7) }
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
        self.skiplistIds = Self.loadSkiplist(directory: directory, tokenizer: tok)
        self.docEmbeddings = []
        self.embeddingDim = 0
        self.docMatrices = []
        self.docRows = []
        self.projDim = 0
    }

    /// ColBERT document skiplist: PyLate drops punctuation tokens (from
    /// `config_sentence_transformers.json`'s `skiplist_words`) from documents before
    /// MaxSim. Resolve them to single-token ids once at load.
    private static func loadSkiplist(
        directory: URL, tokenizer: any Tokenizers.Tokenizer
    ) -> Set<Int32> {
        guard
            let data = try? Data(
                contentsOf: directory.appending(component: "config_sentence_transformers.json")),
            let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let words = obj["skiplist_words"] as? [String]
        else { return [] }
        var ids = Set<Int32>()
        for word in words {
            let enc = tokenizer.encode(text: word, addSpecialTokens: false)
            if enc.count == 1 { ids.insert(Int32(enc[0])) }
        }
        return ids
    }

    /// Encode every tool's routing text into the index (documents).
    func buildIndex(routingTexts: [String]) {
        guard let config else { return }
        switch head {
        case .embedding:
            let prefix = config.mlx.prompts?["document"] ?? "document: "
            let vectors = routingTexts.map { encodeEmbedding(prefix + $0) }
            embeddingDim = vectors.first?.count ?? 0
            docEmbeddings = vectors.flatMap { $0 }  // one contiguous (nTools × hidden) buffer
        case .colbert:
            let prefix = config.mlx.documentPrefix ?? "[D] "
            projDim = config.mlx.projDim ?? 128
            // Documents are not augmented; drop skiplist (punctuation) tokens like PyLate.
            let mats = routingTexts.map {
                encodeColbert(prefix + $0, padToQueryLength: false, dropSkiplist: true)
            }
            docMatrices = mats.map(\.data)
            docRows = mats.map(\.rows)
        }
    }

    /// Score the query against every indexed tool; returns one score per tool (tool order).
    func scores(for query: String) -> [Float] {
        guard let config else { return [] }
        switch head {
        case .embedding:
            let prefix = config.mlx.prompts?["query"] ?? "query: "
            let q = encodeEmbedding(prefix + query)
            return cosineScores(query: q)  // one BLAS matrix-vector product (cosine)
        case .colbert:
            let prefix = config.mlx.queryPrefix ?? "[Q] "
            let q = encodeColbert(prefix + query)
            return zip(docMatrices, docRows).map {
                maxSim(query: q.data, queryRows: q.rows, doc: $0.0, docRows: $0.1, dim: projDim)
            }
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
    private func encodeColbert(
        _ text: String, padToQueryLength: Bool = true, dropSkiplist: Bool = false
    ) -> (data: [Float], rows: Int) {
        guard let model else { return ([], 0) }
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
        // Row-major (L × projDim) contiguous buffer, ready for BLAS.
        let all = pooled.reshaped(l, d).asArray(Float.self)
        guard dropSkiplist, !skiplistIds.isEmpty else { return (all, l) }
        // Drop punctuation/skiplist document tokens before they reach MaxSim.
        var kept = [Float]()
        kept.reserveCapacity(all.count)
        var rows = 0
        for i in 0 ..< l where !skiplistIds.contains(ids[i]) {
            kept.append(contentsOf: all[(i * d) ..< ((i + 1) * d)])
            rows += 1
        }
        return (kept, rows)
    }

    /// Cosine scores for every indexed embedding doc as one BLAS matrix-vector
    /// product (`Docs · query`) — vectors are L2-normalized, so the dot product is
    /// the cosine. Replaces N separate per-doc dot products.
    private func cosineScores(query: [Float]) -> [Float] {
        let dim = embeddingDim
        guard dim > 0, query.count >= dim, !docEmbeddings.isEmpty else { return [] }
        let n = docEmbeddings.count / dim
        var out = [Float](repeating: 0, count: n)
        docEmbeddings.withUnsafeBufferPointer { docs in
            query.withUnsafeBufferPointer { q in
                out.withUnsafeMutableBufferPointer { result in
                    // result(n) = Docs(n × dim) · q(dim)
                    cblas_sgemv(
                        CblasRowMajor, CblasNoTrans, Int32(n), Int32(dim),
                        1, docs.baseAddress, Int32(dim),
                        q.baseAddress, 1, 0, result.baseAddress, 1)
                }
            }
        }
        return out
    }
}

// MARK: - Scoring helpers (Accelerate BLAS over flat, row-major Float buffers)

/// ColBERT MaxSim via one BLAS matrix multiply per document: form the full
/// query×document token-similarity matrix `Q · Dᵀ` with `cblas_sgemm`, then take the
/// per-query-token max (`vDSP_maxv`), sum, and normalize by the query-token count.
/// Vectors are L2-normalized, so each entry is a cosine. This replaces an
/// O(Lq·Ld) Swift double loop of individual dot products with a single GEMM + a
/// handful of vDSP reductions.
private func maxSim(query: [Float], queryRows: Int, doc: [Float], docRows: Int, dim: Int) -> Float {
    guard queryRows > 0, docRows > 0, dim > 0 else { return 0 }
    var sim = [Float](repeating: 0, count: queryRows * docRows)
    query.withUnsafeBufferPointer { q in
        doc.withUnsafeBufferPointer { d in
            sim.withUnsafeMutableBufferPointer { s in
                // sim(Lq × Ld) = Q(Lq × dim) · Dᵀ(dim × Ld)
                cblas_sgemm(
                    CblasRowMajor, CblasNoTrans, CblasTrans,
                    Int32(queryRows), Int32(docRows), Int32(dim),
                    1, q.baseAddress, Int32(dim),
                    d.baseAddress, Int32(dim),
                    0, s.baseAddress, Int32(docRows))
            }
        }
    }
    var total: Float = 0
    sim.withUnsafeBufferPointer { s in
        for i in 0 ..< queryRows {
            var best: Float = 0
            vDSP_maxv(s.baseAddress! + i * docRows, 1, &best, vDSP_Length(docRows))
            total += best
        }
    }
    return total / Float(queryRows)
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

    var backend: Backend = .colbert  // default to ColBERT, matching the web app's mode
    var quant: Quant = .bf16

    private var engine = RetrievalEngine()
    private var loadedKey: String?
    private var loadTask: Task<Void, Never>?
    private var lastDownloadPercent = -1

    var domains: [String] {
        Array(Set(tools.map(\.domain))).filter { !$0.isEmpty }.sorted()
    }

    /// Load the selected model + build the tool index. Loads are serialized through
    /// `loadTask` so rapid backend/precision changes can't interleave or land out of
    /// order; the per-key check keeps it idempotent.
    func loadIfNeeded() {
        enqueueLoad()
    }

    /// Switch backend/quant and reload (serialized).
    func reload(backend: Backend, quant: Quant) {
        self.backend = backend
        self.quant = quant
        loadedKey = nil
        results = []
        enqueueLoad()
    }

    private func enqueueLoad() {
        let previous = loadTask
        loadTask = Task { [weak self] in
            await previous?.value  // serialize: let any in-flight load finish first
            await self?.performLoad()
        }
    }

    private func performLoad() async {
        let key = "\(backend.rawValue)-\(quant.suffix)"
        if loadedKey == key, case .ready = status { return }
        let repo = modelRepoId(backend: backend, quant: quant)
        lastDownloadPercent = -1
        status = .loading("Downloading \(backend.modelName) (\(quant.title))…")
        do {
            // Download + cache the converted model files from the Hugging Face Hub.
            let directory = try await ModelDownloader.download(repoId: repo) { [weak self] fraction in
                guard let self else { return }
                Task { @MainActor in self.reportDownload(fraction) }
            }
            status = .loading("Loading \(backend.modelName)…")
            try await engine.load(directory: directory)
            status = .loading("Indexing \(tools.count) tools…")
            await engine.buildIndex(routingTexts: tools.map(\.routingText))
            loadedKey = key
            status = .ready
            if !lastQuery.isEmpty { await search(lastQuery) }
        } catch {
            status = .failed("\(error.localizedDescription)\n\(repo)")
        }
    }

    /// Throttled download-progress -> status (only updates while still downloading).
    private func reportDownload(_ fraction: Double) {
        let pct = max(0, min(100, Int(fraction * 100)))
        guard pct != lastDownloadPercent else { return }
        lastDownloadPercent = pct
        if case .loading(let message) = status, message.hasPrefix("Downloading") {
            status = .loading("Downloading \(backend.modelName) (\(quant.title))… \(pct)%")
        }
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
        // Drop stale results: a newer search or a clear superseded this query.
        guard !Task.isCancelled, query == lastQuery else { return }
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
