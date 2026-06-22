//  SmartToolSelectionTests.swift
//  End-to-end checks for the on-device retrieval engine. Run via Xcode/xcodebuild
//  (the MLX Metal library is only bundled there). The retrieval tests are skipped
//  automatically when the local model directories are absent.

import Foundation
import Testing

@testable import SmartToolSelection

struct SmartToolSelectionTests {

    static let docs = [
        "Paris is the capital and most populous city of France.",
        "Berlin is the capital of Germany and a major cultural hub.",
        "The Eiffel Tower, located in Paris, was completed in 1889.",
        "Bananas are an excellent dietary source of potassium.",
        "France is a country in Western Europe known for its cuisine.",
    ]
    static let query = "What is the capital of France?"

    // Tests load from the local converted dirs (fast, no network); the app downloads from HF.
    static let localModelsRoot = URL(
        fileURLWithPath: "/Users/ronaldmannak/Developer/Projects/Models/mlx-models")
    static func localDir(_ backend: Backend, _ quant: Quant = .bf16) -> URL {
        localModelsRoot.appending(component: "\(backend.modelName)-\(quant.suffix)")
    }
    static func dirExists(_ backend: Backend) -> Bool {
        FileManager.default.fileExists(
            atPath: localDir(backend).appending(component: "config.json").path)
    }

    private func order(_ scores: [Float]) -> [Int] {
        scores.enumerated().sorted { $0.element > $1.element }.map(\.offset)
    }

    @Test(
        "Embedding backend ranks France/Paris above an unrelated doc",
        .enabled(if: dirExists(.embedding)))
    func embeddingRanking() async throws {
        let engine = RetrievalEngine()
        try await engine.load(directory: Self.localDir(.embedding))
        await engine.buildIndex(routingTexts: Self.docs)
        let scores = await engine.scores(for: Self.query)

        #expect(scores.count == Self.docs.count)
        let ranked = order(scores)
        #expect([0, 2, 4].contains(ranked[0]))  // a France/Paris doc is top
        #expect(!ranked.prefix(2).contains(3))  // bananas is not near the top
    }

    @Test(
        "ColBERT backend ranks France/Paris above an unrelated doc",
        .enabled(if: dirExists(.colbert)))
    func colbertRanking() async throws {
        let engine = RetrievalEngine()
        try await engine.load(directory: Self.localDir(.colbert))
        await engine.buildIndex(routingTexts: Self.docs)
        let scores = await engine.scores(for: Self.query)

        #expect(scores.count == Self.docs.count)
        let ranked = order(scores)
        #expect([0, 2, 4].contains(ranked[0]))
        #expect(!ranked.prefix(2).contains(3))
    }

    @Test(
        "ColBERT query augmentation lifts the top score (~0.62 -> ~0.8)",
        .enabled(if: dirExists(.colbert)))
    func colbertAugmentationLiftsScore() async throws {
        let engine = RetrievalEngine()
        try await engine.load(directory: Self.localDir(.colbert))
        let doc = ToolCatalog.load().first { $0.name == "search_products" }!.routingText
        await engine.buildIndex(routingTexts: [doc])
        let scores = await engine.scores(for: "show me cheap blue outdoor chairs")
        // Un-augmented this query scores ~0.62; query augmentation (with the correct
        // pad token = eos) lifts it to ~0.83. Guards against a wrong augmentation token.
        #expect(scores[0] > 0.72, "expected augmentation lift, got \(scores[0])")
    }

    @Test("routingText embeds parameter names, enum options, and keyword examples")
    func routingTextComposition() {
        let tool = Tool(
            name: "search_products",
            description: "Search the catalog.",
            domain: "ecommerce",
            parameters: [
                ToolParameter(
                    name: "price_range", type: "string", description: "",
                    enumValues: ["under_50", "over_500"], required: false,
                    itemsType: nil, itemsEnum: nil)
            ],
            keywords: ["table", "lamp"])
        let rt = tool.routingText
        #expect(rt.contains("parameters: price range"))
        #expect(rt.contains("options: under 50, over 500"))
        #expect(rt.contains("examples: table, lamp"))
    }

    @Test("Catalog loads all 151 bundled tools across 7 domains")
    func catalogLoads() {
        let tools = ToolCatalog.load()
        #expect(tools.count == 151)
        #expect(Set(tools.map(\.domain)).count == 7)
    }
}
