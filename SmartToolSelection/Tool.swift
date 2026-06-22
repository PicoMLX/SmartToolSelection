//  Tool.swift
//  Tool-catalog model — a Swift port of the reference server's toolset.py.
//
//  A *pack* is JSON in the standard function-calling shape (OpenAI / JSON-Schema /
//  MCP tools/list). Seven curated packs ship in Resources/packs (151 tools). Each
//  tool is indexed from its `routingText` (name + description + parameter names +
//  enum values + keywords); typing a request retrieves the few tools that matter.

import Foundation

/// One tool parameter parsed from a JSON-Schema property.
struct ToolParameter: Identifiable, Hashable {
    let name: String
    let type: String
    let description: String
    let enumValues: [String]?
    let required: Bool

    var id: String { name }
    var isEnum: Bool { (enumValues?.isEmpty == false) }
}

/// A single callable tool, its parameters, and the pack (domain) it came from.
struct Tool: Identifiable, Hashable {
    let name: String
    let description: String
    let domain: String
    let parameters: [ToolParameter]
    let keywords: [String]

    var id: String { "\(domain)|\(name)" }

    /// Text indexed for retrieval: name + description + parameter names + enum
    /// values + keywords. Enum values surface the discriminating vocabulary
    /// ("aws", "business class", "urgent"); keywords are index-only document
    /// expansion for implicit queries ("table" -> furniture catalog).
    var routingText: String {
        let desc = description.trimmingCharacters(in: .whitespacesAndNewlines)
        let base = desc.isEmpty ? name : "\(name): \(desc)"
        var extras: [String] = []
        if !parameters.isEmpty {
            extras.append(
                "parameters: "
                    + parameters.map { $0.name.replacingOccurrences(of: "_", with: " ") }
                    .joined(separator: ", "))
            let values =
                parameters
                .filter { $0.isEnum }
                .flatMap { $0.enumValues ?? [] }
                .map { $0.replacingOccurrences(of: "_", with: " ") }
            if !values.isEmpty {
                extras.append("options: " + values.joined(separator: ", "))
            }
        }
        if !keywords.isEmpty {
            extras.append("examples: " + keywords.joined(separator: ", "))
        }
        return extras.isEmpty ? base : "\(base) (\(extras.joined(separator: "; ")))"
    }

    /// The standard function-calling schema — exactly what a downstream LLM
    /// receives — rendered as pretty JSON for the result card (key order preserved).
    var schemaJSON: String {
        func esc(_ s: String) -> String {
            var out = ""
            for c in s {
                switch c {
                case "\"": out += "\\\""
                case "\\": out += "\\\\"
                case "\n": out += "\\n"
                case "\t": out += "\\t"
                default: out.append(c)
                }
            }
            return out
        }
        func arr(_ xs: [String]) -> String {
            "[" + xs.map { "\"\(esc($0))\"" }.joined(separator: ", ") + "]"
        }

        var props: [String] = []
        for p in parameters {
            var spec = ["\"type\": \"\(esc(p.type))\""]
            if !p.description.isEmpty { spec.append("\"description\": \"\(esc(p.description))\"") }
            if let e = p.enumValues, !e.isEmpty { spec.append("\"enum\": \(arr(e))") }
            props.append("      \"\(esc(p.name))\": { \(spec.joined(separator: ", ")) }")
        }
        let required = parameters.filter { $0.required }.map { $0.name }

        var lines = ["{"]
        lines.append("  \"name\": \"\(esc(name))\",")
        lines.append("  \"description\": \"\(esc(description))\",")
        lines.append("  \"parameters\": {")
        lines.append("    \"type\": \"object\",")
        if props.isEmpty {
            lines.append("    \"properties\": {}")
        } else {
            lines.append("    \"properties\": {")
            lines.append(props.joined(separator: ",\n"))
            lines.append(required.isEmpty ? "    }" : "    },")
        }
        if !required.isEmpty {
            lines.append("    \"required\": \(arr(required))")
        }
        lines.append("  }")
        lines.append("}")
        return lines.joined(separator: "\n")
    }
}

// MARK: - Catalog loading

enum ToolCatalog {
    /// Load every bundled pack, merged into one catalog in deterministic (sorted) order.
    static func load() -> [Tool] {
        guard let packsURL = Bundle.main.url(forResource: "packs", withExtension: nil),
            let entries = try? FileManager.default.contentsOfDirectory(
                at: packsURL, includingPropertiesForKeys: nil)
        else {
            // Fall back to top-level bundled json (flat resource layout).
            return loadFlat()
        }
        let packFiles = entries.filter { $0.pathExtension == "json" }.sorted {
            $0.lastPathComponent < $1.lastPathComponent
        }
        return packFiles.flatMap { loadPack(at: $0) }
    }

    private static func loadFlat() -> [Tool] {
        let names = [
            "devops", "ecommerce", "finance", "healthcare", "support", "travel", "workplace",
        ]
        return names.sorted().compactMap { Bundle.main.url(forResource: $0, withExtension: "json") }
            .flatMap { loadPack(at: $0) }
    }

    private static func loadPack(at url: URL) -> [Tool] {
        guard let data = try? Data(contentsOf: url),
            let pack = try? JSONDecoder().decode(RawPack.self, from: data)
        else { return [] }
        let domain = pack.name ?? url.deletingPathExtension().lastPathComponent
        return pack.tools.map { $0.tool(domain: domain) }
    }
}

// MARK: - Raw JSON decoding (preserves property order)

private struct DynamicKey: CodingKey {
    var stringValue: String
    var intValue: Int? { nil }
    init(stringValue: String) { self.stringValue = stringValue }
    init?(intValue: Int) { nil }
}

private struct RawPack: Decodable {
    let name: String?
    let tools: [RawTool]
}

private struct RawTool: Decodable {
    let name: String
    let description: String?
    let parameters: RawParameters?
    let keywords: [String]?

    func tool(domain: String) -> Tool {
        let required = Set(parameters?.required ?? [])
        let params = (parameters?.properties ?? []).map { entry in
            ToolParameter(
                name: entry.0,
                type: entry.1.type ?? "string",
                description: entry.1.description ?? "",
                enumValues: entry.1.enumValues,
                required: required.contains(entry.0))
        }
        return Tool(
            name: name, description: description ?? "", domain: domain,
            parameters: params, keywords: keywords ?? [])
    }
}

private struct RawSpec: Decodable {
    let type: String?
    let description: String?
    let enumValues: [String]?

    enum Keys: String, CodingKey {
        case type, description
        case enumValues = "enum"
    }
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: Keys.self)
        type = try c.decodeIfPresent(String.self, forKey: .type)
        description = try c.decodeIfPresent(String.self, forKey: .description)
        // enum values may be strings or other scalars; coerce to strings.
        if let strings = try? c.decodeIfPresent([String].self, forKey: .enumValues) {
            enumValues = strings
        } else {
            enumValues = nil
        }
    }
}

private struct RawParameters: Decodable {
    let properties: [(String, RawSpec)]
    let required: [String]

    enum Keys: String, CodingKey { case properties, required }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: Keys.self)
        required = (try? c.decode([String].self, forKey: .required)) ?? []
        var props: [(String, RawSpec)] = []
        if c.contains(.properties),
            let pc = try? c.nestedContainer(keyedBy: DynamicKey.self, forKey: .properties)
        {
            for key in pc.allKeys {
                if let spec = try? pc.decode(RawSpec.self, forKey: key) {
                    props.append((key.stringValue, spec))
                }
            }
        }
        properties = props
    }
}
