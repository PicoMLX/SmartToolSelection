//  ContentView.swift
//  A native SwiftUI port of the Smart Tool Selection website. Retrieval runs
//  on-device via MLX (mlx-swift-lm) instead of a FastAPI server.

import SwiftUI

// MARK: - Brand palette (from the original site)

extension Color {
    init(hex: UInt32) {
        self.init(
            .sRGB,
            red: Double((hex >> 16) & 0xff) / 255,
            green: Double((hex >> 8) & 0xff) / 255,
            blue: Double(hex & 0xff) / 255,
            opacity: 1)
    }
}

enum Brand {
    static let lpurple = Color(hex: 0xCD82F0)
    static let purple = Color(hex: 0x5505AF)
    static let border = Color(hex: 0xE3BAF7)
    static let lp10 = Color(hex: 0xFAF2FE)
    static let lp30 = Color(hex: 0xF0DAFA)
    static let textMid = Color(hex: 0x4D4D4D)
    static let textLight = Color(hex: 0x808080)

    static let domainColors: [String: Color] = [
        "devops": Color(hex: 0x6AA9E9), "ecommerce": Color(hex: 0xCD82F0),
        "finance": Color(hex: 0x4FB286), "healthcare": Color(hex: 0xE98AA9),
        "support": Color(hex: 0xF0A35E), "travel": Color(hex: 0x7B6EF0),
        "workplace": Color(hex: 0x9AA0A8),
    ]
    static func color(for domain: String) -> Color { domainColors[domain] ?? textLight }
}

private let examples: [(domain: String, query: String)] = [
    ("ecommerce", "show me cheap blue outdoor chairs"),
    ("devops", "spin up a new postgres database on aws"),
    ("travel", "find me a business class flight to tokyo"),
    ("finance", "there's a charge I didn't make, dispute it"),
    ("healthcare", "book a telehealth visit with a dermatologist"),
    ("support", "bump this ticket up to a manager"),
    ("workplace", "request a week of vacation"),
    ("ecommerce", "where is my delivery right now"),
]

// MARK: - Root

struct ContentView: View {
    @State private var model = AppModel()
    @State private var query = ""
    @State private var searchTask: Task<Void, Never>?

    var body: some View {
        ScrollView {
            VStack(spacing: 22) {
                header
                controls
                searchBar
                exampleChips
                statusLine
                results
            }
            .frame(maxWidth: 720)
            .padding(.horizontal, 20)
            .padding(.vertical, 32)
            .frame(maxWidth: .infinity)
        }
        .background(
            LinearGradient(colors: [Brand.lp10, .white], startPoint: .top, endPoint: .bottom)
                .ignoresSafeArea()
        )
        // The design is a light-only, purple-on-white theme (matching the original
        // website). Pin light appearance so `.primary` text (title, tool names, the
        // search field's text) stays dark on the hardcoded light surfaces in dark mode.
        .preferredColorScheme(.light)
        .task { await model.loadIfNeeded() }
    }

    // MARK: Header

    private var header: some View {
        VStack(spacing: 10) {
            (Text("Powered by ")
                + Text(model.backend.modelName).foregroundStyle(Brand.purple))
                .font(.system(.caption2, design: .monospaced))
                .tracking(2)
                .textCase(.uppercase)
                .foregroundStyle(Brand.textLight)

            Text("Smart Tool Selection")
                .font(.system(size: 34, weight: .bold))
                .tracking(-0.5)

            (Text("An agent with ")
                + Text("\(model.tools.count) tools").foregroundStyle(Brand.purple).bold()
                + Text(" can't fit them all in one prompt. Send a request and the model pre-selects the ")
                + Text("5 most relevant").foregroundStyle(Brand.purple).bold()
                + Text(" ones to reduce context rot."))
                .font(.callout)
                .foregroundStyle(Brand.textMid)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 540)
        }
        .padding(.top, 8)
    }

    // MARK: Backend / quant controls

    private var controls: some View {
        HStack(spacing: 12) {
            Picker("Backend", selection: $model.backend) {
                ForEach(Backend.allCases) { Text($0.title).tag($0) }
            }
            .pickerStyle(.segmented)
            .fixedSize()

            Picker("Precision", selection: $model.quant) {
                ForEach(Quant.allCases) { Text($0.title).tag($0) }
            }
            .pickerStyle(.segmented)
            .fixedSize()
        }
        .onChange(of: model.backend) { _, b in reload(backend: b, quant: model.quant) }
        .onChange(of: model.quant) { _, q in reload(backend: model.backend, quant: q) }
    }

    // MARK: Search

    private var searchBar: some View {
        HStack(spacing: 10) {
            Image(systemName: "magnifyingglass").foregroundStyle(Brand.textLight)
            TextField("describe what the user wants to do…", text: $query)
                .textFieldStyle(.plain)
                .font(.body)
                .onChange(of: query) { _, _ in debouncedSearch() }
                .onSubmit { runSearch() }
            if !query.isEmpty {
                Button {
                    query = ""
                    model.clearResults()
                } label: {
                    Image(systemName: "xmark.circle.fill").foregroundStyle(Brand.textLight)
                }
                .buttonStyle(.plain)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 13)
        .background(.white, in: RoundedRectangle(cornerRadius: 16))
        .overlay(RoundedRectangle(cornerRadius: 16).strokeBorder(Brand.border, lineWidth: 1.5))
        .shadow(color: Brand.lpurple.opacity(0.12), radius: 8, y: 3)
    }

    private var exampleChips: some View {
        FlowLayout(spacing: 8) {
            ForEach(examples, id: \.query) { ex in
                Button {
                    query = ex.query
                    runSearch()
                } label: {
                    HStack(spacing: 6) {
                        Circle().fill(Brand.color(for: ex.domain)).frame(width: 7, height: 7)
                        Text(ex.query).lineLimit(1)
                    }
                    .font(.caption)
                    .foregroundStyle(Brand.textMid)
                    .padding(.horizontal, 11)
                    .padding(.vertical, 7)
                    .background(.white.opacity(0.7), in: Capsule())
                    .overlay(Capsule().strokeBorder(Brand.border.opacity(0.6)))
                }
                .buttonStyle(.plain)
            }
        }
    }

    // MARK: Status + results

    @ViewBuilder private var statusLine: some View {
        switch model.status {
        case .idle, .loading:
            HStack(spacing: 8) {
                ProgressView().controlSize(.small)
                Text(loadingText)
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(Brand.textLight)
            }
        case .failed(let message):
            Text(message)
                .font(.system(.caption, design: .monospaced))
                .foregroundStyle(.red)
                .multilineTextAlignment(.center)
                .textSelection(.enabled)
        case .ready:
            if !model.results.isEmpty {
                Text(
                    "top \(model.results.count) of \(model.tools.count) tools · \(model.lastLatencyMs) ms"
                )
                .font(.system(.caption, design: .monospaced))
                .foregroundStyle(Brand.textLight)
            }
        }
    }

    private var loadingText: String {
        if case .loading(let s) = model.status { return s }
        return "Starting…"
    }

    @ViewBuilder private var results: some View {
        if model.results.isEmpty, case .ready = model.status {
            Text("Type a request above to retrieve the tools that match ✨")
                .font(.callout)
                .foregroundStyle(Brand.textLight)
                .padding(.top, 24)
        } else {
            VStack(spacing: 12) {
                ForEach(model.results) { ResultCard(result: $0) }
            }
        }
    }

    // MARK: Search helpers

    private func debouncedSearch() {
        searchTask?.cancel()
        let q = query
        searchTask = Task {
            try? await Task.sleep(for: .milliseconds(250))
            if Task.isCancelled { return }
            await model.search(q)
        }
    }

    private func runSearch() {
        searchTask?.cancel()
        let q = query
        searchTask = Task { await model.search(q) }
    }

    private func reload(backend: Backend, quant: Quant) {
        Task { await model.reload(backend: backend, quant: quant) }
    }
}

// MARK: - Result card

private struct ResultCard: View {
    let result: SearchResult
    @State private var expanded = false

    private var tool: Tool { result.tool }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Button { withAnimation(.snappy) { expanded.toggle() } } label: {
                cardHeader
            }
            .buttonStyle(.plain)

            if expanded { expandedDetail }
        }
        .padding(16)
        .background(.white.opacity(0.55), in: RoundedRectangle(cornerRadius: 16))
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16))
        .overlay(
            RoundedRectangle(cornerRadius: 16)
                .strokeBorder(
                    result.rank == 1 ? Brand.lpurple.opacity(0.8) : Brand.border.opacity(0.7),
                    lineWidth: result.rank == 1 ? 1.6 : 1)
        )
    }

    private var cardHeader: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 10) {
                Text("#\(result.rank)")
                    .font(.system(.caption, design: .monospaced)).foregroundStyle(Brand.textLight)
                Text(tool.name)
                    .font(.system(.body, design: .monospaced)).fontWeight(.semibold)
                Spacer()
                DomainBadge(domain: tool.domain)
                Text("\(Int((result.score * 100).rounded()))%")
                    .font(.system(.caption, design: .monospaced).bold())
                    .foregroundStyle(Brand.purple)
                Image(systemName: "chevron.right")
                    .font(.caption.bold()).foregroundStyle(Brand.textLight)
                    .rotationEffect(.degrees(expanded ? 90 : 0))
            }
            Text(tool.description)
                .font(.callout).foregroundStyle(Brand.textMid)
                .multilineTextAlignment(.leading)
                .fixedSize(horizontal: false, vertical: true)
            if !tool.parameters.isEmpty {
                FlowLayout(spacing: 6) {
                    ForEach(tool.parameters) { ParamChip(parameter: $0) }
                }
            }
        }
        .contentShape(Rectangle())
    }

    private var expandedDetail: some View {
        VStack(alignment: .leading, spacing: 12) {
            Divider()
            if !tool.parameters.isEmpty {
                Text("ARGUMENTS")
                    .font(.system(.caption2, design: .monospaced)).tracking(1.5)
                    .foregroundStyle(Brand.textLight)
                ForEach(tool.parameters) { p in
                    VStack(alignment: .leading, spacing: 2) {
                        HStack(spacing: 6) {
                            Text(p.name).font(.system(.caption, design: .monospaced).bold())
                            Text(p.type).font(.caption2).foregroundStyle(Brand.textLight)
                            if p.required {
                                Text("required").font(.caption2).foregroundStyle(Brand.purple)
                            }
                        }
                        if !p.description.isEmpty {
                            Text(p.description).font(.caption).foregroundStyle(Brand.textMid)
                        }
                        if let e = p.enumValues, !e.isEmpty {
                            Text(e.joined(separator: " · "))
                                .font(.system(.caption2, design: .monospaced))
                                .foregroundStyle(Brand.textLight)
                        }
                    }
                }
            }
            Text("TOOL DEFINITION")
                .font(.system(.caption2, design: .monospaced)).tracking(1.5)
                .foregroundStyle(Brand.textLight)
            Text(tool.schemaJSON)
                .font(.system(.caption2, design: .monospaced))
                .foregroundStyle(Brand.textMid)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(10)
                .background(Brand.lp10, in: RoundedRectangle(cornerRadius: 10))
        }
        .transition(.opacity.combined(with: .move(edge: .top)))
    }
}

private struct DomainBadge: View {
    let domain: String
    var body: some View {
        Text(domain)
            .font(.system(.caption2, design: .monospaced))
            .padding(.horizontal, 8).padding(.vertical, 3)
            .background(Brand.color(for: domain).opacity(0.15), in: Capsule())
            .foregroundStyle(Brand.color(for: domain))
    }
}

private struct ParamChip: View {
    let parameter: ToolParameter
    var body: some View {
        HStack(spacing: 3) {
            Text(parameter.name)
            if parameter.isEnum { Image(systemName: "chevron.down").font(.system(size: 7)) }
        }
        .font(.system(.caption2, design: .monospaced))
        .foregroundStyle(Brand.textMid)
        .padding(.horizontal, 8).padding(.vertical, 4)
        .background(Brand.lp30.opacity(0.5), in: Capsule())
    }
}

// MARK: - Simple flow layout (wrapping chips)

struct FlowLayout: Layout {
    var spacing: CGFloat = 8

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout Void) -> CGSize {
        let maxWidth = proposal.width ?? .infinity
        var x: CGFloat = 0, y: CGFloat = 0, rowHeight: CGFloat = 0
        for view in subviews {
            let size = view.sizeThatFits(.unspecified)
            if x + size.width > maxWidth, x > 0 {
                x = 0
                y += rowHeight + spacing
                rowHeight = 0
            }
            x += size.width + spacing
            rowHeight = max(rowHeight, size.height)
        }
        return CGSize(width: maxWidth == .infinity ? x : maxWidth, height: y + rowHeight)
    }

    func placeSubviews(
        in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout Void
    ) {
        var x = bounds.minX
        var y = bounds.minY
        var rowHeight: CGFloat = 0
        for view in subviews {
            let size = view.sizeThatFits(.unspecified)
            if x + size.width > bounds.maxX, x > bounds.minX {
                x = bounds.minX
                y += rowHeight + spacing
                rowHeight = 0
            }
            view.place(at: CGPoint(x: x, y: y), proposal: ProposedViewSize(size))
            x += size.width + spacing
            rowHeight = max(rowHeight, size.height)
        }
    }
}

#Preview {
    ContentView()
}
