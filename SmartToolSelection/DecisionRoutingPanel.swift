import PicoDecisions
import SwiftUI

/// On-device decision controls and diagnostics over the retrieved candidate set.
struct DecisionRoutingPanel: View {
    @Bindable var model: DecisionRoutingModel
    let configurationChanged: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Toggle("PicoDecisions routing", isOn: $model.enabled)
                .font(.headline)
                .tint(Brand.purple)
                .accessibilityIdentifier("pico-routing-toggle")

            if model.enabled {
                configuration
                status
                if let recommendation = model.recommendation {
                    DecisionRecommendationView(
                        recommendation: recommendation,
                        disposition: model.disposition,
                        loadLatencyMilliseconds: model.loadLatencyMilliseconds)
                }
            } else {
                Text("Enable Laya to recommend one of the retrieved tools, or no match.")
                    .font(.caption)
                    .foregroundStyle(Brand.textMid)
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.white.opacity(0.8), in: RoundedRectangle(cornerRadius: 16))
        .overlay(RoundedRectangle(cornerRadius: 16).strokeBorder(Brand.border))
        .onChange(of: model.enabled) { configurationChanged() }
        .onChange(of: model.precision) { configurationChanged() }
    }

    private var configuration: some View {
        VStack(alignment: .leading, spacing: 12) {
            Picker("Decision precision", selection: $model.precision) {
                ForEach(DecisionPrecision.allCases) { precision in
                    Text(precision.title).tag(precision)
                }
            }
            .pickerStyle(.segmented)
            .accessibilityIdentifier("pico-routing-precision")

            DisclosureGroup("Acceptance thresholds") {
                VStack(alignment: .leading, spacing: 12) {
                    DecisionThresholdControl(
                        title: "Minimum probability", value: $model.minimumProbability)
                    DecisionThresholdControl(
                        title: "Minimum margin", value: $model.minimumMargin)
                    Text("Zero disables a threshold. These values are uncalibrated; changing them reevaluates the existing recommendation without rerunning inference.")
                        .font(.caption)
                        .foregroundStyle(Brand.textMid)
                }
                .padding(.top, 8)
            }
            .font(.subheadline)
            .tint(Brand.purple)
        }
    }

    private var status: some View {
        VStack(alignment: .leading, spacing: 8) {
            switch model.status {
            case .idle:
                Text("Decision model is idle.")
            case .loading(let message):
                HStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    Text(message)
                }
            case .ready:
                if let message = model.routingError {
                    Text(message).foregroundStyle(.red).textSelection(.enabled)
                    Text("Edit the request to try again. Retrieved tools remain available.")
                }
                if model.isSelecting {
                    HStack(spacing: 8) {
                        ProgressView().controlSize(.small)
                        Text("Choosing from the retrieved candidates…")
                    }
                } else if model.recommendation == nil {
                    Text("Decision model ready. Enter a request to test routing.")
                } else {
                    Text("Decision model ready.")
                }
                if model.recommendation == nil, let latency = model.loadLatencyMilliseconds {
                    Text("Model load: \(latency, format: .number.precision(.fractionLength(1))) ms")
                }
            case .failed(let message):
                Text(message)
                    .foregroundStyle(.red)
                    .textSelection(.enabled)
                Button("Retry decision model", action: configurationChanged)
                    .buttonStyle(.bordered)
            }
        }
        .font(.caption)
        .foregroundStyle(Brand.textMid)
        .accessibilityIdentifier("pico-routing-status")
    }
}

private struct DecisionThresholdControl: View {
    let title: String
    @Binding var value: Double

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(title)
                Spacer()
                Text(value, format: .number.precision(.fractionLength(2)))
                    .monospacedDigit()
            }
            Slider(value: $value, in: 0...1, step: 0.01) {
                Text(title)
            }
            .tint(Brand.purple)
        }
    }
}

private struct DecisionRecommendationView: View {
    let recommendation: RoutingRecommendation
    let disposition: ToolDecisionDisposition?
    let loadLatencyMilliseconds: Double?

    private var selection: ToolDecisionSelection { recommendation.selection }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Divider()
            VStack(alignment: .leading, spacing: 4) {
                Label(dispositionTitle, systemImage: dispositionSymbol)
                    .font(.headline)
                    .foregroundStyle(dispositionColor)
                Text("Raw recommendation: \(candidateName(selection.selectedCandidateID))")
                    .font(.subheadline)
                    .textSelection(.enabled)
                Text("Request: \(recommendation.query)")
                    .font(.caption)
                    .foregroundStyle(Brand.textMid)
                    .textSelection(.enabled)
                if let disposition, !disposition.reasons.isEmpty {
                    ForEach(disposition.reasons, id: \.rawValue) { reason in
                        Text(reasonDescription(reason))
                            .font(.caption)
                            .foregroundStyle(Brand.textMid)
                    }
                }
            }
            .accessibilityElement(children: .combine)

            VStack(alignment: .leading, spacing: 6) {
                if let probability = disposition?.selectedProbability {
                    metric("Selected probability", value: probability.formatted(.percent.precision(.fractionLength(1))))
                }
                if let margin = disposition?.probabilityMargin {
                    metric("Margin over strongest rival", value: margin.formatted(.number.precision(.fractionLength(3))))
                }
                if let tokenCount = selection.inputTokenCount {
                    metric("Input tokens", value: "\(tokenCount)")
                }
                metric("Decision latency (\(recommendation.precision.title))",
                       value: "\(recommendation.latencyMilliseconds.formatted(.number.precision(.fractionLength(1)))) ms")
                if let latency = loadLatencyMilliseconds {
                    metric("Model load", value: "\(latency.formatted(.number.precision(.fractionLength(1)))) ms")
                }
            }

            if let diagnostics = selection.inputDiagnostics {
                inputDiagnostics(diagnostics)
            } else {
                Text(selection.candidateProbabilities.isEmpty
                     ? "Inference skipped: there were no candidates."
                     : "Prompt diagnostics unavailable.")
                    .font(.caption)
                    .foregroundStyle(Brand.textMid)
            }

            if !selection.candidateProbabilities.isEmpty {
                DisclosureGroup("Decision probabilities") {
                    VStack(alignment: .leading, spacing: 8) {
                        ForEach(selection.candidateProbabilities, id: \.optionID) { item in
                            probabilityRow(id: item.optionID, probability: item.probability)
                        }
                        if let probability = selection.noMatchProbability {
                            probabilityRow(id: ToolDecisionSelector.noMatchID, probability: probability)
                        }
                        Text("Model probabilities apply to this candidate set and are not calibrated guarantees. Retrieval scores are shown separately on the tool cards.")
                            .font(.caption)
                            .foregroundStyle(Brand.textMid)
                    }
                    .padding(.top, 8)
                }
                .font(.subheadline)
                .tint(Brand.purple)
            }
        }
    }

    private var dispositionTitle: String {
        switch disposition?.status {
        case .acceptedTool: "Tool accepted by policy"
        case .acceptedNoMatch: "No match accepted by policy"
        case .abstained: "Policy abstained"
        case nil: "Raw recommendation"
        }
    }

    private var dispositionSymbol: String {
        switch disposition?.status {
        case .acceptedTool: "checkmark.circle"
        case .acceptedNoMatch: "minus.circle"
        case .abstained: "pause.circle"
        case nil: "arrow.triangle.branch"
        }
    }

    private var dispositionColor: Color {
        disposition?.status == .abstained ? .orange : Brand.purple
    }

    private func candidateName(_ id: String?) -> String {
        guard let id, id != ToolDecisionSelector.noMatchID else { return "No matching tool" }
        return recommendation.candidates.first(where: { $0.id == id })?.tool.name ?? id
    }

    private func reasonDescription(_ reason: ToolDecisionAbstentionReason) -> String {
        switch reason {
        case .belowMinimumProbability: "Selected probability is below the configured minimum."
        case .belowMinimumMargin: "The margin over the strongest rival is below the configured minimum."
        case .invalidSelection: "The selection did not contain a valid probability distribution."
        }
    }

    private func metric(_ title: String, value: String) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 12) {
            Text(title)
                .foregroundStyle(Brand.textMid)
            Spacer(minLength: 8)
            Text(value)
                .monospacedDigit()
        }
        .font(.caption)
        .accessibilityElement(children: .combine)
    }

    private func probabilityRow(id: String, probability: Double) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Text(candidateName(id))
                .font(.system(.caption, design: .monospaced))
                .frame(maxWidth: .infinity, alignment: .leading)
            if id == (selection.selectedCandidateID ?? ToolDecisionSelector.noMatchID) {
                Image(systemName: "checkmark")
                    .foregroundStyle(Brand.purple)
                    .accessibilityLabel("Raw recommendation")
            }
            Text(probability, format: .percent.precision(.fractionLength(1)))
                .font(.caption)
                .monospacedDigit()
        }
        .accessibilityElement(children: .combine)
    }

    private func inputDiagnostics(_ diagnostics: DecisionInputDiagnostics) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            if diagnostics.wasTruncated {
                Label("Prompt shortened: some decision criteria may have been omitted.",
                      systemImage: "exclamationmark.triangle")
                    .font(.caption)
                    .foregroundStyle(.orange)
            } else {
                Label("Prompt retained without truncation", systemImage: "checkmark.circle")
                    .font(.caption)
                    .foregroundStyle(Brand.textMid)
            }
            DisclosureGroup("Prompt token details") {
                VStack(alignment: .leading, spacing: 8) {
                    tokenRow("Request", tokens: diagnostics.state)
                    tokenRow("Routing instructions", tokens: diagnostics.instructions)
                    ForEach(diagnostics.options, id: \.optionID) { option in
                        tokenRow(candidateName(option.optionID), tokens: option.tokens)
                    }
                    Text("Counts show retained / original tokens after prompt normalization, including component labels and excluding structural markers.")
                        .font(.caption)
                        .foregroundStyle(Brand.textMid)
                }
                .padding(.top, 8)
            }
            .font(.subheadline)
            .tint(Brand.purple)
        }
    }

    private func tokenRow(_ title: String, tokens: DecisionInputDiagnostics.TokenCounts) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Text(title)
                .frame(maxWidth: .infinity, alignment: .leading)
            Text("\(tokens.retainedTokenCount) / \(tokens.originalTokenCount)")
                .monospacedDigit()
            if tokens.wasTruncated {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundStyle(.orange)
                    .accessibilityLabel("Truncated")
            }
        }
        .font(.caption)
        .accessibilityElement(children: .combine)
    }
}
