# Smart Tool Selection

A native, on-device Swift port of LiquidAI's [ColBERT Tool Selection Space](https://huggingface.co/spaces/LiquidAI/colbert-tool-selection) — the same demo, running entirely on Apple Silicon with no server.

> An agent with **151 tools** can't fit them all in one prompt. Type a request and Embedding, ColBERT, or Laya pre-selects the **5 most relevant** tools, so you hand a small candidate set to the LLM instead of dumping every schema into the context window.

<p align="center">
  <img src="Screenshots/home.png" width="49%" alt="Home screen: backend and precision toggles, a search box, and example queries">
  &nbsp;
  <img src="Screenshots/results.png" width="49%" alt="Results: the top-5 ranked tools with match scores and parameter chips">
</p>

## What it does

The app indexes 151 tool definitions spanning 7 domains (e-commerce, devops, travel, support, and more). For each request it scores every tool and shows the top 5 — the candidate set you would route to an LLM instead of all 151 schemas.

Three backends, switchable in the UI, each scoring the full catalog:

- **Embedding** — `LFM2.5-Embedding-350M`: one CLS-pooled vector per tool, ranked by cosine similarity.
- **ColBERT** — `LFM2.5-ColBERT-350M`: one vector per token, ranked by MaxSim late interaction. This is the default, matching the original Space.
- **Laya** — the multilingual decision model through [PicoDecisions](https://github.com/PicoMLX/PicoDecisions): one independent boolean relevance question per tool, ranked by probability of true.

All three return the same top-five tool cards for a downstream LLM to choose from. The demo does not generate arguments or execute tools. Laya scores all 151 tools directly, without an embedding or ColBERT shortlist.

## How it works

Everything runs in-process on the device:

1. **Load** the selected backend. Switching backends releases the previous model and index before loading its replacement.
2. **Score** the catalog. Embedding and ColBERT index each tool's routing text (name, description, parameter names, enum values, and keywords), reuse that cached index, and encode the request with [MLX Swift](https://github.com/ml-explore/mlx-swift) and [MLXEmbedders](https://github.com/PicoMLX/mlx-swift-lm), and rank using [Accelerate](https://developer.apple.com/documentation/accelerate) BLAS. Laya uses the request as shared state and asks one boolean question per tool about whether its described capability is required (falling back to the tool name for an empty description). PicoDecisions processes these questions in batches of 16.
3. **Rank** scores and show the top five. Equal scores retain catalog order. No relevance threshold or single-tool decision is applied.

Each Laya card shows **Relevance**, the probability the tool is useful, and **Confidence**, certainty in either relevance or irrelevance (`max(p, 1 - p)`). Only relevance controls ranking. Neither value is a calibrated accuracy guarantee, and scores from different backends are not directly comparable.

The displayed search latency covers completed scoring and result conversion over the full catalog, excluding model loading and queued GPU waiting. Laya re-evaluates every tool for each request; its full-catalog timing is not comparable to the earlier one-question routing benchmark. Editing or clearing a request and changing backend or precision cancel obsolete searches; stale results are discarded.

## Models & quantization

Models are downloaded from the [Hugging Face Hub](https://huggingface.co/mlx-community) on first run and cached locally; a **Precision** toggle picks the on-device weights. The bf16 checkpoints are ~680–690 MB each. We ran a quant-aware retrieval eval to choose the quantized ship precisions:

| Retriever | Quantized | Size | NDCG@10 retention vs bf16 |
|---|---|---|---|
| **Embedding-350M** | int4 | **195 MB** (3.5× smaller) | **100.0%** |
| **ColBERT-350M** | int8 | **363 MB** (1.9× smaller) | **100.0%** |
| ColBERT-350M | int4 | ~195 MB (3.5× smaller) | 98.7% |

Retention is NDCG@10 against the bf16 baseline, measured over **NanoBEIR** (English) plus **MIRACL** dev judged pools (Spanish / German / Japanese / Arabic).

- **Embedding tolerates int4 losslessly** — CLS pooling averages out quantization noise, so int4 is the recommended precision (3.5× smaller, no measurable retrieval loss, even on Japanese and Arabic).
- **ColBERT's per-token MaxSim is more quant-sensitive** — int8 is lossless and the safe default; int4 retains 98.7% (loss concentrated in English NQ, ~95%) and is the aggressive option when size-bound.

> A useful gotcha from the eval: raw embedding cosine drifts more under int4 (~0.96 vs bf16), but rank metrics ignore those small magnitude shifts — which is why retrieval stays ~lossless. Always measure quantization impact on the task metric (retrieval), not on vector cosine.

## Requirements

- Apple Silicon Mac running macOS 26.4+, or a physical iOS 26.4+ device
- Xcode 26.6+ with Swift 6.3+; development builds also use Xcode 27

## Build & run

1. Open `SmartToolSelection.xcodeproj`.
2. Select the **SmartToolSelection** scheme and run.
3. On first launch the app downloads the selected model from Hugging Face into `~/Library/Application Support/SmartToolSelection/models/` (the full path is printed to the console). Later launches load from that cache.

Pick a backend (Embedding / ColBERT / Laya) and precision (bf16 / int8 / int4 for retrieval, FP16 / FP32 for Laya) at the top of the window; switching either downloads the matching model as needed and rebuilds the index.

The PicoDecisions dependency pins commit `7b94c97695c4592d5067d50a8bd8982a6a26536e`, which includes prompt-truncation diagnostics from [PicoDecisions PR #6](https://github.com/PicoMLX/PicoDecisions/pull/6). That PR is merged; the commit remains pinned for reproducible builds.

## Use Laya

Select **Laya** in the backend picker. FP16 is the default; FP32 is also available. The first use downloads approximately 648 MB from [convaiinnovations/laya-multilingual](https://huggingface.co/convaiinnovations/laya-multilingual) at the validated revision `052592a15d198d9ad47da779604259b10b47b7aa`. Later uses load the cached checkpoint. FP16 and FP32 use the same downloaded files, with the selected precision applied when loading.

The cache is inside the app's Application Support directory under:

```text
SmartToolSelection/models/convaiinnovations/laya-multilingual/
  052592a15d198d9ad47da779604259b10b47b7aa/
    model.safetensors
    encoder/config.json
    rl_agent_config.json
    tokenizer/tokenizer.json
    tokenizer/tokenizer_config.json
```

The console prints the full device-specific path. Keep the nested directories intact if you inspect or copy the checkpoint.

Expand a tool card and **Prompt details** to inspect retained/original token counts. A warning appears if any of the 151 questions loses prompt text. Overlong request state is rejected rather than silently shortened. Search errors leave the model available to retry with another request.

## Test on a physical device

Physical-device correctness, memory, and performance still require measurement.

1. Open `SmartToolSelection.xcodeproj`, select the **SmartToolSelection** scheme, and configure the signing team for your connected device.
2. Select **Laya** and wait for its checkpoint to download and load.
3. Try an ordinary request, a negated request such as `Find my order; do not cancel it`, and a multi-tool request such as `Find a flight, book it, and add it to my calendar`. Compare the top five against Embedding and ColBERT. Missing arguments should not automatically exclude a useful capability.
4. Inspect relevance and confidence separately. Try unrelated and long requests, and inspect any prompt-shortening warning. Every successful search still returns five candidates, including when all scores are low.
5. Record full-catalog search latency, model-load time, and Xcode's process-memory gauge in FP16 and FP32. Clear or edit the request during inference and switch backends to check that obsolete results do not reappear.

The deterministic tests inject fake search and decision models and do not download weights. Set `SMART_TOOL_SELECTION_SKIP_MODEL_LOAD=1` in the scheme's Test environment, or pass `TEST_RUNNER_SMART_TOOL_SELECTION_SKIP_MODEL_LOAD=1` to `xcodebuild test`, to keep the app test host offline. Existing retrieval tests use cached model files and skip when absent. Actual inference needs Apple silicon GPU access. Opt-in full-catalog Laya smoke tests use a local checkpoint directory supplied as `SMART_TOOL_SELECTION_LAYA_MODEL` (or `TEST_RUNNER_SMART_TOOL_SELECTION_LAYA_MODEL` with `xcodebuild`). Run this suite separately from the cached retriever tests to avoid concurrent GPU execution across suites:

```sh
TEST_RUNNER_SMART_TOOL_SELECTION_SKIP_MODEL_LOAD=1 \
TEST_RUNNER_SMART_TOOL_SELECTION_LAYA_MODEL=/absolute/path/to/laya-multilingual \
xcodebuild test -project SmartToolSelection.xcodeproj -scheme SmartToolSelection \
  -destination 'platform=macOS,arch=arm64' \
  -only-testing:SmartToolSelectionTests/LayaCatalogSmokeTests \
  -parallel-testing-enabled NO
```

The smoke run prints full-catalog times, ranked tools, and truncation counts for ordinary, negated, and multi-tool requests. It verifies complete, finite outputs, not benchmark accuracy.

## Development validation

On October 7, 2026, the updated app passed the macOS test suite (18 test functions, with the uncached embedding check and opt-in Laya suite skipped), cached ColBERT inference checks, and an unsigned iOS device build using Xcode 27. A separate live FP16 Laya test completed all 151 questions for each of three requests, with finite relevance/confidence values and zero truncated questions. The Debug full-catalog scoring times were 720, 835, and 821 ms on the development Mac; this three-request smoke run is not a Release latency benchmark.

The capability prompt was selected using these same development examples, not held-out data. It included `search_products` in the top five for the chair request and ranked `book_flight` first for the flight request. However, `Find my order; do not cancel it` ranked unrelated pharmacy/medical tools with probabilities above 0.99. This is a working comparison harness, not evidence that independent Laya relevance scores outperform retrieval or that high confidence establishes correctness. Representative candidate-recall and downstream tool-calling evaluation remain outstanding.

## Credits

- **Models** — [LiquidAI](https://huggingface.co/LiquidAI) LFM2.5-Embedding-350M and LFM2.5-ColBERT-350M, converted to MLX ([model repositories](https://huggingface.co/mlx-community)).
- **Original demo** — LiquidAI's [ColBERT Tool Selection Space](https://huggingface.co/spaces/LiquidAI/colbert-tool-selection).
- **On-device inference** — [MLX-Swift](https://github.com/ml-explore/mlx-swift) and [mlx-swift-lm](https://github.com/PicoMLX/mlx-swift-lm).
- **Laya retrieval** — [PicoDecisions](https://github.com/PicoMLX/PicoDecisions), using Convai Innovations' [multilingual Laya checkpoint](https://huggingface.co/convaiinnovations/laya-multilingual).
