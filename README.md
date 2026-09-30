# Smart Tool Selection

A native, on-device Swift port of LiquidAI's [ColBERT Tool Selection Space](https://huggingface.co/spaces/LiquidAI/colbert-tool-selection) — the same demo, running entirely on Apple Silicon with no server.

> An agent with **151 tools** can't fit them all in one prompt. Type a request and an [LFM2.5](https://huggingface.co/LiquidAI) retriever pre-selects the **5 most relevant** tools, so you hand a small candidate set to the LLM instead of dumping every schema into the context window.

<p align="center">
  <img src="Screenshots/home.png" width="49%" alt="Home screen: backend and precision toggles, a search box, and example queries">
  &nbsp;
  <img src="Screenshots/results.png" width="49%" alt="Results: the top-5 ranked tools with match scores and parameter chips">
</p>

## What it does

The app indexes 151 tool definitions spanning 7 domains (e-commerce, devops, travel, support, and more). For each request it scores every tool and shows the top 5 — the candidate set you would route to an LLM instead of all 151 schemas.

Enable **PicoDecisions routing** to pass those five candidates to the multilingual Laya decision model. It recommends one tool or **No matching tool**, then a configurable acceptance policy accepts the recommendation or abstains. The panel shows decision probabilities, latency, and prompt-truncation details alongside the existing retrieval results. This is a routing test harness: it does not collect arguments, authorize actions, or execute tools.

Two retrievers, switchable in the UI:

- **Embedding** — `LFM2.5-Embedding-350M`: one CLS-pooled vector per tool, ranked by cosine similarity.
- **ColBERT** — `LFM2.5-ColBERT-350M`: one vector per token, ranked by MaxSim late interaction. This is the default, matching the original Space.

## How it works

Everything runs in-process on the device:

1. **Tokenize** the request (via [swift-transformers](https://github.com/huggingface/swift-transformers)).
2. **Encode** the query and each tool's routing text with LFM2.5 — the encoder forward runs on the **Apple Silicon GPU through [MLX-Swift](https://github.com/ml-explore/mlx-swift)**, using [mlx-swift-lm](https://github.com/PicoMLX/mlx-swift-lm)'s `MLXEmbedders` (which gained native LFM2.5 bidirectional-encoder support for this app).
3. **Score** on the CPU with **[Accelerate](https://developer.apple.com/documentation/accelerate)** — a single BLAS call per query over flat, L2-normalized vectors: `cblas_sgemv` for embedding cosine, `cblas_sgemm` + `vDSP_maxv` for ColBERT MaxSim.
4. **Rank** and show the top 5.
5. **Decide**, when PicoDecisions routing is enabled: [PicoDecisions](https://github.com/PicoMLX/PicoDecisions) runs Laya in FP16 or FP32 over the retrieved tool descriptions plus an explicit no-match option.
6. **Apply the acceptance policy** and display the raw recommendation, policy result, and prompt diagnostics. Retrieval scores and decision probabilities remain separate.

The tool index is built once when a model loads; each keystroke only re-encodes the query, so a search takes tens of milliseconds (the per-query latency is shown in the UI).

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

Pick a backend (Embedding / ColBERT) and precision (bf16 / int8 / int4) at the top of the window; switching either downloads the matching model as needed and rebuilds the index.

The PicoDecisions dependency pins commit `7b94c97695c4592d5067d50a8bd8982a6a26536e`, which includes prompt-truncation diagnostics from [PicoDecisions PR #6](https://github.com/PicoMLX/PicoDecisions/pull/6). That PR is the integration's pending upstream dependency; the commit pin also resolves before it merges.

## Enable PicoDecisions routing

Turn on **PicoDecisions routing** below the search field. The first use downloads approximately 648 MB from [convaiinnovations/laya-multilingual](https://huggingface.co/convaiinnovations/laya-multilingual) at the validated revision `052592a15d198d9ad47da779604259b10b47b7aa`. Later uses load the cached checkpoint. FP16 and FP32 use the same downloaded files, with the selected precision applied when loading.

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

Both **Acceptance thresholds** start at zero, disabling their gates. Increase **Minimum probability** or **Minimum margin** to experiment with abstention; the margin compares the selected option with its strongest rival, including no match. Changing a threshold reevaluates the current recommendation without another inference call. These thresholds and model probabilities are uncalibrated, so acceptance does not establish that a tool is appropriate or authorized.

Expand **Decision probabilities** to compare the five candidates and no match. Expand **Prompt token details** to inspect retained/original token counts for the request, routing instructions, and each candidate description. A shortened-description warning means decision criteria may have been omitted. Overlong request state is rejected rather than silently shortened; the retrieved cards remain available if decision inference fails.

## Test on a physical device

This integration provides the device test harness. Physical-device correctness, memory, and performance results still need to be collected.

1. Open `SmartToolSelection.xcodeproj` in Xcode and select the **SmartToolSelection** scheme.
2. Select your connected iPhone or iPad as the run destination and configure the app target's signing team, then run it.
3. Allow the selected retriever to download and index the catalog. Enable **PicoDecisions routing** and wait for the decision model to report ready.
4. Try an ordinary request such as `show me cheap blue outdoor chairs`, a negated request such as `Do not cancel my order`, and an unrelated request such as `Tell me a bedtime story`. Compare the retrieved candidates with the raw recommendation and no-match probability; these inputs probe behavior rather than guarantee particular outputs.
5. Expand **Prompt token details** and inspect candidates with long descriptions. Try a long request to check the capacity-error path. Record input-token counts, any omitted tokens, model-load time, decision latency, and the separately reported retrieval latency. Model-load time excludes the download, and decision latency excludes waiting for other queued GPU work.
6. Switch **Decision precision** between FP16 and FP32 and repeat the same requests. Use Xcode's memory gauge while loading and routing to compare memory use. Edit or clear a request during inference, and switch retrieval backends, to check that older recommendations do not reappear.

The routing unit tests use a fake decision engine and do not require Hugging Face downloads. To keep the test host offline, set `SMART_TOOL_SELECTION_SKIP_MODEL_LOAD=1` in the scheme's Test environment, or pass `TEST_RUNNER_SMART_TOOL_SELECTION_SKIP_MODEL_LOAD=1` to `xcodebuild test`. Existing retrieval tests use cached model files and skip when those files are absent. Run the **SmartToolSelectionTests** target through Xcode; actual Metal inference requires an Apple Silicon Mac or physical device.

## Credits

- **Models** — [LiquidAI](https://huggingface.co/LiquidAI) LFM2.5-Embedding-350M and LFM2.5-ColBERT-350M, converted to MLX ([model repositories](https://huggingface.co/mlx-community)).
- **Original demo** — LiquidAI's [ColBERT Tool Selection Space](https://huggingface.co/spaces/LiquidAI/colbert-tool-selection).
- **On-device inference** — [MLX-Swift](https://github.com/ml-explore/mlx-swift) and [mlx-swift-lm](https://github.com/PicoMLX/mlx-swift-lm).
- **Decision routing** — [PicoDecisions](https://github.com/PicoMLX/PicoDecisions), using Convai Innovations' [multilingual Laya checkpoint](https://huggingface.co/convaiinnovations/laya-multilingual).
