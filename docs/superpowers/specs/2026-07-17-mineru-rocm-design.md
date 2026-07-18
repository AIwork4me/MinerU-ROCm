# MinerU-ROCm — Design Specification

| | |
|---|---|
| **Date** | 2026-07-17 |
| **Status** | Approved (design); spec under review — **revised 2026-07-17 with full VLM research** |
| **Author** | Claude (AIwork4me) |
| **Upstream** | [opendatalab/MinerU](https://github.com/opendatalab/MinerU) v3.4.2 (74.9k★) |
| **Platform slot** | `OmniDocBench-AMD/hub/registry.yaml` → `mineru2.5 → AIwork4me/MinerU-ROCm` |
| **Lives in** | this repo, `docs/superpowers/specs/` |

---

## 0. TL;DR

**MinerU-ROCm** is the AMD port of OpenDataLab's MinerU, shipping **two models in one repo**, slotted into the OmniDocBench-AMD platform:

| Model | What | Upstream engine | Official OmniDocBench v1.6 | Our ROCm engine(s) |
|---|---|---|---|---|
| **MinerU 3.4 `pipeline`** | PP-DocLayoutV2 + UniMERNet-small + PytorchPaddleOCR (PP-OCRv6) + SLANet/UNet tables | pure PyTorch (`MINERU_DEVICE_MODE=cuda`) + ONNX tables (CPU on ROCm) | **86.47** | pure PyTorch on ROCm (no engine choice) |
| **MinerU2.5-Pro-2605-1.2B** | Qwen2-VL 1.156B VLM, **two-step** layout→extract inference | vLLM (default) / LMDeploy / transformers / OpenAI-compatible | **95.75 (#1)** | vLLM-on-ROCm and/or transformers (see §8 decision) |

**Build order:** scaffold (done) → pipeline (Linux-verified) → VLM → precision comparison.

**Platforms:** `linux-rocm` verified **in this environment** — **4× AMD gfx1100 (Radeon PRO W7900, 48 GB VRAM each), all idle**, enough to run the pipeline + both VLM engine probes concurrently on separate GPUs; `windows-hip` documented in a self-contained handoff (§14) for a colleague on Strix Halo.

**Precision bar (honest):** reproduce the official OmniDocBench v1.6 number on the **full 1651-page set on gfx1100**, with real artifacts + provenance. Not bit-exact-to-upstream-CUDA. Repo is **"evaluation-backed,"** mirroring HunyuanOCR-ROCm.

---

## 1. Background & motivation

MinerU is the leading open-source document parser (74.9k★). As of v3.4 it has become **dramatically more portable**:

- The classic `pipeline` is now **end-to-end pure-PyTorch** (OCR/layout/formula). PaddleOCR was ported to `PytorchPaddleOCR` (PP-OCRv6); the license-restricted models (DocLayout-YOLO, mfd_yolov8, layoutreader) were removed in v3.0. `get_device()` keys off `torch.cuda.is_available()` → **true on ROCm** → pipeline lands on GPU automatically. **PaddlePaddle is no longer a blocker.** (Only the 3 table models are ONNX — they fall back to CPU on ROCm; see §7.)
- **MinerU2.5-Pro-2605-1.2B** is a stock **Qwen2-VL** (`Qwen2VLForConditionalGeneration`) at **#1 on OmniDocBench v1.6 (95.75)**, using **two-step decoupled inference** (§8).

**There is already a community-contributed ROCm port guide in MinerU master**, written for the 7900 XTX (gfx1100 — our exact chip family): [`docs/zh/usage/acceleration_cards/AMD.md`](https://github.com/opendatalab/MinerU/blob/master/docs/zh/usage/acceleration_cards/AMD.md) + [Discussion #3662](https://github.com/opendatalab/MinerU/discussions/3662) (per-sub-model perf) + the applied-patches fork [`healy-hub/MinerU-AMD-RDNA`](https://github.com/healy-hub/MinerU-AMD-RDNA). It is community content (no upstream ROCm CI), but it gives us a **working recipe + concrete patches** (notably the vLLM `Qwen2VisionPatchEmbed` Conv3d-BF16 GEMM/Triton patch) to build on rather than rediscover. Three upstream issues also ask for ROCm/DirectML ([#313](https://github.com/opendatalab/MinerU/issues/313), [#2013](https://github.com/opendatalab/MinerU/issues/2013), [#4655](https://github.com/opendatalab/MinerU/issues/4655)); all closed without upstream action.

**This project's contribution = the missing evaluation data** (full OmniDocBench v1.6 numbers on AMD) + a conformant, badge-backed per-model repo that wraps that recipe and verifies it end-to-end.

---

## 2. Goals & non-goals

**Goals**
- G1. Run **both** MinerU models on AMD gfx1100/ROCm.
- G2. Reproduce official OmniDocBench v1.6 (86.47 pipeline; 95.75 VLM) on the full 1651-page set, with artifacts + provenance.
- G3. Ship a conformant, CI-green, contributor-friendly per-model repo (quality bar = upstream + sibling repos HunyuanOCR-ROCm / Unlimited-OCR-ROCm).
- G4. Provide a runnable Windows-hip path + handoff doc for parallel verification by a colleague.

**Non-goals**
- N1. Bit-exact parity with upstream CUDA (impossible — no same-engine CUDA control; official numbers may be a third engine).
- N2. Re-implementing MinerU's models. We **wrap** upstream `mineru[all]` + `mineru-vl-utils` + the upstream VLM weights.
- N3. Windows verification in this environment (no Windows hardware — handoff only).
- N4. SGLang (dropped by upstream Sep 2025; plus `sgl-kernel` CUDA-only + Triton-attention HSAIL still blocks on gfx1100 per 2026-07-15).

---

## 3. Scope

### 3.1 Two models, one repo
"分别搞定 MinerU 3.4 pipeline 和 MinerU2.5-Pro-2605-1.2B" → **one repo hosting two adapters**, distinguished by the `--backend` knob (§9). `model_card.json` (VLM, primary) + `model_card.pipeline.json`.

### 3.2 Platforms
- **`linux-rocm`** — verified here on **4× gfx1100 (Radeon PRO W7900, 48 GB VRAM, 96 CU each), all idle**; ROCm 7.2.1; torch 2.9.1+rocm; vLLM 0.16.1.dev0 rocm; transformers 4.57.6. Four idle GPUs let the pipeline and both VLM probes run concurrently (§15).
- **`windows-hip`** — path + setup + docs built here; **verification handed off** to a colleague on Strix Halo (§14).

### 3.3 Sequence
scaffold (done) → **pipeline + both VLM engine probes run concurrently** on separate GPUs (4 idle) → **promote the VLM winner** + precision comparison.

---

## 4. Precision framework — "evaluation-backed," not "precision-aligned"

Learned from HunyuanOCR-ROCm: MinerU's *official* 95.75 may be measured on a different engine than what's portable to ROCm (upstream serves via vLLM/LMDeploy; the OmniDocBench README does not state the engine). So **"全量对齐" = reproduce the official number on gfx1100 on the full set, with provenance naming the engine we used.** We report multiple backends and their deltas explicitly (HunyuanOCR reported vLLM 94.81 / transformers 94.11 / llama.cpp 92.09).

**Pass tolerance:** VLM Overall within **0.5 pp** of official; pipeline Overall within **1.0 pp**. Below → investigate, don't publish as verified.

---

## 5. System architecture

### 5.1 Dispatcher + two adapter modules
A single `adapter/run_adapter.py` is the **contract entrypoint** (engine invokes it as a subprocess). Thin dispatcher branching on `--backend`:

```
run_adapter.py (dispatcher; contract signature + _run_stats.json write)
   ├── backend == "pipeline"         → pipeline_adapter._infer()   (in-process mineru API on cuda)
   ├── backend == "vlm-vllm"         → vlm_adapter._infer()        (mineru-vl-utils → vLLM-on-ROCm server)
   ├── backend == "vlm-transformers" → vlm_adapter._infer()        (mineru-vl-utils → transformers server)
   └── backend == "smoke"            → placeholder (CI, no GPU)
```

Rejected: one giant `_infer` switch (mashes stacks); two repos (contradicts one-project directive + single registry entry).

### 5.2 Where MinerU-ROCm fits
```
OmniDocBench-AMD engine (stage_infer → stage_score → stage_publish)
   │ subprocess: python adapter/run_adapter.py --img-dir ... --out-dir ... --platform ...
   ▼
MinerU-ROCm adapter → out_dir/<stem>.md + out_dir/_run_stats.json   (R1: filesystem boundary; engine never imports adapter)
   │ pipeline: mineru[all] in-process on cuda   (MINERU_DEVICE_MODE=cuda)
   │ VLM: mineru-vl-utils two-step → persistent server (vLLM-on-ROCm / transformers)
   ▼
engine stage_score (OmniDocBench pdf_validation, eval-venv Py3.11) → metric_result.json
engine stage_publish (full-set enforcement + provenance + check_conformance → badge)
```

Adapter ends at `.md` + `_run_stats.json`. Eval/scoring/CDM/provenance are engine-owned.

---

## 6. Repository structure

Rendered from the platform cookiecutter template (done 2026-07-17), then refined:

```
MinerU-ROCm/
  adapter/
    run_adapter.py            # dispatcher: branches on --backend
    adapter_config.py         # defaults: backend, server_url, weights_dir, api_model_name
    pipeline_adapter.py       # backend=pipeline: in-process mineru API on cuda        [Phase 1]
    vlm_adapter.py            # backend=vlm-*: drives mineru-vl-utils two-step         [Phase 2]
    setup/{00-install-deps.sh, 00-install-deps.ps1, .env.local.example}
  eval/configs/omnidocbench_v16.yaml
  examples/
    demo.png, run_demo.sh                 # pipeline smoke
    serve_vlm_vllm.sh                     # vLLM-on-ROCm server (AMD.md recipe + patches)
    serve_vlm_transformers.sh             # transformers OpenAI-compatible server (fallback)
  results/omnidocbench/v16/
    linux-rocm/{pipeline, vlm-vllm, vlm-transformers}/
    windows-hip/                          # colleague (§14)
  model_card.json                         # VLM (primary)
  model_card.pipeline.json                # pipeline
  patches/{vllm, onnx}/                   # Conv3d GEMM patch + platform detection + ONNX ROCm EP
  docs/{backends,how-it-works,known-gaps,reproducibility,benchmark-methodology,HANDOFF-windows-hip}.md
  docs/superpowers/{specs,plans}/
  README.md, README.zh-CN.md              # 5 required sections each
  pyproject.toml                          # depends on omnidocbench-amd
  .github/workflows/ci.yml                # CPU: pytest + check_conformance
```

---

## 7. Pipeline adapter (`backend=pipeline`)

Wraps upstream **`mineru[all]`** on ROCm. Loads the pipeline **once** via the in-process `mineru` Python API (not a subprocess per page), warms sub-models on `cuda`, loops page images → Markdown:

| Stage | Upstream class | Backbone | Runtime | ROCm |
|---|---|---|---|---|
| Layout | `PPDocLayoutV2LayoutModel` | RT-DETR + LayoutLMv3 + GlobalPointer (25 classes, joint detection + reading order) | PyTorch | pure torch ✓ |
| Formula recog | `UnimernetModel` (`unimernet_hf_small_2503`) | Swin encoder + mBART decoder | PyTorch | ✓ |
| OCR | `PytorchPaddleOCR` | PP-OCRv6 (PyTorch port of PaddleOCR) | PyTorch | ✓ |
| Table (wireless) | SLANet-Plus | SLANet, 488×488 | **ONNX** | **CPU fallback** (see below) |
| Table (wired) | UNet-Structure | UNet ruling-line seg | **ONNX** | **CPU fallback** |
| Table cls | PP-LCNet x1.0 | wired vs wireless | **ONNX** | **CPU fallback** |
| Reading order | post-processing (GlobalPointer + XY-cut) | — | — | — |

- **Device:** `MINERU_DEVICE_MODE=cuda` (explicit).
- **Output (R4):** formulas `$…$`/`$$…$$`; tables HTML; reading order = document order; images `![](path)`.
- **Tables run on CPU by default (correct, slow):** the 3 table ONNX models silently fall to `CPUExecutionProvider` because `onnxruntime_provider.py` only enables `CUDAExecutionProvider`. **Accuracy unaffected; throughput is.** Optional fix: patch the selector to allow `ROcmExecutionProvider` + pin `onnxruntime-rocm ≤ 1.22` (legacy ROCm EP removed in ORT 1.23) — `patches/onnx/`.
- **Hard avoid:** `MINERU_FORMULA_CH_SUPPORT=true` (pulls native-PaddlePaddle `pp_formulanet_plus_m`, the one ROCm blocker).
- **Verified 7900 XTX throughput (Discussion #3662):** UniMERNet 106 it/s, PytorchPaddleOCR 422 it/s (rec) / 14 it/s (det) — unmodified. The stale "ROCm 2× slower than CPU" ([#2013](https://github.com/opendatalab/MinerU/issues/2013)) predates the PyTorch OCR port; expected resolved.

All pipeline weights: umbrella repo [`opendatalab/PDF-Extract-Kit-1.0`](https://huggingface.co/opendatalab/PDF-Extract-Kit-1.0) (`mineru-models-download -s huggingface -m pipeline`, via hf-mirror).

---

## 8. VLM adapter (`backend=vlm-vllm` / `vlm-transformers`)

> **Engine decision (resolved 2026-07-18): probe BOTH engines in parallel on separate GPUs, promote whichever hits the 95.75 tolerance first to primary.** Four idle GPUs make this free; it hedges the vLLM-on-ROCm numerics risk against the transformers logits-processor-porting risk. Both engines are reported regardless.

The VLM is **not** a single forward → Markdown. MinerU2.5-Pro uses **two-step decoupled inference** ([`opendatalab/mineru-vl-utils`](https://github.com/opendatalab/mineru-vl-utils), `mineru_client.py:two_step_extract`):

1. **Layout pass** — page resized to fixed ~1036×1036, one forward; emits `<|box_start|>x1 y1 x2 y2<|box_end|><|ref_start|>{TYPE}<|ref_end|>{angle}` per block (coords 0–1000; TYPE ∈ text/title/table/image/code/equation…; angle ∈ rotate_up/right/down/left).
2. **Extract pass** — each block cropped from the **high-res** original, deskewed if angled, re-fed with a type-specific prompt (`Text Recognition:` / `Table Recognition:` / `Formula Recognition:`). **N+1 forwards per page.**

So the adapter **drives `mineru-vl-utils`** (the upstream inference lib), not a hand-rolled prompt. Tables come out as **OTSL** (One-Token Structured Language, token IDs 151661–151667) → deterministically converted to HTML by `mineru_vl_utils/post_process/otsl2html.py`. Text/formula → Markdown wrapped in `<|md_start|>`/`<|md_end|>` (cross-page continuation `<|txt_contd|>`).

**`MinerULogitsProcessor` (precision-critical):** a DeepseekOCR-style **no-repeat-ngram** processor (`no_repeat_ngram_size=100`), auto-registered with vLLM via `--logits-processors mineru_vl_utils:MinerULogitsProcessor`. Same pattern as the Unlimited-OCR NGram (35/128) recipe. **Mandatory for fidelity** — without it the output diverges.

- **`vlm-vllm` (highest fidelity, pre-plumbed):** serve via vLLM-on-ROCm. Two-step + logits processor + OTSL→HTML are all wired for the vLLM engine in `mineru-vl-utils`. ROCm work = the `AMD.md` recipe: platform-detection patch **plus** the `Qwen2VisionPatchEmbed` Conv3d-BF16 MIOpen-gap patch (Triton patchify **or** 5D-GEMM; GEMM ~15% faster). Fork [`healy-hub/MinerU-AMD-RDNA`](https://github.com/healy-hub/MinerU-AMD-RDNA) already carries these — diff against it. Verified ~1.99 it/s two-step on 7900 XTX.
- **`vlm-transformers` (controlled fallback):** stock `Qwen2VLForConditionalGeneration` via HF `transformers` (loads on our existing torch 2.9.1 / transformers 4.57.6 — no rebuild). Cost: must **port `MinerULogitsProcessor` into the `generate` loop** and drive `mineru-vl-utils`'s two-step against a transformers engine. Best debuggability; sidesteps vLLM entirely.

**Required env:** `PYTORCH_ROCM_ARCH=gfx1100`, `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`, `MINERU_MODEL_SOURCE=modelscope` (or huggingface via hf-mirror). Weights: [`opendatalab/MinerU2.5-Pro-2605-1.2B`](https://huggingface.co/opendatalab/MinerU2.5-Pro-2605-1.2B) (HF lowercase; ModelScope `OpenDataLab/…` capitalized).

---

## 9. Backend matrix

| `--backend` | Model | linux-rocm | windows-hip |
|---|---|---|---|
| `smoke` | — | no-GPU placeholder (CI) | no-GPU placeholder |
| `pipeline` | 3.4 pipeline | mineru on cuda (ONNX tables on CPU) | mineru on DirectML/ONNX or CPU fallback |
| `vlm-vllm` | 2.5-Pro | vLLM-on-ROCm (AMD.md patches) | llama.cpp-GGUF (HIP/Vulkan) |
| `vlm-transformers` | 2.5-Pro | transformers (logits processor ported) | transformers-DirectML |

`adapter_config.py::BACKEND` + `--backend` select the path; the engine interface is unchanged.

---

## 10. Evaluation & precision protocol

- **Full set:** OmniDocBench v1.6, 1651 pages, via the platform engine (`stage_infer` → `stage_score` [eval-venv Py3.11] → `stage_publish`).
- **Composite:** `Overall = ((1 − Text_EditDist) × 100 + Table_TEDS + Formula_CDM) / 3`.
- **Result sets** under `results/omnidocbench/v16/linux-rocm/`: `pipeline/`, `vlm-vllm/`, `vlm-transformers/`.
- **Comparison table** in README:

  | Backend | Overall ↑ | Text Edit ↓ | Formula CDM ↑ | Table TEDS ↑ | TEDS-S ↑ | Read-Order ↓ |
  |---|---|---|---|---|---|---|
  | _official_ VLM | 95.75 | 0.036 | 97.45 | 93.42 | 95.92 | 0.120 |
  | ours vlm-vllm | _TBD_ | | | | | |
  | ours vlm-transformers | _TBD_ | | | | | |
  | _official_ pipeline | 86.47 | — | — | — | — | — |
  | ours pipeline | _TBD_ | | | | | |

- **Gate:** PASS within tolerance (§4); `publish` enforces `limit_pages=null`.

---

## 11. Venv & storage isolation (mandatory)

`/workspace` = **10 GB NFS, code only**. Heavy → `/root` (2 TB free).
- Repo code: `/workspace/MinerU-ROCm` (small). ✓
- Venvs on `/root`: `MinerU-ROCm-venv` (`mineru[all]` + ROCm torch); VLM venv (vLLM/transformers).
- Weights: `${HF_HOME:-/root/.cache/huggingface}` via hf-mirror. Outputs/logs: `/root/ocr-eval/`.
- **Model venv (Py3.12 OK) separate from eval-venv (Py3.11).**
- Avoid documented landmines: no venv/wheel/source/weights on a real `/workspace` path.

---

## 12. Error handling (contract iron rules)
- **R2 — per-page failure → record, continue, never raise.** `try/except` per page; append `failed: <reason>` `PageStatus`; proceed.
- **Server-down (VLM):** record all pages `failed`, don't raise.
- **`smoke` backend** keeps the repo runnable in CI without a GPU.
- Adapter must not import eval/scoring code, run CDM, or write engine-owned JSON.

---

## 13. Testing strategy
- **Conformance:** `check_conformance.py MinerU-ROCm` exit 0 (adapter, eval config, declared result dirs non-empty, bilingual READMEs w/ 5 sections, `examples/` non-empty, `pyproject` depends on `omnidocbench-amd`, `model_card.json` valid). CI on every push.
- **Unit (CPU):** dispatcher routing, config layering, output conventions, `_run_stats.json` shape.
- **Smoke:** `examples/run_demo.sh` → `.md` on `examples/demo.png`.
- **Eval gate:** full-set Overall within tolerance before any `verified` badge.

---

## 14. Windows-hip handoff (deliverable for colleague)

> Linux verified by Claude here; Windows built + documented here, verified in parallel by a colleague on Strix Halo. Mirrors the PaddleOCR-VL Track A/B split.

Self-contained `docs/HANDOFF-windows-hip.md`:
1. **Target:** Ryzen AI MAX+ 395 (Strix Halo), Windows, DirectML.
2. **Pipeline:** `pip install -U "mineru[all]"`; tables via `onnxruntime-directml` (`DmlExecutionProvider`, Microsoft Olive) — fixes the CPU-fallback the Linux side optionally patches. Ref: https://ryzenai.docs.amd.com/en/latest/gpu/ryzenai_gpu.html.
3. **VLM:** `transformers`-DirectML or `llama.cpp` GGUF (HIP/Vulkan); drive `mineru-vl-utils` two-step.
4. **Run:** `python adapter/run_adapter.py --img-dir <pages> --out-dir results/omnidocbench/v16/windows-hip/<backend> --platform windows-hip --backend {pipeline|vlm-vllm|vlm-transformers}`.
5. **Score:** platform engine `stage_score` (Py3.11 eval-venv).
6. **Land artifacts** + update `model_card.json` `badge.windows-hip`; ping Claude to update `hub/registry.yaml`.
7. **Claude provides:** platform-aware dispatcher, Windows setup stub, eval config, badge mechanics, this handoff. Colleague provides the Windows run + artifacts.

---

## 15. Phased implementation plan (high level — detailed via writing-plans)

- **Phase 0 — Repo spine (mostly done):** cookiecutter render ✓; refactor `run_adapter.py` → dispatcher + `pipeline_adapter` + `vlm_adapter`; add `model_card.pipeline.json`; wire serve scripts; CI green with `smoke`; `check_conformance` passes.
- **Phase 1 — Pipeline on Linux ROCm (GPU 3):** `mineru[all]` venv on /root; in-process API on cuda; per-page img→md; full eval → **86.47**; write Windows handoff doc. Accept ONNX tables on CPU (correct). Update `model_card.pipeline.json`.
- **Phase 2 — VLM, both engines probed in parallel** (4 idle GPUs ⇒ no contention):
  - **2a — vLLM-on-ROCm (GPU 0–1):** AMD.md Conv3d + platform-detection patches (try patching the installed vLLM 0.16.1 first, rebuild only if needed); drive `mineru-vl-utils` two-step (logits processor pre-plumbed); full eval → `vlm-vllm/`.
  - **2b — transformers (GPU 2):** stock Qwen2-VL on torch 2.9.1 / transformers 4.57.6; port `MinerULogitsProcessor` into `generate`; drive two-step; full eval → `vlm-transformers/`.
  - Phase 1 + 2a + 2b can run concurrently (independent adapter modules + separate GPUs).
- **Phase 3 — Promote + finalize:** compare both VLM engines; **promote the winner to primary `model_card.json`** (other stays as a reported backend); finalize badges + `hub/registry.yaml` + bilingual READMEs + comparison table.

---

## 16. Known gaps & risks

1. **Two-step fidelity + `MinerULogitsProcessor`** — both mandatory for 95.75. vLLM path has them pre-plumbed; **transformers path must port the no-repeat-100-gram processor into `generate`** (non-trivial). Mitigation: drive `mineru-vl-utils` on both engines.
2. **vLLM Conv3d-BF16 MIOpen gap (perf, not correctness)** — `Qwen2VisionPatchEmbed` has no optimized BF16 Conv3d kernel on RDNA3 → ~12 s/forward fallback. Fix = AMD.md Triton/GEMM patch (in `healy-hub/MinerU-AMD-RDNA`). Needed for a 1651-page eval to finish in reasonable time.
3. **vLLM-on-ROCm precision gap** — HunyuanOCR-class ~0.5–1 pp cross-engine delta + documented EOS/first-token risks. Mitigation: report both engines.
4. **Build deps (vLLM path)** — AMD.md wants torch 2.10 nightly rocm7.0 + vLLM from source + aiter; we have torch 2.9.1 rocm7.2 + vLLM 0.16.1.dev0. Try patching the installed vLLM first; rebuild only if patches don't apply.
5. **ViT max_pixels = 1,605,632 (~1266×1266)** — below the >14k-patch threshold that triggered HunyuanOCR's ViT nondeterminism; likely **does not** apply. Verify.
6. **ONNX tables on CPU** — correct but slow; patch for ROCm EP only if throughput matters (pipeline perf, not accuracy).
7. **CAD/vector PDF regression** in 2605 weights ([Discussion #5091](https://github.com/opendatalab/MinerU/discussions/5091)) — only relevant if CAD drawings are in scope (OmniDocBench v1.6 is mostly not). Consider testing both 2604 and 2605.
8. **windows-hip unverified** here → `community-wanted` until colleague lands artifacts.
9. **License** — repo Apache-2.0; confirm compatibility with MinerU's upstream license before distributing pipeline-wrapped builds.

---

## 17. Success criteria (definition of done)
- [ ] Both models run on gfx1100.
- [ ] Full 1651-page Overall within tolerance (pipeline ≤1.0 pp of 86.47; VLM ≤0.5 pp of 95.75).
- [ ] `check_conformance` passes; CI green.
- [ ] Real artifacts + provenance under `results/omnidocbench/v16/linux-rocm/`.
- [ ] Bilingual READMEs with comparison table + reproducibility.
- [ ] Windows-hip handoff doc delivered; `hub/registry.yaml` `mineru2.5` updated.
- [ ] `model_card.json` + `model_card.pipeline.json` schema-valid.

---

## 18. Open questions
- Q1. ~~Engine order~~ — **resolved 2026-07-18: both engines probed in parallel (4 idle GPUs), promote the winner (§8, §15).**
- Q2. Whether the installed vLLM 0.16.1.dev0 accepts the Conv3d/platform patches or requires a from-source rebuild.
- Q3. Whether OmniDocBench v1.6 contains CAD/vector pages that trigger the 2605 regression (→ test 2604 too).
- Q4. MinerU's Python requirement for `mineru[all]` vs the platform's Py3.11 eval-venv.
- Q5. Pipeline submetric breakdown for 86.47 (not published) — obtained from our own eval.

---

## 19. References
- Upstream: [opendatalab/MinerU](https://github.com/opendatalab/MinerU) v3.4.2; [`model_init.py`](https://github.com/opendatalab/MinerU/blob/master/mineru/backend/pipeline/model_init.py), [`config_reader.py`](https://github.com/opendatalab/MinerU/blob/master/mineru/utils/config_reader.py), [`onnxruntime_provider.py`](https://github.com/opendatalab/MinerU/blob/master/mineru/model/table/rec/onnxruntime_provider.py)
- **ROCm port guide:** [`docs/zh/usage/acceleration_cards/AMD.md`](https://github.com/opendatalab/MinerU/blob/master/docs/zh/usage/acceleration_cards/AMD.md); [Discussion #3662](https://github.com/opendatalab/MinerU/discussions/3662); fork [`healy-hub/MinerU-AMD-RDNA`](https://github.com/healy-hub/MinerU-AMD-RDNA)
- VLM: [HF `opendatalab/MinerU2.5-Pro-2605-1.2B`](https://huggingface.co/opendatalab/MinerU2.5-Pro-2605-1.2B); [`opendatalab/mineru-vl-utils`](https://github.com/opendatalab/mineru-vl-utils) (`two_step_extract`, `MinerULogitsProcessor`, `otsl2html.py`); paper [arXiv:2604.04771](https://arxiv.org/abs/2604.04771)
- Pipeline weights: [`opendatalab/PDF-Extract-Kit-1.0`](https://huggingface.co/opendatalab/PDF-Extract-Kit-1.0)
- Benchmarks: [opendatalab/OmniDocBench](https://github.com/opendatalab/OmniDocBench) (v1.6_full leaderboard)
- Issues: [#313](https://github.com/opendatalab/MinerU/issues/313), [#2013](https://github.com/opendatalab/MinerU/issues/2013), [#4655](https://github.com/opendatalab/MinerU/issues/4655); [#5091](https://github.com/opendatalab/MinerU/discussions/5091) (CAD regression)
- Platform: this org's `OmniDocBench-AMD` (contracts/engine/template/registry); siblings `HunyuanOCR-ROCm`, `Unlimited-OCR-ROCm`.
