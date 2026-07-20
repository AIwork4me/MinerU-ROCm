# Reproducibility

A score is only meaningful if someone else can reproduce it from the committed repo. The standalone `mineru-rocm` CLI is the primary path; `reproducibility.lock.yaml` is the single source of truth (pinned commits, byte-exact weight/GT SHAs, scorer commit, both venvs' environment, the metric formula, the official anchors, and the ROCm recipe).

## Results (OmniDocBench v1.6, full 1651 pages, gfx1100 / ROCm 7.2)

| Backend | Overall | Text EditDist ↓ | Formula CDM ↑ | Table TEDS ↑ | read-order EditDist |
|---|---:|---:|---:|---:|---:|
| MinerU 3.4 pipeline | **86.48** | 0.0566 | 83.07 | 82.04 | 0.1534 |
| MinerU2.5-Pro VLM (vLLM-on-ROCm) | **95.46** | 0.0360 | 96.46 | 93.54 | 0.1236 |

Official anchors (upstream README "Local Deployment" table): pipeline **86.47** (Δ +0.01 pp), vlm-engine **95.30** (Δ +0.16 pp). These are contextual reference anchors, **not** a controlled CUDA-vs-ROCm comparison (the upstream table does not pin identical hardware, model revision, build, or decoding config).

**Overall** = `((1 − text_EditDist) × 100 + formula_CDM × 100 + table_TEDS × 100) / 3`, OmniDocBench `page.ALL` aggregation; reading-order EditDist is reported separately and is **not** part of Overall.

## The gfx1100 ROCm recipe

- GPU: AMD gfx1100 (Radeon PRO W7900, 48 GB). ROCm 7.2, bf16.
- `HSA_OVERRIDE_GFX_VERSION`:
  - **pipeline backend** (in-process PyTorch): **not required** — PyTorch-ROCm auto-detects gfx1100.
  - **VLM backend via vLLM**: **required** — `export HSA_OVERRIDE_GFX_VERSION=11.0.0` (observed with the tested vLLM-on-ROCm build; tested on gfx1100 only — not claimed for other RDNA3 variants or architectures).
- Performance: pipeline ~3–6 s/page (no patches). VLM via vLLM is **correct without patches but slow** (~15–16 s/page); for speed, community Triton patches for the `qwen2_vl.py` Conv3d exist upstream — see the upstream `docs/zh/usage/acceleration_cards/AMD.md`.

## The two venvs (reality)

Inference and scoring need different environments (MinerU pulls a ROCm torch; OmniDocBench's scorer pins its own deps and uses **no** torch for CDM). Use two venvs:

- **infer venv** — Python 3.11/3.12, `mineru[all]` 3.4.4 (+ ROCm torch wheel); for the VLM also `mineru_vl_utils` 1.0.5 + a vLLM-on-ROCm wheel. Versions pinned in the lock.
- **scorer venv** — OmniDocBench's pinned scoring deps (`bs4`, `apted`, `Levenshtein`, `pylatexenc`, `scipy`, …). CDM shells out to `pdflatex`/`magick`.

## Reproduce (commands; substitute your own paths)

Paths below are placeholders — set them to your own (no machine-private defaults are assumed; the scorer repo + venv must be supplied explicitly):

```bash
export GT_JSON=/path/to/OmniDocBench.json
export IMAGES_DIR=/path/to/images
export OMNIDOCBENCH_REPO=/path/to/OmniDocBench        # the scorer checkout (lock: omnidocbench.scorer_commit)
export SCORER_VENV=/path/to/OmniDocBench/.venv         # the scorer venv (separate from the infer venv)
export PRED_DIR_PIPELINE=/path/to/out-pipeline
export PRED_DIR_VLM=/path/to/out-vlm
```

```bash
# 1. pipeline (infer venv)
export HSA_OVERRIDE_GFX_VERSION=11.0.0   # harmless for pipeline; required by the tested VLM (vLLM-on-ROCm) build on gfx1100
mineru-rocm predict --backend pipeline \
  --gt-json "$GT_JSON" --images-dir "$IMAGES_DIR" \
  --pred-dir "$PRED_DIR_PIPELINE" --platform linux-rocm
mineru-rocm validate --gt-json "$GT_JSON" --pred-dir "$PRED_DIR_PIPELINE"
mineru-rocm score --gt-json "$GT_JSON" --pred-dir "$PRED_DIR_PIPELINE" \
  --label pipeline \
  --omnidocbench-repo "$OMNIDOCBENCH_REPO" --venv-python "$SCORER_VENV/bin/python"
mineru-rocm manifest verify --pred-dir "$PRED_DIR_PIPELINE"

# 2. VLM via vLLM — first serve the model (see "Serving the VLM" below), then:
mineru-rocm predict --backend vlm-vllm \
  --gt-json "$GT_JSON" --images-dir "$IMAGES_DIR" \
  --pred-dir "$PRED_DIR_VLM" --platform linux-rocm
mineru-rocm score --gt-json "$GT_JSON" --pred-dir "$PRED_DIR_VLM" \
  --label vlm-vllm \
  --omnidocbench-repo "$OMNIDOCBENCH_REPO" --venv-python "$SCORER_VENV/bin/python"
```

The authoritative artefacts land under each pred-dir: `run_manifest.json` (conservation laws), `metric_result.json`, `_errors.jsonl`, `predict.log`. The committed copies live under `results/omnidocbench/v1.6/{pipeline,vlm-vllm}/`.

## Serving the VLM (vLLM-on-ROCm)

```bash
HIP_VISIBLE_DEVICES=0 HSA_OVERRIDE_GFX_VERSION=11.0.0 VLLM_USE_V1=1 \
  bash examples/serve_vlm_vllm.sh     # serves --served-model-name mineru-pro, bf16, --enforce-eager
bash examples/wait_vlm.sh             # polls /v1/models
```

The tested server launch script is tracked in `examples/serve_vlm_vllm.sh`; its key flags (served-model-name, dtype, max-model-len, enforce-eager, `VLLM_USE_V1`, …) are summarized in the lock under `rocm_recipe.vlm_server`. Empty outputs: 2/1651 VLM pages produced empty output, observed alongside EOS-first-token behavior — the root cause was not isolated and is not attributed to MinerU.

## Non-determinism

- **pipeline**: deterministic across runs (byte-identical predictions).
- **VLM (vLLM)**: ~0.1 pp run-to-run drift (the P2/P3 re-run scored 95.46 vs a prior 95.56 — Δ −0.10 pp). The source of the drift was not isolated (no profiler / minimal repro / upstream evidence).

## Provenance in the lock

`reproducibility.lock.yaml` records: the `mineru_rocm.release` tag + release commit + annotated-tag object SHA, the **per-run benchmark commits** (`mineru_rocm.benchmark_run_commits`, also recorded in each `run_manifest.json`), the **upstream `mineru`/`mineru_vl_utils` git commits** (resolved via `git ls-remote` against the release tags), byte-exact weight + GT SHAs, the scorer commit, both venvs' full environment, the official anchors, and the metric formula. Deferred fields (`canary_*`, `table_sha256`) are annotated `→ docs/known-gaps.md`.
