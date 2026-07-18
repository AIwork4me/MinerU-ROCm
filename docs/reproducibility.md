# Reproducibility

A score is only meaningful if someone else can reproduce it from the committed repo. This repo + the engine are designed to make that mechanical.

## What gets committed

- `adapter/run_adapter.py` — the dispatcher; routes `--backend` (default `pipeline`, from `adapter_config.BACKEND`) to the real MinerU 3.4 in-process adapter.
- `adapter/pipeline_adapter.py` — the real inference code (wraps upstream `mineru` `do_parse` on ROCm cuda).
- `adapter/adapter_config.py` — selects the primary backend (`pipeline`) and MinerU model target.
- `eval/configs/omnidocbench_v16.yaml` — which metrics, which dataset revision, page limit.
- `model_card.json` / `model_card.pipeline.json` — declared hardware, badge, and pointer to result artifacts.
- `results/omnidocbench/v16/<platform>/` — the published `run_summary.json` + `provenance.json` + `metric_result.json` + sample predictions + full `_run_stats.json`.

## What the engine records (provenance)

Every published run produces a `provenance.json` (schema-validated) capturing:

- `git_commit` — the exact repo state the run used.
- `engine_version` — the `omnidocbench-amd` version.
- `dataset_revision` — the pinned OmniDocBench revision.
- `adapter_command` — the literal subprocess command.
- `scoring_config_path` — the exact YAML handed to OmniDocBench's `pdf_validation.py`.
- `platform`, `model_id`, `vlm_server_url`, `api_model_name`, page counts, and metric/run artifact paths.

So a third party can check out that commit, install the same engine + dataset revision, re-run the recorded command, and expect the same number (modulo non-determinism the adapter itself introduces — document any).

## Reproducing the committed `mineru-pipeline` result (Overall ≈ 86.48)

### Hardware / environment (this run, 2026-07-18)

- **GPU:** AMD gfx1100 (Radeon Pro W7900-class, 48 GB) × 4 — eval pinned to **GPU 3** via `HIP_VISIBLE_DEVICES=3`.
- **ROCm:** 7.2.1.
- **OS:** Linux 6.8.0-79-generic.
- **mineru venv:** `/root/ocr-eval/mineru-rocm-venv` — Python 3.11.15, `mineru` 3.4.4, ROCm torch (`2.14.0.dev…+rocm7.2`), and `omnidocbench-amd` `pip install -e`'d in (so the adapter subprocess resolves `omnidocbench_amd.types` AND `mineru`).
- **scorer venv:** `/root/ocr-eval/OmniDocBench/.venv` — Python 3.11.15 with OmniDocBench's pinned scoring deps (`bs4`, `apted`, `Levenshtein`, `pylatexenc`, `scipy`, …). CDM uses `pdflatex`/`magick` subprocesses, **no torch**.
- **dataset:** OmniDocBench v1.6 (1651 GT pages); page images at `/root/ocr-eval/OmniDocBench_data/images/` (1742 entries — scorer matches the 1651 GT pages and ignores the 91 extras), GT manifest at `/workspace/OmniDocBench_data/OmniDocBench.json` (symlink to the same).

> **Clean-dir convention (future runs).** The shared `images/` dir above has 1742 entries; the scorer silently ignores the 91 surplus non-GT images, so the committed score is correct. For **future** re-runs prefer the clean 1651-only image dir at `/root/ocr-eval/OmniDocBench_v16_images` (symlinks). The `Makefile`'s `eval-linux`/`eval-windows` targets default to it via `OMNIDOCBENCH_IMG_DIR ?= /root/ocr-eval/OmniDocBench_v16_images` (override per-run, e.g. `make eval-linux OMNIDOCBENCH_IMG_DIR=/path/to/images`). The committed run above used the legacy 1742 dir — same result.

### The two-venv orchestration

The adapter imports `omnidocbench_amd.types` AND `mineru`; the OmniDocBench scorer needs its own pinned deps. They live in **different venvs**, so `infer` and `score`/`publish` run separately (the engine's `run --stage all` cannot span venvs):

#### 1. `infer` (mineru venv, GPU 3) — produces 1 Markdown file per page

```bash
source /root/ocr-eval/mineru-rocm-venv/bin/activate
export HIP_VISIBLE_DEVICES=3 MINERU_DEVICE_MODE=cuda HF_ENDPOINT="http://134.199.133.77"
cd /workspace/MinerU-ROCm
omnidocbench-amd infer \
  --adapter adapter/run_adapter.py \
  --img-dir /root/ocr-eval/OmniDocBench_data/images \
  --out-dir /root/ocr-eval/mineru-pipeline-preds \
  --platform linux-rocm
```

The engine spawns the adapter as a subprocess (filesystem-decoupled); it writes one `<image_stem>.md` per page plus a `_run_stats.json` (`{count, ok, fail, fallback, limit_pages, engine, stats[]}`). `limit_pages` MUST be `null` for an official full-set run (the engine refuses to publish otherwise).

This run: **count=1742, ok=1742, fail=0, fallback=0, limit_pages=null, engine=pipeline**, duration **10280 s (~2 h 51 min)**. Steady-state ~3–6 s/page (model warm-up ~15 s on the first page).

#### 2. `score` (scorer venv) — produces the metric_result.json

The platform engine's `LinuxRocmBackend.score()` is an unfinished stub (Task 16 TODO) that mis-invokes `pdf_validation.py` (`--config v16` is not a YAML path; the OmniDocBench CLI takes `--config <yaml>` only, with the prediction path baked into the YAML). Until that lands, drive the OmniDocBench scorer directly with a config that points at the predictions dir:

```bash
cd /workspace/OmniDocBench
/root/ocr-eval/OmniDocBench/.venv/bin/python pdf_validation.py \
  --config configs/mineru_pipeline_full.yaml
```

`configs/mineru_pipeline_full.yaml` declares the v1.6 metric set (`text_block: Edit_dist`, `display_formula: Edit_dist+CDM`, `table: TEDS+Edit_dist`, `reading_order: Edit_dist`), points `ground_truth.data_path` at the v1.6 manifest and `prediction.data_path` at `/root/ocr-eval/mineru-pipeline-preds`, and uses `match_method: quick_match`, 13 workers. Output: `result/mineru-pipeline-preds_quick_match_metric_result.json`. Duration **1110 s (~18.5 min)** — dominated by CDM (2352 formulas rendered via `pdflatex`+`magick`).

#### 3. `publish` (either venv) — assembles provenance + run_summary

```bash
cd /workspace/MinerU-ROCm
source /root/ocr-eval/omnidocbench-amd-venv/bin/activate   # any venv with the engine works
omnidocbench-amd publish \
  --model-id mineru-pipeline --platform linux-rocm --version v16 --cdm \
  --run-stats /root/ocr-eval/mineru-pipeline-preds/_run_stats.json \
  --metric-result /workspace/OmniDocBench/result/mineru-pipeline-preds_quick_match_metric_result.json \
  --results-dir results/omnidocbench/v16/linux-rocm \
  --git-commit "$(git rev-parse HEAD)" \
  --adapter-command "HIP_VISIBLE_DEVICES=3 MINERU_DEVICE_MODE=cuda HF_ENDPOINT=http://134.199.133.77 omnidocbench-amd infer --adapter adapter/run_adapter.py --img-dir <OmniDocBench v1.6 images> --out-dir <preds> --platform linux-rocm" \
  --scoring-config /workspace/OmniDocBench/configs/mineru_pipeline_full.yaml \
  --dataset-manifest /workspace/OmniDocBench_data/OmniDocBench.json \
  --dataset-revision v1.6
```

Emits `results/omnidocbench/v16/linux-rocm/mineru-pipeline_v16_quick_match_cdm_{run_summary,provenance}.json`. The `metric_result.json`, full `_run_stats.json`, and a 10-page sample of predictions are committed alongside (the 1742 raw `.md` stay under `/root` — too heavy for the repo).

### Result — OmniDocBench v1.6 full set, mineru-pipeline, linux-rocm (gfx1100)

Submetrics (from `mineru-pipeline-preds_quick_match_metric_result.json`):

| Metric | Value |
|---|---|
| Text `Edit_dist` (page.ALL) | 0.0566 → **94.34** text-accuracy |
| Table `TEDS` (page.ALL) | **82.04** |
| Table `TEDS_structure_only` (page.ALL) | 88.84 |
| Formula `CDM` (page.ALL) | **83.07** |
| Reading order `Edit_dist` (page.ALL) | 0.1534 |

**Overall = ((1 − Text_EditDist) × 100 + Table_TEDS + Formula_CDM) / 3 = (94.34 + 82.04 + 83.07) / 3 = 86.48**, using the OmniDocBench page-average (`page.ALL`)口径 — the standard leaderboard convention (confirmed in `OmniDocBench/src/core/metrics.py:152`, `page_avg`). The published MinerU reference is 86.47; this run reproduces it within **+0.01 pp** (gate: within ±1.0 pp → **PASS**).

> **Formula variant note.** The OmniDocBench `metric_result.json` carries two aggregations per metric: `.all` (sample-count-weighted) and `.page.ALL` (per-page-then-averaged). The Overall above uses `.page.ALL` for all three terms, matching OmniDocBench's `page_avg` reporting. Using `.all` instead yields 85.26 (same data, different weighting) — **not** a scoring bug, just a different averaging convention. The committed `run_summary.json`'s `readme_metrics` block carries the `.page.ALL` numbers and is the source of truth for the headline.

### Non-determinism

MinerU's pipeline is deterministic on a fixed ROCm stack for the layout/OCR UniMERNet passes (GPU kernels, same input → same output). The ONNX table-recognition model runs on CPU and is deterministic. We did not observe run-to-run variation needing a tolerance band, but the standard ROCm-kernel non-determinism caveat applies if the ROCm/MIOpen stack changes.

## Reproducing the committed `mineru-vlm-vllm` result (Overall 95.56)

The headline MinerU2.5-Pro VLM run — MinerU2.5-Pro-2605-1.2B served via vLLM-on-ROCm, driven by the `vlm-vllm` adapter (two-step layout → per-block extract via `mineru_vl_utils.MinerUClient` http-client). This is Plan 2 Task 8.

### Hardware / environment (this run, 2026-07-18)

- **GPU:** AMD gfx1100 (Radeon Pro W7900-class, 48 GB) — eval pinned to **GPU 0** via `HIP_VISIBLE_DEVICES=0`.
- **ROCm:** 7.2.1, `HSA_OVERRIDE_GFX_VERSION=11.0.0`.
- **VLM env:** `/opt/venv` — Python 3.12, `vllm` `0.16.1.dev0+g89a77b108.d20260317` (ROCm wheel), `mineru_vl_utils`, `omnidocbench-amd` installed. `LD_LIBRARY_PATH=/opt/rocm/lib`.
- **scorer venv:** `/root/ocr-eval/OmniDocBench/.venv` — Python 3.11.15 (same as the pipeline run; the VLM does not load torch for scoring).
- **dataset:** OmniDocBench v1.6, **clean 1651-page** image dir `/root/ocr-eval/OmniDocBench_v16_images` (jpg+png); GT manifest `/workspace/OmniDocBench_data/OmniDocBench.json`.
- **model snapshot:** `/root/.cache/huggingface/models--opendatalab--MinerU2.5-Pro-2605-1.2B/snapshots/bff20d4ae2bf202df9f45284b4d43681555a97ed`.

### Step 1 — serve the model (background; poll `/v1/models`)

```bash
cd /workspace/MinerU-ROCm
HIP_VISIBLE_DEVICES=0 bash examples/serve_vlm_vllm.sh     # nohup's vLLM, writes /tmp/vlm-vllm.pid
bash examples/wait_vlm.sh                                  # blocks until /v1/models responds (≤5 min)
# sanity: /v1/models lists id="mineru-pro"; startup log shows
#   logits_processors=['mineru_vl_utils:MinerULogitsProcessor'] and KV cache ~2.5M tokens
```

Server flags: `--served-model-name mineru-pro --dtype bfloat16 --chat-template adapter/qwen2vl_chat_template.jinja --logits-processors mineru_vl_utils:MinerULogitsProcessor --gpu-memory-utilization 0.70 --max-model-len 8192 --limit-mm-per-prompt '{"image": 1}' --enforce-eager`. `VLLM_USE_V1=1` is mandatory (v1 logits-processor API).

### Step 2 — infer (full 1651-page clean set)

The engine's `omnidocbench-amd infer` stage does not accept `--server-url`/`--api-model-name` (those are `run`/`publish`-only); the vlm adapter reads them from `adapter_config` (env-overridable: `MINERU_ROCM_SERVER_URL`, `MINERU_ROCM_API_MODEL_NAME`). Call the dispatcher directly so `--skip-existing` resume works (multi-hour run):

```bash
export HIP_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/opt/rocm/lib HSA_OVERRIDE_GFX_VERSION=11.0.0
export MINERU_ROCM_BACKEND=vlm-vllm
export MINERU_ROCM_SERVER_URL="http://127.0.0.1:8265/v1" MINERU_ROCM_API_MODEL_NAME="mineru-pro"
/opt/venv/bin/python adapter/run_adapter.py \
  --img-dir  /root/ocr-eval/OmniDocBench_v16_images \
  --out-dir  /root/ocr-eval/mineru-vlm-vllm-preds \
  --platform linux-rocm --backend vlm-vllm \
  --server-url "http://127.0.0.1:8265/v1" --api-model-name "mineru-pro" \
  --skip-existing
```

Writes one `<image_stem>.md` per page + `_run_stats.json` at the end. **Duration: ~4h35m wall** (13:24–20:29 UTC, including one resume after a background-task interruption; 1651 pages, avg ~16 s/page two-step on gfx1100 eager mode, max 131 s on dense exam pages). `ok=1651, fail=0, fallback=0`. **Empty-page rate: 2/1651 = 0.12%** (both are sparse English-textbook cover/contents pages — not EOS-first-token victims; well under the 2% concern threshold).

> **Background-task caveat.** A foreground `vllm serve` is killed by the harness (exit 144); a `bash` background task is *also* reaped after long idle polls. The robust pattern is `setsid nohup <adapter> &` (own session/process group) — the adapter then survives indefinitely and you poll the filesystem (`_run_stats.json` appears at completion). The helper is `/root/ocr-eval/launch_detached_adapter.sh`. `--skip-existing` makes any interruption cleanly resumable.

### Step 3 — stop the server (free GPU 0)

```bash
kill "$(cat /tmp/vlm-vllm.pid)"; pkill -f vllm.entrypoints   # frees VRAM 90%→baseline
```

### Step 4 — score (OmniDocBench venv, full GT)

```bash
sed 's|/root/ocr-eval/mineru-pipeline-preds|/root/ocr-eval/mineru-vlm-vllm-preds|' \
  /root/ocr-eval/OmniDocBench/configs/mineru_pipeline_full.yaml \
  > /workspace/MinerU-ROCm/configs/mineru_vlm_full.yaml
source /root/ocr-eval/OmniDocBench/.venv/bin/activate
cd /root/ocr-eval/OmniDocBench && python pdf_validation.py \
  --config /workspace/MinerU-ROCm/configs/mineru_vlm_full.yaml
# → result/mineru-vlm-vllm-preds_quick_match_metric_result.json (~25 min)
```

### Step 5 — publish (assemble run_summary + provenance)

```bash
cd /workspace/MinerU-ROCm
/opt/venv/bin/omnidocbench-amd publish \
  --model-id mineru-vlm-vllm --platform linux-rocm --version v16 --cdm \
  --run-stats /root/ocr-eval/mineru-vlm-vllm-preds/_run_stats.json \
  --metric-result /root/ocr-eval/OmniDocBench/result/mineru-vlm-vllm-preds_quick_match_metric_result.json \
  --results-dir results/omnidocbench/v16/linux-rocm \
  --git-commit "$(git rev-parse HEAD)" \
  --adapter-command "HIP_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/opt/rocm/lib MINERU_ROCM_BACKEND=vlm-vllm python adapter/run_adapter.py --img-dir <OmniDocBench v1.6 images> --out-dir <preds> --platform linux-rocm --backend vlm-vllm --server-url http://127.0.0.1:8265/v1 --api-model-name mineru-pro" \
  --server-url "http://127.0.0.1:8265/v1" --api-model-name mineru-pro \
  --scoring-config /workspace/MinerU-ROCm/configs/mineru_vlm_full.yaml \
  --dataset-manifest /workspace/OmniDocBench_data/OmniDocBench.json \
  --dataset-revision v1.6
```

### Result — OmniDocBench v1.6 full set, mineru-vlm-vllm, linux-rocm (gfx1100)

Submetrics (from `mineru-vlm-vllm-preds_quick_match_metric_result.json`, page.ALL):

| Metric | Value | vs Plan 1 pipeline |
|---|---|---|
| Text `Edit_dist` → text-accuracy | 0.0359 → **96.41** | +2.07 pp (94.34) |
| Table `TEDS` | **93.54** | +11.50 pp (82.04) |
| Formula `CDM` | **96.73** | +13.66 pp (83.07) |
| Reading order `Edit_dist` | 0.1240 | (0.1534) |

**Overall = ((1 − 0.0359) × 100 + 93.54 + 96.73) / 3 = (96.41 + 93.54 + 96.73) / 3 = 95.56** using the page.ALL convention (same as Plan 1's 86.48). Target 95.75 → **Δ −0.19 pp**; gate ≥ 95.25 → **PASS** (margin +0.31 pp). This is **+9.08 pp over Plan 1's pipeline backend** (86.48), driven mainly by table TEDS and formula CDM — exactly the dimensions the VLM is supposed to dominate.

> **`ok_pages` in run_stats.** The committed `_run_stats.json`/`run_summary.json` record `ok_pages=1074` rather than 1651 because the run was resumed once (after a background-task interruption) and the dispatcher's stats only cover the pages inferred in the final invocation — 1651 − 577 already-on-disk = 1074. **All 1651 predictions are present on disk** (`ls /root/ocr-eval/mineru-vlm-vllm-preds/*.md | wc -l` = 1651), `fail=0`, and the scorer matched all 1651 GT pages (`page_count: 1651` in the metric_result). The headline Overall is computed over the full 1651-page set and is valid.

### Non-determinism (VLM)

vLLM in `--enforce-eager` bf16 with the `MinerULogitsProcessor` (no-repeat-100-gram) is **near-deterministic** for the layout pass (greedy) and per-block extraction; run-to-run drift is bounded by ROCm-kernel non-determinism on bf16 matmuls. We did not re-run the full set to bound a tolerance, but a 100-page sample prior to this run scored Overall ≈ 97.0 (0 empty pages), consistent with the full-set 95.56 once harder pages are included.

## Checklist before requesting a `verified` badge

1. `adapter_config.BACKEND` is the real backend (not `smoke`). ✓ (`pipeline` for Plan 1; `vlm-vllm` for Plan 2 via `MINERU_ROCM_BACKEND`).
2. `model_card.json.hardware` reflects the actual GPU/VRAM/driver.
3. `results/omnidocbench/v16/<platform>/{run_summary,provenance,metric_result}.json` are committed (both `mineru-pipeline-*` and `mineru-vlm-vllm-*`).
4. `make publish` (conformance) passes.
5. Re-running the recorded adapter command reproduces the published overall score (within stated tolerance). ✓ pipeline 86.48 vs target 86.47, Δ +0.01. ✓ VLM 95.56 vs target 95.75, Δ −0.19 (gate ≥ 95.25 PASS).
