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

## Checklist before requesting a `verified` badge

1. `adapter_config.BACKEND` is the real backend (not `smoke`). ✓ (`pipeline`)
2. `model_card.json.hardware` reflects the actual GPU/VRAM/driver.
3. `results/omnidocbench/v16/<platform>/{run_summary,provenance,metric_result}.json` are committed.
4. `make publish` (conformance) passes.
5. Re-running the recorded adapter command reproduces the published overall score (within stated tolerance). ✓ (86.48 vs target 86.47, Δ +0.01).
