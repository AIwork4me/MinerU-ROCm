# Reproducibility

A score is only meaningful if someone else can reproduce it from the committed repo. This repo + the engine are designed to make that mechanical.

## What gets committed

- `adapter/run_adapter.py` — the exact inference code (the `smoke` branch is a placeholder; replace `_infer`).
- `eval/configs/omnidocbench_v16.yaml` — which metrics, which dataset revision, page limit.
- `model_card.json` — declared hardware, badge, and pointer to result artifacts.
- `results/omnidocbench/v16/<platform>/` — the published `run_summary.json` + `provenance.json`.

## What the engine records (provenance)

Every published run produces a `provenance.json` (schema-validated) capturing:

- `git_commit` — the exact repo state the run used.
- `engine_version` — the `omnidocbench-amd` version.
- `dataset_revision` — the pinned OmniDocBench revision.
- `adapter_command` — the literal subprocess command.
- `platform`, `model_id`, `vlm_server_url`, `api_model_name`, page counts, and metric/run artifact paths.

So a third party can check out that commit, install the same engine + dataset revision, re-run the recorded command, and expect the same number (modulo non-determinism the adapter itself introduces — document any).

## Checklist before requesting a `verified` badge

1. `_infer` is wired to your real model (not `smoke`).
2. `model_card.json.hardware` reflects the actual GPU/VRAM/driver.
3. `results/omnidocbench/v16/<platform>/{run_summary,provenance}.json` are committed.
4. `make publish` (conformance) passes.
5. Re-running the recorded adapter command reproduces the published overall score (within stated tolerance).
