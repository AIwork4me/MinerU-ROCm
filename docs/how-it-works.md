# How it works

`MinerU-ROCm` is **benchmark infrastructure for evaluating opendatalab MinerU on AMD ROCm** — not a model port. It runs the MinerU 3.4 pipeline and the MinerU2.5-Pro VLM, and scores them on OmniDocBench v1.6. The standalone CLI (`mineru-rocm`) is the primary interface; the omnidocbench-rocm engine is an *optional* consumer via the `[platform]` extra.

## The contract

Every per-model repo implements one function:

```python
def run_adapter(img_dir: Path, out_dir: Path, *, platform: str, config: dict) -> dict:
```

It must, for each image in `img_dir`:

1. Write `out_dir/<image_stem>.md` — the model's markdown prediction for that page.
2. Record a `PageStatus` (`ok` / `failed: <reason>` / `fallback: <reason>`).
3. Never raise — a per-page failure is caught and recorded (a missing page scores zero).

Finally it writes a schema-valid `_run_stats.json` (via `RunSummary.write`) and returns `RunSummary.to_run_stats()`. The engine consumes those artifacts downstream.

## Backends

The `config["backend"]` key selects the inference path inside `run_adapter`:

| backend | what it does | GPU? |
|---|---|---|
| `pipeline` (default) | wraps upstream `mineru[all]` in-process on `cuda` (PyTorch-ROCm exposes the HIP device as `cuda`) → markdown | yes |
| `smoke` | writes a placeholder `.md` per image | no |

See [`backends.md`](backends.md) for the recommended backend per model type × platform.

## Stages (engine-side)

> **Primary interface is the `mineru-rocm` CLI** (`predict` → `validate` → `score` → `manifest verify`). The `omnidocbench-rocm` stages below apply only when using the optional `[platform]` engine extra.

The `omnidocbench-rocm` CLI (`make eval-linux`) runs:

1. **download** — fetch the pinned OmniDocBench v1.6 dataset revision.
2. **infer** — invoke `adapter/run_adapter.py` as a subprocess over the dataset images.
3. **score** — Edit_dist / TEDS / CDM against the gold answers.
4. **publish** — assemble + schema-validate `run_summary.json` and `provenance.json` into `results/`.

`make publish` (or `scripts/check_conformance.py .`) verifies this repo still satisfies the contract.

## Two model cards, one repo (registry story)

This repo ships **two** model cards for one upstream model family (MinerU 2.5 / MinerU 3.4):

| Card | File | What it is | Status |
|---|---|---|---|
| **primary** (registry row) | `model_card.json` + `hub/registry.yaml` `mineru2.5` | the **VLM** (MinerU 2.5 served via vLLM-on-ROCm) | reproduced, linux-rocm `community` (Overall **95.56** platform CDM, 1651/1651, gate PASS) |
| **secondary** (Plan 1 result) | `model_card.pipeline.json` + platform-specific cards + the README comparison table | the **MinerU 3.4 pipeline** (layout → OCR → table → formula, in-process) | reproduced: linux-rocm **86.48**, windows-hip **86.59**, both `community` |

The platform's `hub/registry.yaml` carries **one row per `model_id`**, and that row is the VLM (`mineru2.5`). The pipeline (`mineru-pipeline`) is a secondary card inside this same repo, surfaced via `model_card.pipeline.json` + the README table — **so no new registry row is needed for the pipeline**. The VLM result now fills the primary `model_card.json` (Overall **95.56** platform CDM, badge linux-rocm `community`); the pipeline card stays as the secondary entry. (The prior standalone-path VLM Overall was **95.46** — Formula CDM 96.46 — on the same 1651 predictions; superseded by the platform CDM 95.56, see `docs/reproducibility.md`.)

> **Registry update note:** the `hub/registry.yaml` lives in the **separate** platform repo [`OmniDocBench-ROCm`](https://github.com/AIwork4me/OmniDocBench-ROCm) and carries the primary `mineru2.5` VLM row. The VLM's `windows-hip` entry remains `community-wanted`; the supplementary pipeline Windows result is recorded here in `model_card.pipeline.windows-hip.json` and its conformant bundle. (`verified` still requires maintainer reproduction.)
