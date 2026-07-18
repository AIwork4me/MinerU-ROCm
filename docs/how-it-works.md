# How it works

`MinerU-ROCm` is a per-model adapter repo for the **omnidocbench-amd** engine. The engine drives the OmniDocBench v1.6 pipeline; this repo only supplies the model-specific inference step.

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
| `smoke` (default) | writes a placeholder `.md` per image | no |
| your model | calls `_infer(img, platform, config)` → markdown | yes |

See [`backends.md`](backends.md) for the recommended backend per model type × platform.

## Stages (engine-side)

The `omnidocbench-amd` CLI (`make eval-linux`) runs:

1. **download** — fetch the pinned OmniDocBench v1.6 dataset revision.
2. **infer** — invoke `adapter/run_adapter.py` as a subprocess over the dataset images.
3. **score** — Edit_dist / TEDS / CDM against the gold answers.
4. **publish** — assemble + schema-validate `run_summary.json` and `provenance.json` into `results/`.

`make publish` (or `scripts/check_conformance.py .`) verifies this repo still satisfies the contract.
