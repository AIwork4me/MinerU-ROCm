# MinerU-ROCm

A per-model adapter repo for the [omnidocbench-amd](https://github.com/AIwork4me/OmniDocBench-AMD) document-parsing evaluation platform. Rendered from the official cookiecutter template; ships with a no-GPU `smoke` backend so it runs out of the box.

- Model: `mineru2.5` (v0.1.0)
- Platforms: linux-rocm, windows-hip
- Badge: community-wanted (both platforms) — replace with `verified` once you commit reproducible results.

## Install

```bash
pip install -e ".[dev]"
pip install omnidocbench-amd        # the engine (provides the `omnidocbench-amd` CLI + types)
```

For platform provisioning (weights, ROCm/DirectML runtime), run:

```bash
make setup-linux     # or: make setup-windows
```

## Demo

The `smoke` backend needs no GPU — it writes a placeholder `.md` per image so you can verify the contract end-to-end:

```bash
bash examples/run_demo.sh        # Linux/macOS
# .\examples\run_demo.ps1        # Windows
```

Or directly:

```bash
python adapter/run_adapter.py --img-dir examples --out-dir /tmp/out --platform linux-rocm --backend smoke
```

## Evaluation

Run the full OmniDocBench v1.6 pipeline (download → infer → score → publish) once `_infer` is wired up:

```bash
make eval-linux      # linux-rocm
# make eval-windows  # windows-hip (run on Windows)
```

Eval config: [`eval/configs/omnidocbench_v16.yaml`](eval/configs/omnidocbench_v16.yaml).

### Results — MinerU 3.4 pipeline (secondary model card, `mineru-pipeline`)

| Backend | Overall ↑ | Text Edit ↓ | Formula CDM ↑ | Table TEDS ↑ |
|---|---:|---:|---:|---:|
| _official_ MinerU pipeline | 86.47 | — | — | — |
| **ours (ROCm gfx1100, linux-rocm)** | **86.48** | 0.0566 | 83.07 | 82.04 |
| windows-hip | _pending (colleague)_ | | | |

The `linux-rocm` row is **reproduced** (self-attested, `badge: community`, conformance-passing) — see [`docs/reproducibility.md`](docs/reproducibility.md). The primary `mineru2.5` VLM model card is a separate row (Plan 2, not yet run). The VLM lives in `hub/registry.yaml`; the pipeline lives here in `model_card.pipeline.json` + this table.

## Reproducibility

Results live under `results/omnidocbench/v16/<platform>/`. Each run produces a schema-validated `run_summary.json` + `provenance.json` (engine version, git commit, dataset revision, adapter command) so a number is independently reproducible from the committed adapter + config on the declared hardware. See [`docs/reproducibility.md`](docs/reproducibility.md).

## Known Gaps

- The `smoke` backend emits placeholder text, not real OCR; `pipeline` (the default, real MinerU 3.4 in-process adapter) is the production path. CI/conformance can force `smoke` via `BACKEND=smoke` or `--backend smoke`.
- `mineru-pipeline` is `community` on linux-rocm (Overall **86.48** on OmniDocBench v1.6, gfx1100 — see [`docs/reproducibility.md`](docs/reproducibility.md)); windows-hip still `community-wanted`.
- Provisioning scripts (`adapter/setup/`) are stubs.
- See [`docs/known-gaps.md`](docs/known-gaps.md) for the full list.
