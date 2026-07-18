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

## Reproducibility

Results live under `results/omnidocbench/v16/<platform>/`. Each run produces a schema-validated `run_summary.json` + `provenance.json` (engine version, git commit, dataset revision, adapter command) so a number is independently reproducible from the committed adapter + config on the declared hardware. See [`docs/reproducibility.md`](docs/reproducibility.md).

## Known Gaps

- The `smoke` backend emits placeholder text, not real OCR. Replace `_infer` in `adapter/run_adapter.py` with your model's inference.
- No `verified` results on either platform yet (`badge: community-wanted`).
- Provisioning scripts (`adapter/setup/`) are stubs.
- See [`docs/known-gaps.md`](docs/known-gaps.md) for the full list.
