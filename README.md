# MinerU-ROCm

A per-model adapter repo for the [omnidocbench-amd](https://github.com/AIwork4me/OmniDocBench-AMD) document-parsing evaluation platform. Rendered from the official cookiecutter template; ships with a no-GPU `smoke` backend so it runs out of the box.

- Model: `mineru2.5` (VLM checkpoint 2605)
- Platforms: linux-rocm, windows-hip
- Badge: linux-rocm `community` (Overall **95.56** on OmniDocBench v1.6, reproduced); windows-hip `community-wanted`. `verified` needs maintainer Docker reproduction.

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

### Results — MinerU2.5-Pro VLM (primary model card, `mineru2.5`)

| Model / Backend | Overall ↑ | Text Edit ↓ | Formula CDM ↑ | Table TEDS ↑ |
|---|---:|---:|---:|---:|
| _official_ MinerU2.5-Pro | 95.75 | 0.036 | 97.45 | 93.42 |
| **ours MinerU2.5-Pro (vlm-vllm, ROCm)** | **95.56** | 0.0359 | 96.73 | 93.54 |
| ours MinerU2.5-Pro (vlm-transformers, ROCm) | _sample-only (clean; ~44 h full)_ | | | |

The `vlm-vllm` row is **reproduced** on linux-rocm (self-attested, `badge: community`, conformance-passing): 1651/1651 pages, 0 fail, ~7 h on GPU 0 (gfx1100), empty-rate 0.12%, read-order EditDist 0.1240. Gate PASS at +0.31 pp from the official 95.75 (≤0.5 pp). The `vlm-transformers` backend is a clean but slow fallback (~100–150 s/page; full-set ≈44 h not run), so it carries no full Overall. `windows-hip` is `community-wanted`.

### Results — MinerU 3.4 pipeline (secondary model card, `mineru-pipeline`)

| Model / Backend | Overall ↑ | Text Edit ↓ | Formula CDM ↑ | Table TEDS ↑ |
|---|---:|---:|---:|---:|
| _official_ MinerU 3.4 pipeline | 86.47 | — | — | — |
| **ours MinerU 3.4 pipeline (ROCm gfx1100, linux-rocm)** | **86.48** | 0.0566 | 83.07 | 82.04 |
| windows-hip | _pending (colleague)_ | | | |

Both `linux-rocm` rows are **reproduced** (self-attested, `badge: community`, conformance-passing) — see [`docs/reproducibility.md`](docs/reproducibility.md). The primary `mineru2.5` VLM row above fills the `hub/registry.yaml` `mineru2.5` entry; the pipeline lives here in `model_card.pipeline.json` + this table (no separate registry row).

## Reproducibility

Results live under `results/omnidocbench/v16/<platform>/`. Each run produces a schema-validated `run_summary.json` + `provenance.json` (engine version, git commit, dataset revision, adapter command) so a number is independently reproducible from the committed adapter + config on the declared hardware. See [`docs/reproducibility.md`](docs/reproducibility.md).

## Known Gaps

- The `smoke` backend emits placeholder text, not real OCR; `pipeline` (the default, real MinerU 3.4 in-process adapter) is the production path. CI/conformance can force `smoke` via `BACKEND=smoke` or `--backend smoke`.
- `mineru-pipeline` is `community` on linux-rocm (Overall **86.48** on OmniDocBench v1.6, gfx1100 — see [`docs/reproducibility.md`](docs/reproducibility.md)); windows-hip still `community-wanted`.
- Provisioning scripts (`adapter/setup/`) are stubs.
- See [`docs/known-gaps.md`](docs/known-gaps.md) for the full list.
