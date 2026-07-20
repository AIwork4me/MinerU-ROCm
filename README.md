# MinerU-ROCm

> Evaluation-backed AMD ROCm port of [MinerU](https://github.com/opendatalab/MinerU)
> — runs the **MinerU 3.4 pipeline** and the **MinerU2.5-Pro** VLM on AMD
> **gfx1100 (RDNA3)** and reports **OmniDocBench v1.6** results across multiple
> inference backends. **Not** a precision-aligned port: no same-page-set CUDA
> control exists, and the upstream headline may use a different engine. See
> [Benchmark methodology](docs/benchmark-methodology.md).

[![OmniDocBench v1.6](https://img.shields.io/badge/OmniDocBench-v1.6-blue)](https://github.com/opendatalab/OmniDocBench)
[![VLM full](https://img.shields.io/badge/MinerU2.5--Pro%20VLM%20(full)-95.46-green)](#results--mineru25-pro-vlm)
[![pipeline full](https://img.shields.io/badge/MinerU%203.4%20pipeline%20(full)-86.48-yellowgreen)](#results--mineru-34-pipeline)
[![status: evaluation-backed](https://img.shields.io/badge/status-evaluation--backed-blue)](reproducibility.lock.yaml)
[![license: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0%20(+MinerU%20terms)-blue)](NOTICE)

## At a glance

- **What it is.** Tooling to run opendatalab MinerU (3.4 pipeline + 2.5-Pro VLM) on AMD ROCm and score it on OmniDocBench v1.6.
- **Where verified.** AMD **gfx1100 (RDNA3), Radeon PRO W7900 (48 GB), ROCm 7.2**, bf16. Host has 4× W7900; **each benchmark used 1 GPU (no tensor parallel).**
- **Most reliable results.** **MinerU2.5-Pro VLM (vLLM-on-ROCm) full 1651 = 95.46 Overall**; **MinerU 3.4 pipeline full 1651 = 86.48 Overall**.
- **Most important limitation.** **Not precision-aligned.** No same-engine CUDA control exists; the upstream headline may be measured with a different engine. The official anchor (vlm-engine 95.30) is aligned to the upstream README "Local Deployment" table and is **community-verified, not official support** — see `reproducibility.lock.yaml` (`benchmark.official_reference`).
- **Upstream.** This is a port OF [opendatalab/MinerU](https://github.com/opendatalab/MinerU); the [omnidocbench-amd](https://github.com/AIwork4me/OmniDocBench-AMD) engine is one *optional* consumer (install the `[platform]` extra), not the definition of this repo.

## Install

The core package is GPU-free and has no platform dependency.

```bash
pip install -e ".[dev]"          # core + dev/CI tooling (pytest, ruff, reuse)
# optional: omnidocbench-amd engine integration (the adapter/run_adapter.py path)
pip install -e ".[platform]"
```

For platform provisioning (weights, ROCm runtime), run `make setup-linux` (or
`make setup-windows`). GPU backends additionally need a ROCm torch + (VLM)
vLLM-on-ROCm, installed separately from a verified ROCm wheel source — see
`docs/reproducibility.md`.

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

Run the full OmniDocBench v1.6 eval (infer + score for both backends) via the standalone CLI:

```bash
# predict → score (set GT_JSON / IMAGES_DIR / PRED_DIR / SCORER_VENV to your paths)
mineru-rocm predict --backend pipeline \
  --gt-json OmniDocBench.json --images-dir images/ --pred-dir out/ --platform linux-rocm
mineru-rocm score --gt-json OmniDocBench.json --pred-dir out/ --label pipeline \
  --venv-python <scorer-venv>/bin/python
# repeat with --backend vlm-vllm for the VLM
```

Or via make (overrides via env): `make eval-linux GT_JSON=… IMAGES_DIR=… PRED_DIR=… SCORER_VENV=…`.
See [`docs/reproducibility.md`](docs/reproducibility.md) for the full recipe and [`docs/benchmark-methodology.md`](docs/benchmark-methodology.md) for the reproduce commands.

### Results — MinerU2.5-Pro VLM (primary model card, `mineru2.5`)

| Model / Backend | Overall ↑ | Text Edit ↓ | Formula CDM ↑ | Table TEDS ↑ |
|---|---:|---:|---:|---:|
| _official_ MinerU2.5-Pro _(upstream README vlm-engine row; community-verified, not official support)_ | 95.30 | — | — | — |
| **ours MinerU2.5-Pro (vlm-vllm, ROCm)** | **95.46** | 0.0360 | 96.46 | 93.54 |
| ours MinerU2.5-Pro (vlm-transformers, ROCm) | _sample-only (clean; ~44 h full)_ | | | |

The `vlm-vllm` row is **reproduced** on linux-rocm (self-attested, `badge: community`, conformance-passing): 1651/1651 requests attempted, 1649 non-empty predictions (2 empty), no process crashes; ~7 h on a single GPU (gfx1100); read-order EditDist 0.1236. Overall 95.46 is **consistent with the published upstream reference range** (vlm-engine 95.30; Δ +0.16 pp — **not** a controlled CUDA-vs-ROCm comparison) and within ~0.1 pp of our own prior run (95.56). The upstream anchor is from the upstream README "Local Deployment" table, recorded as **community-verified, not official support** — see `reproducibility.lock.yaml` (`benchmark.official_reference`). The `vlm-transformers` backend is a clean but slow fallback (~100–150 s/page; full-set ≈44 h not run), so it carries no full Overall. `windows-hip` is `community-wanted`.

### Results — MinerU 3.4 pipeline (secondary model card, `mineru-pipeline`)

| Model / Backend | Overall ↑ | Text Edit ↓ | Formula CDM ↑ | Table TEDS ↑ |
|---|---:|---:|---:|---:|
| _official_ MinerU 3.4 pipeline | 86.47 | — | — | — |
| **ours MinerU 3.4 pipeline (ROCm gfx1100, linux-rocm)** | **86.48** | 0.0566 | 83.07 | 82.04 |
| windows-hip | _pending (colleague)_ | | | |

Both `linux-rocm` rows are **reproduced** (self-attested, `badge: community`, conformance-passing) — see [`docs/reproducibility.md`](docs/reproducibility.md). The primary `mineru2.5` VLM row above fills the `hub/registry.yaml` `mineru2.5` entry; the pipeline lives here in `model_card.pipeline.json` + this table (no separate registry row).

## License — read before downloading weights

This repo is **Apache-2.0** (original packaging/tooling). The MinerU pipeline is
under the **MinerU Open Source License** (Apache-2.0 + additional terms:
commercial use above MAU 100M or USD 20M/mo revenue needs a separate license;
online services must attribute MinerU). `mineru-vl-utils` and the MinerU2.5-Pro
weights are Apache-2.0. The **PDF-Extract-Kit-1.0** pipeline weights declare **no
license** on their HF card — treat as license-ambiguous, do not redistribute. Full
breakdown in [NOTICE](NOTICE) and [LICENSES/](LICENSES). Not affiliated with the
MinerU Team / OpenDataLab.

## Reproducibility

[`reproducibility.lock.yaml`](reproducibility.lock.yaml) is the single source of
truth — pinned commits, byte-exact weight/GT SHA256 cross-checked against the
upstream HF repos, environment versions, and the metric formula. Verified values
were populated from the full 1651-page reruns completed on 2026-07-19. See
[docs/reproducibility.md](docs/reproducibility.md).

## Issues filed

- **[ROCm/AMDMIGraphX#5078](https://github.com/ROCm/AMDMIGraphX/issues/5078)** — Loop-subgraph parser bug affecting ONNX table recognition on ROCm.
- Upstream `opendatalab/MinerU` AMD.md contribution + PDF-Extract-Kit-1.0 license clarification are planned (P4).

## Known Gaps

- The `smoke` backend emits placeholder text, not real OCR; `pipeline` (the default, real MinerU 3.4 in-process adapter) is the production path. CI/conformance can force `smoke` via `BACKEND=smoke` or `--backend smoke`.
- `mineru-pipeline` is `community` on linux-rocm (Overall **86.48** on OmniDocBench v1.6, gfx1100 — see [`docs/reproducibility.md`](docs/reproducibility.md)); windows-hip still `community-wanted`.
- Provisioning scripts (`adapter/setup/`) are stubs.
- See [`docs/known-gaps.md`](docs/known-gaps.md) for the full list.
