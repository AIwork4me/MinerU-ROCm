# MinerU-ROCm

> Evaluation-backed AMD ROCm port of [MinerU](https://github.com/opendatalab/MinerU)
> — runs the **MinerU 3.4 pipeline** and the **MinerU2.5-Pro** VLM on AMD
> **gfx1100 (RDNA3)** and reports **OmniDocBench v1.6** results across multiple
> inference backends. **Not** a precision-aligned port: no same-page-set CUDA
> control exists, and the upstream headline may use a different engine. See
> [Benchmark methodology](docs/benchmark-methodology.md).

[![OmniDocBench v1.6](https://img.shields.io/badge/OmniDocBench-v1.6-blue)](https://github.com/opendatalab/OmniDocBench)
[![VLM full](https://img.shields.io/badge/MinerU2.5--Pro%20VLM%20(full)-95.46-green)](#evaluation)
[![pipeline full](https://img.shields.io/badge/MinerU%203.4%20pipeline%20(full)-86.48-yellowgreen)](#evaluation)
[![status: evaluation-backed](https://img.shields.io/badge/status-evaluation--backed-blue)](reproducibility.lock.yaml)
[![license: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0%20(+MinerU%20terms)-blue)](NOTICE)

## At a glance

- **What it is.** Tooling to run opendatalab MinerU (3.4 pipeline + 2.5-Pro VLM) on AMD ROCm and score it on OmniDocBench v1.6.
- **Where verified.** AMD **gfx1100 (RDNA3), Radeon PRO W7900 (48 GB), ROCm 7.2**, bf16. Host has 4x W7900; **each benchmark used 1 GPU (no tensor parallel).**
- **Most reliable results.** **MinerU2.5-Pro VLM (vLLM-on-ROCm) full 1651 = 95.46 Overall**; **MinerU 3.4 pipeline full 1651 = 86.48 Overall**.
- **Most important limitation.** **Not precision-aligned.** No same-engine CUDA control exists; the upstream headline may be measured with a different engine. The official anchor (vlm-engine 95.30) is aligned to the upstream README "Local Deployment" table and is **community-verified, not official support**.
- **Upstream.** This is a port OF [opendatalab/MinerU](https://github.com/opendatalab/MinerU); the [omnidocbench-rocm](https://github.com/AIwork4me/OmniDocBench-ROCm) engine is one *optional* consumer (install the `[platform]` extra), not the definition of this repo.

## Install

The core package is GPU-free and has no platform dependency.

```bash
pip install -e ".[dev]"          # core + dev/CI tooling (pytest, ruff, reuse)
# optional: omnidocbench-rocm engine integration (the adapter/run_adapter.py path)
pip install -e ".[platform]"
```

For platform provisioning (weights, ROCm runtime), run `make setup-linux` (or
`make setup-windows`). GPU backends additionally need a ROCm torch + (VLM)
vLLM-on-ROCm, installed separately from a verified ROCm wheel source — see
`docs/reproducibility.md`.

**Note:** provisioning scripts (`adapter/setup/`) are stubs — they document the
manual steps required; they do not automate the full environment setup.
See `adapter/setup/00-install-deps.sh` for the Linux recipe.

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

The canonical OmniDocBench-ROCm platform evaluation uses `omnidocbench-rocm`:

### MinerU2.5-Pro VLM (primary model card, `mineru2.5`)

```bash
omnidocbench-rocm run \
  --stage all \
  --platform linux-rocm \
  --version v16 \
  --revision 2b161d0 \
  --adapter adapter/run_adapter.py \
  --model-id mineru2.5 \
  --backend vlm-vllm \
  --server-url http://127.0.0.1:8265/v1 \
  --api-model-name mineru-pro \
  --git-commit "$(git rev-parse HEAD)" \
  --results-dir results/omnidocbench/v16/linux-rocm \
  --skip-existing
```

### MinerU 3.4 Pipeline (supplementary, `mineru-pipeline`)

```bash
omnidocbench-rocm run \
  --stage all \
  --platform linux-rocm \
  --version v16 \
  --revision 2b161d0 \
  --adapter adapter/run_adapter.py \
  --model-id mineru-pipeline \
  --backend pipeline \
  --git-commit "$(git rev-parse HEAD)" \
  --results-dir results/omnidocbench/v16/linux-rocm \
  --skip-existing
```

### Split-stage execution (for long VLM runs)

```bash
omnidocbench-rocm infer --backend vlm-vllm ...
omnidocbench-rocm score ...
omnidocbench-rocm publish --predictions-dir <real-prediction-dir> ...
```

The standalone `mineru-rocm` CLI (`predict` / `score`) remains available for
developer debugging — see `docs/reproducibility.md` for the full recipe.

### Results — MinerU2.5-Pro VLM (primary model card, `mineru2.5`)

| Model / Backend | Overall | Text Edit | Formula CDM | Table TEDS |
|---|---:|---:|---:|---:|
| _official_ MinerU2.5-Pro _(upstream README vlm-engine row; community-verified, not official support)_ | 95.30 | — | — | — |
| **ours MinerU2.5-Pro (vlm-vllm, ROCm)** | **95.56** | 0.0359 | 96.73 | 93.54 |
| ours MinerU2.5-Pro (vlm-transformers, ROCm) | _sample-only_ | | | |

The `vlm-vllm` row is reproduced on linux-rocm (self-attested, `badge: community`):
1651/1651 pages attempted, 1649 non-empty predictions (2 empty), no process
crashes; ~7 h on a single GPU (gfx1100); read-order EditDist 0.1240.
Overall 95.56 is consistent with the published upstream reference range
(vlm-engine 95.30; delta +0.26 pp — **not** a controlled CUDA-vs-ROCm
comparison). The upstream anchor is from the upstream README "Local Deployment"
table, recorded as **community-verified, not official support** — see
`reproducibility.lock.yaml` (`benchmark.official_reference`). `windows-hip` is
`community-wanted` (no results yet).

### Results — MinerU 3.4 pipeline (supplementary model card, `mineru-pipeline`)

| Model / Backend | Overall | Text Edit | Formula CDM | Table TEDS |
|---|---:|---:|---:|---:|
| _official_ MinerU 3.4 pipeline | 86.47 | — | — | — |
| **ours MinerU 3.4 pipeline (ROCm gfx1100, linux-rocm)** | **86.48** | 0.0566 | 83.07 | 82.04 |
| windows-hip | _community-wanted_ | | | |

Pipeline results are at `results/omnidocbench/v1.6/pipeline/`. Known: 1 empty
output page. The primary registry card is `mineru2.5` (VLM); pipeline is a
supplementary card in the same repo — see `model_card.pipeline.json`.

## Reproducibility

[`reproducibility.lock.yaml`](reproducibility.lock.yaml) is the single source of
truth — pinned commits, byte-exact weight/GT SHA256 cross-checked against the
upstream HF repos, environment versions, and the metric formula. Verified values
were populated from the full 1651-page reruns completed on 2026-07-19.

Hardware: AMD gfx1100 (Radeon PRO W7900), 48 GB VRAM, ROCm 7.2, bf16.
The official reference (pipeline 86.47, vlm-engine 95.30) is sourced from the
upstream MinerU README "Local Deployment" table as a community-verified anchor,
not official support. See `docs/reproducibility.md` for the full recipe.

## License — read before downloading weights

This repo is **Apache-2.0** (original packaging/tooling). The MinerU pipeline
is under the **MinerU Open Source License** (Apache-2.0 + additional terms:
commercial use above MAU 100M or USD 20M/mo revenue needs a separate license;
online services must attribute MinerU). `mineru-vl-utils` and the MinerU2.5-Pro
weights are Apache-2.0. The **PDF-Extract-Kit-1.0** pipeline weights declare
**no license** on their HF card — treat as license-ambiguous, do not
redistribute. Full breakdown in [NOTICE](NOTICE) and [LICENSES/](LICENSES).
Not affiliated with the MinerU Team / OpenDataLab.

## Issues filed

- **[ROCm/AMDMIGraphX#5078](https://github.com/ROCm/AMDMIGraphX/issues/5078)** — Loop-subgraph parser bug affecting ONNX table recognition on ROCm.
- Upstream `opendatalab/MinerU` AMD.md contribution + PDF-Extract-Kit-1.0 license clarification are planned (P4).

## Known Gaps

- The `smoke` backend emits placeholder text, not real OCR. CI/conformance can force `smoke` via `--backend smoke` to validate the adapter contract without a GPU.
- **Windows-HIP** is `community-wanted` — no formal results yet. The `windows-hip` platform badge remains `community-wanted` across both model cards.
- **Provisioning scripts** (`adapter/setup/`) are stubs that document the steps; they do not automate full environment setup.
- **Platform-standard artifacts** (run_summary.json, provenance.json) are not yet generated for `results/omnidocbench/v16/linux-rocm/`. Existing results at `results/omnidocbench/v1.6/` are real, verified full-1651 predictions, but in legacy format. Run `omnidocbench-rocm score` + `publish` to migrate to platform-standard schema.
- **VLM empty outputs:** 2 of 1651 VLM pages produced empty predictions (recorded as failures).
- **Pipeline empty outputs:** 1 of 1651 pipeline pages produced an empty prediction.
- **Conformance** passes all structural checks; full `CONFORMANT` status requires platform-standard artifacts generated via `omnidocbench-rocm publish`.
- Full list: [`docs/known-gaps.md`](docs/known-gaps.md).
