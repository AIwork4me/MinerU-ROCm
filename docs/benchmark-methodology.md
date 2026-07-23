# Benchmark Methodology

> **TL;DR:** MinerU-ROCm is **evaluation-backed, not precision-aligned.** It runs
> the upstream opendatalab/MinerU models on AMD ROCm and scores them on
> OmniDocBench v1.6 — the same dataset + scorer the upstream uses. It does NOT
> claim bit-identical results vs a CUDA baseline (no same-engine CUDA control
> exists). The numbers below are reproducible from `reproducibility.lock.yaml`.

## What this repo measures

| Model | Backend | Overall | Text EditDist | Formula CDM | Table TEDS |
|---|---|---:|---:|---:|---:|
| MinerU 3.4 pipeline | in-process (MinerUPipelineRunner on ROCm) | **86.48** | 0.0566 | 83.07 | 82.04 |
| MinerU 3.4 pipeline | Windows ROCm PyTorch + DirectML ONNX | **86.59** | 0.0565 | 83.39 | 82.04 |
| MinerU2.5-Pro VLM | vLLM-on-ROCm `0.16.1.dev0+g89a77b108.d20260317` (http-client) | **95.56** | 0.0359 | 96.73 | 93.54 |

All rows use OmniDocBench v1.6, 1651 pages, `quick_match`, and CDM. Linux used
AMD gfx1100 (Radeon PRO W7900, 48 GB; one GPU per benchmark), ROCm 7.2. Windows
used Ryzen AI MAX+ 395 / Radeon 8060S with Windows ROCm 7.2.1 and DirectML.
(The prior standalone `mineru-rocm score` path scored the VLM at 95.46 —
Formula CDM 96.46 — on the same predictions; Δ +0.10 pp, entirely the CDM
configuration.)

## "Evaluation-backed" — what it means

This repo takes the **upstream model** (the same MinerU 3.4 pipeline +
MinerU2.5-Pro VLM from opendatalab/MinerU), runs it on **AMD ROCm** (not CUDA),
and scores it on the **same OmniDocBench v1.6 benchmark** with the **same
scorer** (`pdf_validation.py`). The scores are methodologically comparable to any other
platform that runs the same model + same dataset + same scorer — but this is **not**
a controlled CUDA-vs-ROCm hardware-level comparison.

## "Not precision-aligned" — what it does NOT claim

- **No CUDA control.** There is no side-by-side CUDA baseline on the same
  hardware. The ROCm-vs-CUDA delta is not measured. If you need that, run MinerU
  on an NVIDIA GPU with the same model + dataset + scorer and compare.
- **Scoring-configuration delta, not inference drift.** The platform CDM-scored
  VLM Overall is **95.56**; the prior standalone-path score was **95.46** (same
  1651 predictions, scorer revision `2b161d0`). The Δ +0.10 pp is the Formula-CDM
  submetric (96.46 → 96.73), attributable to the CDM scoring configuration at
  scoring time — not vLLM run-to-run drift (which was not separately isolated).
  The pipeline backend is deterministic within a fixed environment; the
  Windows-HIP result differs from linux-rocm by +0.11 pp, within the declared
  cross-platform tolerance.
- **Empty pages.** The VLM produces ~0.12% empty-output pages (2/1651) — pages
  where the model returned no content. These score 0 for that page; the Overall
  (averaged over 1651) absorbs the impact (~0.01pp).
- **The official anchor is community-verified.** The upstream README "Local
  Deployment" table records vlm-engine 95.30 / pipeline 86.47; we cite these
  as the comparison anchors (aligned to that table, not as official support).
  See `reproducibility.lock.yaml` (`benchmark.official_reference: source:
  verified`). The prior withdrawn unofficial anchor is no longer cited.

## The Overall formula

```
Overall = ((1 - text_EditDist) × 100 + formula_CDM × 100 + table_TEDS × 100) / 3
```

Reading-order EditDist is reported separately and is **NOT** part of Overall
(following the OmniDocBench v1.6 convention).

## How to reproduce

```bash
# 1. Install
pip install -e .[dev,platform]

# 2. Run the pipeline (or VLM) on the OmniDocBench v1.6 dataset
mineru-rocm predict --backend pipeline \
  --gt-json <OmniDocBench.json> \
  --images-dir <images/> \
  --pred-dir <output/> \
  --platform linux-rocm

# 3. Score (needs the OmniDocBench scorer repo + venv)
mineru-rocm score \
  --gt-json <OmniDocBench.json> \
  --pred-dir <output/> \
  --label pipeline \
  --omnidocbench-repo <OmniDocBench> --venv-python <scorer-venv>/bin/python

# 4. Verify the manifest (conservation laws)
mineru-rocm manifest verify --pred-dir <output/>
```

Every input is pinned in `reproducibility.lock.yaml` (code commit, model weight
SHAs, dataset SHA, scorer commit, environment versions). See that file for the
exact provenance.
