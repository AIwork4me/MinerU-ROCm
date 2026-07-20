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
| MinerU2.5-Pro VLM | vLLM-on-ROCm `0.16.1.dev0+g89a77b108.d20260317` (http-client) | **95.46** | 0.0360 | 96.46 | 93.54 |

Both: OmniDocBench v1.6, 1651 pages, AMD gfx1100 (Radeon PRO W7900, 48 GB; 1 GPU per benchmark, host has 4×),
ROCm 7.2, bf16. Scored via OmniDocBench's `pdf_validation.py` (quick_match).

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
- **vLLM non-determinism.** The VLM backend uses vLLM, which has ~0.1pp
  run-to-run variation (the P2/P3 re-run scored 95.46 vs the prior 95.56 —
  Δ−0.10pp, within the ±0.5pp gate). The pipeline backend is deterministic
  (byte-identical across runs).
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

# 3. Score
mineru-rocm score \
  --gt-json <OmniDocBench.json> \
  --pred-dir <output/> \
  --label pipeline

# 4. Verify the manifest (conservation laws)
mineru-rocm manifest verify --pred-dir <output/>
```

Every input is pinned in `reproducibility.lock.yaml` (code commit, model weight
SHAs, dataset SHA, scorer commit, environment versions). See that file for the
exact provenance.
