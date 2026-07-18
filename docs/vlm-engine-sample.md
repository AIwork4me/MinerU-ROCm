# VLM engine comparison — sample-informed decision (vLLM chosen over transformers)

> Date: 2026-07-18. Decision: **vLLM-on-ROCm is the VLM primary backend.** transformers
> is the slow fallback (clean but ~44 h for a full eval). A full 1651-page vLLM eval is
> running for the real Overall (see `docs/reproducibility.md`).

## Why a sample-informed choice
Both engines were wired (`vlm-vllm` via `MinerUClient(http-client)`, `vlm-transformers`
via `MinerUClient(backend="transformers")` with `no_repeat_ngram_size=100`). Before
committing to hours-long full evals, both were sampled on a representative OmniDocBench v1.6
subset. Two open risks motivated this: (1) a known vLLM EOS-first-token regression that can
empty-out pages, and (2) transformers being ~2× slower per page.

## Results

### vLLM-on-ROCm (primary) — 100-page representative sample
- **Overall ≈ 97.0** (page.ALL convention, same as the pipeline's 86.48):
  `((1 − 0.0335)×100 + 96.89 + 97.53)/3` → text EditDist **0.0335** (→ 96.65),
  Formula CDM **97.53**, Table TEDS **96.89**.
- **0 empty pages** out of 100 (the EOS-first-token concern did **not** materialize on this
  sample — unlike the Unlimited-OCR finding in memory; appears model/page-specific here).
- Speed: ~10–30 s/page → full 1651-page eval feasible in hours.
- vs official MinerU2.5-Pro (text 0.036 / CDM 97.45 / TEDS 93.42 / Overall 95.75): the
  sample is at/above official (the 100-page sample skews easier than the full set incl. hard
  subsets; the full eval gives the real number).

### transformers (secondary/fallback) — 25-page sample
- **0 empty pages** (confirms transformers is clean — no EOS issue; consistent with
  `no_repeat_ngram_size=100` applied via HF `generate`).
- Speed: ~100–150 s/page steady-state → **full 1651-page eval ≈ 44 h** (impractical as the
  primary path). Quality expected to match vLLM (same two-step, same no-repeat-100-gram);
  not full-evaluated.

## Decision
- **vLLM-on-ROCm = primary** (fast, high quality, full-eval-feasible). Full 1651-page eval
  running for the real Overall.
- **transformers = reported secondary** (clean, but too slow for a full eval; documented as
  the slow fallback). Useful as a correctness cross-check / fallback if a future vLLM ROCm
  wheel regression breaks boot.
- The EOS-first-token risk that motivated the "both engines" hedge did not materialize for
  MinerU2.5-Pro on the sampled pages — vLLM is viable here. (The hedge still paid off: it
  gave a clean fallback + confirmed vLLM's quality before the long full run.)

## Reproduction
- vLLM sample preds: `/root/ocr-eval/vlm-vllm-sample-preds` (100 pages); scored against the
  100-page GT subset `/root/ocr-eval/OmniDocBench_data/OmniDocBench_vlm_sample100.json`.
- transformers sample preds: `/root/ocr-eval/vlm-transformers-sample-preds` (25 pages).
- Scoring via `pdf_validation.py` (the engine's linux-rocm `score()` is a stub — Plan 1).
