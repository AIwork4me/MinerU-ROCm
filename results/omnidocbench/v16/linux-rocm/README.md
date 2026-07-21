# Platform-standard results — v16/linux-rocm

This directory holds the **OmniDocBench-ROCm platform-standard, self-contained
result bundles** for the `linux-rocm` platform, generated on **2026-07-21** by
`omnidocbench-rocm publish` (platform version 0.3.1).

## Canonical bundles

One bundle per model (CDM-scored). Each bundle is self-contained — every
artifact reference resolves within this directory:

```
mineru2.5_v16_quick_match_cdm_        # the primary registry model (mineru2.5)
  {metric_result,run_stats,run_summary,provenance,prediction_manifest,
   dataset_identity,scoring_config}.json/.yaml
mineru-pipeline_v16_quick_match_cdm_  # the supplementary pipeline model
  {…same seven artifacts…}
```

- **mineru2.5** (VLM, vLLM-on-ROCm): Overall **95.56** — 1651 pages attempted,
  1649 non-empty predictions, 2 empty (failures), 0 fallback. Formula CDM 96.73,
  Text EditDist 0.0359, Table TEDS 93.54, read-order EditDist 0.1240.
- **mineru-pipeline** (MinerU 3.4 pipeline): Overall **86.48** — 1651 attempted,
  1650 non-empty, 1 empty.

`provenance.json` distinguishes the **packaging commit** (this `publish`) from
the **`prediction_source_commit`** (the real inference commit, recorded in the
legacy `run_manifest.json`); runtime source paths are redacted to `<eval-root>`.

## Validation

```bash
omnidocbench-rocm validate-bundle results/omnidocbench/v16/linux-rocm \
  --model-card model_card.json
# → CONFORMANT
```

## Legacy results

The legacy results under `results/omnidocbench/v1.6/{vlm-vllm,pipeline}/` are
**retained for historical comparison and prediction-source provenance** (the
`run_manifest.json` + `metric_result.json` + `sample_predictions/` from the
standalone `mineru-rocm` path). The prior standalone VLM score was **95.46**
(Formula CDM 96.46) on the same 1651 predictions — superseded as the canonical
result by the platform CDM **95.56** above.
