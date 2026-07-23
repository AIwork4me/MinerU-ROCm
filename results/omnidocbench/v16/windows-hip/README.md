# OmniDocBench v1.6 — Windows-HIP Phase 1

This directory is the self-contained community evidence bundle for the MinerU
3.4.4 pipeline on AMD Ryzen AI MAX+ 395 (Radeon 8060S, Strix Halo), Windows 11.
VLM Phase 2 is not included.

## Result

| Aggregation: `page.ALL` | Value |
|---|---:|
| Overall | **86.59** |
| Text EditDist | 0.0565 |
| Formula CDM | 83.39 |
| Table TEDS | 82.04 |
| Reading-order EditDist | 0.1531 |

Overall is `((1 - text_EditDist) * 100 + formula_CDM + table_TEDS) / 3`.
The score is +0.11 pp from the linux-rocm pipeline result (86.48) and passes
the Phase 1 tolerance of +/-1.0 pp around the 86.47 upstream reference.

## Integrity and execution evidence

- Dataset revision: `2b161d0`; GT SHA-256:
  `a45cd84b04ad8b793e775089640e6b681209abea33ead54c1828ddca35fae496`.
- Predictions: 1651 expected, 1651 non-empty Markdown files, 0 failed, 0
  runtime fallback. The prediction manifest records every file SHA-256.
- PyTorch: `2.9.1+rocm7.2.1`, HIP `7.2.53211`, GPU available on Radeon 8060S.
- Compatible ONNX models used DirectML. `slanet-plus.onnx` used the explicit,
  audited CPU override because its control-flow graph is incompatible with
  DirectML; it was not a silent runtime fallback.
- Native Windows CDM processed 2352 formula samples with no CDM
  timeout/error/exception; TEDS processed 665 tables with no timeout/error.
- One page whose only recognized text was a discarded header was recovered
  from MinerU's content list only after its complete Markdown output was empty.

The canonical published values come from `metric_result.json` at
`text_block.page.Edit_dist.ALL`, `display_formula.page.CDM.ALL`, and
`table.page.TEDS.ALL`. The sample-level `*.all` values are a different
aggregation and must not be used for the headline Overall.

Validate this bundle with:

```powershell
omnidocbench-rocm validate-bundle results\omnidocbench\v16\windows-hip `
  --model-card model_card.pipeline.windows-hip.json
```
