# Architecture

MinerU-ROCm is an evaluation and reproducibility layer around upstream MinerU.
It does not fork model weights or replace the upstream model APIs.

## Runtime layers

1. `mineru-rocm` parses the command and validates user inputs.
2. The driver selects `pipeline` or `vlm-vllm` and runs the resumable page
   runner.
3. A backend converts one input page to Markdown. Per-page failures are
   recorded rather than aborting the complete dataset.
4. The validation and manifest layers enforce page conservation, non-empty
   predictions, stable filenames, and SHA-256 identity.
5. The pinned OmniDocBench scorer produces EditDist, TEDS, CDM, and the final
   Overall score.
6. OmniDocBench-ROCm packages self-contained model cards and evidence bundles.

## Backend boundaries

| Backend | Inference process | Accelerator path |
|---|---|---|
| Linux pipeline | In-process MinerU 3.4.4 | PyTorch ROCm; pipeline ONNX components may use CPU |
| Linux VLM | OpenAI-compatible vLLM server | vLLM ROCm on one gfx1100 GPU |
| Windows pipeline | In-process MinerU 3.4.4 | Windows ROCm PyTorch plus ONNX Runtime DirectML |
| Windows VLM | Not released | Phase 2 serving runtime remains exploratory |

On Windows, all compatible ONNX sessions are DirectML-first. The known
DirectML-incompatible `slanet-plus.onnx` control-flow model is routed through an
explicit CPU override. Runtime DirectML failures are counted and surfaced in
`_run_stats.json`; they are never silent.

## Evidence boundary

Published scores are derived only from committed `metric_result.json` files.
Each formal bundle also contains run statistics, provenance, dataset identity,
and a prediction manifest. Bulk Markdown predictions remain outside Git, but
every formal prediction is represented by filename and SHA-256 in its manifest.

The primary VLM card is `model_card.json`. The pipeline is a supplementary
result with `model_card.pipeline.json` and platform-specific cards. A
`community` badge is self-attested; promotion to `verified` requires an
independent maintainer reproduction.
