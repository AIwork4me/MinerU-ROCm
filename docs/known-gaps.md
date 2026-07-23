# Known gaps

Track the open items for `MinerU-ROCm` here. A `verified` badge requires these
to be resolved or explicitly scoped. This list covers the pipeline path
(MinerU 3.4); VLM-path gaps are tracked separately.

- **ONNX tables run on CPU on Linux ROCm.** The RapidOCR / RapidTable ONNX
  models in the pipeline use the CPU execution provider on the tested Linux
  ROCm build. Output is correct but slower; an optional ONNX Runtime ROCm EP
  path is not wired up here.
- **`MINERU_FORMULA_CH_SUPPORT` must stay off.** Setting it to `true` pulls
  native PaddlePaddle-GPU, which has no ROCm wheel for this stack. Keep it
  `false` and use the fallback formula renderer.
- **Windows-HIP VLM remains unverified.** Pipeline Phase 1 is complete on
  Strix Halo (1651/1651, Overall 86.59, conformant CDM bundle). MinerU2.5 VLM
  Phase 2 still needs a viable Windows serving runtime.

## Deferred upstream-readiness backlog

These items do not invalidate the committed full-set result bundles:

- **Canary subset** is not materialized. The corresponding lock fields remain
  explicitly deferred. Build it with `mineru-rocm canary materialize` and a
  stratified manifest when picked up.
- **`pipeline_weights.table_sha256`** is not recorded. Table sub-models remain
  pinned by the `PDF-Extract-Kit-1.0` revision `ed6b654c`; record a
  representative file SHA when picked up.
- **`environment.inference.hip_visible_devices.pipeline`** is not recorded in
  the historic Linux run manifest. The benchmark used one of four equivalent
  W7900 GPUs; `gpu_count_per_benchmark: 1` is the load-bearing fact.
- **`gpu-smoke.yml`** is not present because a safe self-hosted gfx1100 runner
  topology has not been established.
- **Windows-HIP VLM** remains `community-wanted`; the pipeline result is
  `community`.
