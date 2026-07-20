# Known gaps

Track the open items for `MinerU-ROCm` here. A `verified` badge requires these to be resolved or explicitly scoped. This list covers the **pipeline** path (MinerU 3.4 / `model: pipeline`); VLM-path gaps are tracked separately.

- **ONNX tables run on CPU on ROCm.** The RapidOCR / RapidTable ONNX models in the pipeline fall back to the CPU execution provider on ROCm builds. Output is correct but slow; an optional ROCm-EP patch (building `onnxruntime-rocm` and pointing the table OCR EP at it) is not wired up here.
- **`MINERU_FORMULA_CH_SUPPORT` must stay off.** Setting it to `true` pulls native PaddlePaddle-GPU, which has no ROCm wheel for this stack. Keep it `false` (default) and use the fallback formula renderer.
- **windows-hip unverified.** No results have been produced on a Windows + HIP/DirectML machine. The `windows-hip` platform is declared in `model_card.*.json` but its results dir is empty; validate end-to-end before claiming the badge.

## Deferred — upstream-PR-readiness backlog (2026-07-20)

Tracked here so they are not silently dropped (do not block the upstream PR):

- **Canary subset** not materialized — `reproducibility.lock.yaml` fields `canary_N.*`, `gt_json_canary_sha256`, `canary_manifest_sha256` are annotated `# (deferred → docs/known-gaps.md)`. Build via `mineru-rocm canary materialize` + a stratified manifest when picked up.
- **`pipeline_weights.table_sha256`** not recorded — table sub-models are pinned by the `PDF-Extract-Kit-1.0` `hf_revision ed6b654c`; record a representative file SHA when picked up.
- **`environment.inference.hip_visible_devices.pipeline`** `not_recorded` — the run_manifest `env` captured only torch/hip/transformers/vllm, not `HIP_VISIBLE_DEVICES`; only the VLM's GPU 0 is verified (from `examples/serve_vlm_vllm.sh`). Immaterial to reproduction: any of the 4× W7900 is equivalent, and `gpu_count_per_benchmark: 1` is the load-bearing fact.
- **v1.0.0 release** not cut — needs tag + wheel + `SHA256SUMS` + `release-artifact.md`/`release-checklist.md`.
- **`gpu-smoke.yml`** GPU-CI bridge not added (self-hosted gfx1100 runner topology TBD).
- **Docs**: `architecture.md`, `hardware-matrix.md`, `release-artifact.md`, `release-checklist.md` still missing (spec §8).
- **windows-hip** results still `community-wanted`.
