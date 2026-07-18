# Known gaps

Track the open items for `MinerU-ROCm` here. A `verified` badge requires these to be resolved or explicitly scoped. This list covers the **pipeline** path (MinerU 3.4 / `model: pipeline`); VLM-path gaps are tracked separately.

- **ONNX tables run on CPU on ROCm.** The RapidOCR / RapidTable ONNX models in the pipeline fall back to the CPU execution provider on ROCm builds. Output is correct but slow; an optional ROCm-EP patch (building `onnxruntime-rocm` and pointing the table OCR EP at it) is not wired up here.
- **`MINERU_FORMULA_CH_SUPPORT` must stay off.** Setting it to `true` pulls native PaddlePaddle-GPU, which has no ROCm wheel for this stack. Keep it `false` (default) and use the fallback formula renderer.
- **windows-hip unverified.** No results have been produced on a Windows + HIP/DirectML machine. The `windows-hip` platform is declared in `model_card.*.json` but its results dir is empty; validate end-to-end before claiming the badge.
