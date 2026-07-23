# Recommended inference backend per model type × platform

Pick the backend that best fits your model type and target platform. This is guidance, not a constraint — any backend that satisfies the `run_adapter` contract works.

| Model type | linux-rocm | windows-hip |
|---|---|---|
| pure VLM | vLLM/ROCm | llama.cpp/GGUF (HIP or Vulkan) |
| layout+VLM | ONNX `onnxruntime-rocm` (ROCm EP) + VLM server | ONNX `onnxruntime-directml` (DirectML EP, via Microsoft Olive) + VLM server |
| pipeline (MinerU 3.4) | MinerU on ROCm | ROCm PyTorch + DirectML ONNX; audited CPU override for `slanet-plus.onnx` |

## Windows DirectML path

For the Windows `onnxruntime-directml` path, follow the AMD Ryzen AI GPU documentation, which covers DirectML EP setup and model optimization via Microsoft Olive. The verified MinerU pipeline uses DirectML for compatible ONNX sessions, Windows ROCm PyTorch for layout/MFR/OCR, and routes only the DirectML-incompatible `slanet-plus.onnx` model to CPU:

https://ryzenai.docs.amd.com/en/latest/gpu/ryzenai_gpu.html

## Mapping backend → `config["backend"]`

Whatever backend you choose, set `adapter/adapter_config.py::BACKEND` (or pass `--backend <name>`) and branch on it inside `run_adapter` / `_infer`. The shipped `smoke` backend is the no-GPU default — keep it as a fallback so the repo stays runnable in CI without a GPU.
