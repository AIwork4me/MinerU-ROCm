"""Adapter configuration for MinerU-ROCm.

Edit the defaults below to match how your model is served (local weights, a
vLLM/llama.cpp server, an ONNX runtime EP, ...). The adapter reads these via the
``config`` dict passed to :func:`run_adapter.run_adapter`; the CLI in
``run_adapter.py`` populates the same dict from its flags.
"""
from __future__ import annotations

# Inference backend. ``smoke`` ships a no-GPU placeholder so the repo is
# runnable out of the box. Replace with one of: vllm | llama-cpp-server |
# onnx-rocm | onnx-directml | <your-own> once you wire up ``_infer``.
BACKEND = "smoke"

# vLLM / OpenAI-compatible server URL (empty = spawn locally).
SERVER_URL = ""

# Model name as registered on the server (for API-style backends).
API_MODEL_NAME = "mineru2.5"

# Local weights directory (resolved by the adapter at run time).
WEIGHTS_DIR = ""


def as_dict() -> dict:
    return {
        "backend": BACKEND,
        "server_url": SERVER_URL,
        "api_model_name": API_MODEL_NAME,
        "weights_dir": WEIGHTS_DIR,
    }
