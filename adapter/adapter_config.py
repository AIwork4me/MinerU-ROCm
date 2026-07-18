"""Adapter configuration for MinerU-ROCm.

backend selects the path; model is advisory (which MinerU model a run targets).
"""
from __future__ import annotations

# smoke = no-GPU CI placeholder. Real: pipeline | vlm-vllm | vlm-transformers.
BACKEND = "smoke"
# Which MinerU model this run targets: "pipeline" (3.4) | "vlm" (2.5-Pro).
MODEL = "pipeline"
SERVER_URL = ""               # VLM OpenAI-compatible server (empty = spawn locally)
API_MODEL_NAME = "mineru2.5"  # VLM model name as registered on the server
WEIGHTS_DIR = ""              # resolved at runtime; pipeline weights via mineru-models-download

def as_dict() -> dict:
    return {"backend": BACKEND, "model": MODEL, "server_url": SERVER_URL,
            "api_model_name": API_MODEL_NAME, "weights_dir": WEIGHTS_DIR}
