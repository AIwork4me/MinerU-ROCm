"""Adapter configuration for MinerU-ROCm.

backend selects the path; model is advisory (which MinerU model a run targets).
"""
from __future__ import annotations

import os

# smoke = no-GPU CI placeholder (CI/conformance only). Real: pipeline | vlm-vllm | vlm-transformers.
# Env-overridable so a single repo can run either model: the engine's `infer` does not pass
# --backend, so this default is the source of truth (pipeline for the 3.4 pipeline; set
# MINERU_ROCM_BACKEND=vlm-vllm for the VLM). CI/conformance forces smoke via the unit tests.
BACKEND = os.environ.get("MINERU_ROCM_BACKEND", "pipeline")
# Which MinerU model this run targets: "pipeline" (3.4) | "vlm" (2.5-Pro).
MODEL = "pipeline"
SERVER_URL = ""               # VLM OpenAI-compatible server (empty = spawn locally)
API_MODEL_NAME = "mineru2.5"  # VLM model name as registered on the server
WEIGHTS_DIR = ""              # resolved at runtime; pipeline weights via mineru-models-download

def as_dict() -> dict:
    return {"backend": BACKEND, "model": MODEL, "server_url": SERVER_URL,
            "api_model_name": API_MODEL_NAME, "weights_dir": WEIGHTS_DIR}
