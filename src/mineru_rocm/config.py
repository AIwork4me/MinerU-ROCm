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
# Env-overridable (matches BACKEND's pattern) so a single repo can run against any
# OpenAI-compatible VLM server without editing source. vlm_adapter falls back to
# http://127.0.0.1:8265/v1 + "mineru-pro" (serve_vlm_vllm.sh defaults) when these
# are empty, but the engine's `infer` stage never sees the CLI --server-url /
# --api-model-name flags (they're publish/run-only), so we surface them as env.
SERVER_URL = os.environ.get("MINERU_ROCM_SERVER_URL", "")
API_MODEL_NAME = os.environ.get("MINERU_ROCM_API_MODEL_NAME", "")
WEIGHTS_DIR = ""              # resolved at runtime; pipeline weights via mineru-models-download

def as_dict() -> dict:
    return {"backend": BACKEND, "model": MODEL, "server_url": SERVER_URL,
            "api_model_name": API_MODEL_NAME, "weights_dir": WEIGHTS_DIR}
