#!/usr/bin/env bash
# Launch the MinerU2.5-Pro vLLM-on-ROCm server (GPU 0). Background; poll /v1/models.
set -euo pipefail
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export HSA_OVERRIDE_GFX_VERSION=11.0.0          # gfx1100 / W7900 RDNA3
export VLLM_USE_V1=1                             # MANDATORY: v1 logits-processor API
export HF_ENDPOINT="${HF_ENDPOINT:-http://134.199.133.77}"
export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"

MODEL_DIR="$(ls -d /root/.cache/huggingface/models--opendatalab--MinerU2.5-Pro-2605-1.2B/snapshots/* | head -1)"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${PORT:-8265}"
LOG="${LOG:-/tmp/vlm-vllm.log}"

nohup /opt/venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --served-model-name mineru-pro \
  --trust-remote-code \
  --dtype bfloat16 \
  --chat-template "$REPO/adapter/qwen2vl_chat_template.jinja" \
  --logits-processors mineru_vl_utils:MinerULogitsProcessor \
  --host 127.0.0.1 --port "$PORT" \
  --gpu-memory-utilization 0.70 \
  --max-model-len 8192 \
  --limit-mm-per-prompt '{"image": 1}' \
  --enforce-eager \
  > "$LOG" 2>&1 &
echo $! > /tmp/vlm-vllm.pid
echo "[serve_vlm_vllm] launched pid $(cat /tmp/vlm-vllm.pid), log $LOG"
