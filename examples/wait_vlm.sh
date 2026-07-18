#!/usr/bin/env bash
# examples/wait_vlm.sh — block until /v1/models responds, up to ~5 min.
PORT="${PORT:-8265}"
for i in $(seq 1 60); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    echo "[wait_vlm] server ready: $(curl -sf http://127.0.0.1:$PORT/v1/models)"
    exit 0
  fi
  sleep 5
done
echo "[wait_vlm] server did NOT become ready; tail of log:"; tail -20 /tmp/vlm-vllm.log; exit 1
