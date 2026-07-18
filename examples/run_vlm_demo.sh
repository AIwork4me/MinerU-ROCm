#!/usr/bin/env bash
# End-to-end VLM smoke on a few real OmniDocBench pages (GPU 0).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
# 1) start the server (idempotent: skip if already up)
if ! curl -sf http://127.0.0.1:8265/v1/models >/dev/null 2>&1; then
  bash "$REPO/examples/serve_vlm_vllm.sh"
  bash "$REPO/examples/wait_vlm.sh"
fi
# 2) a tiny img-dir of 3 real pages
SMALL=/tmp/vlm-smoke-imgs; rm -rf "$SMALL"; mkdir -p "$SMALL"
ls /root/ocr-eval/OmniDocBench_data/images/ | grep -E 'PPT|exam_paper|color_textbook' | head -3 \
  | while read f; do ln -sf "/root/ocr-eval/OmniDocBench_data/images/$f" "$SMALL/$f"; done
# 3) run the adapter (VLM env; MINERU_ROCM_BACKEND=vlm-vllm; server_url default)
export MINERU_ROCM_BACKEND=vlm-vllm
OUT=/tmp/vlm-smoke-out; rm -rf "$OUT"
/opt/venv/bin/python "$REPO/adapter/run_adapter.py" \
  --img-dir "$SMALL" --out-dir "$OUT" --platform linux-rocm \
  --server-url http://127.0.0.1:8265/v1 --api-model-name mineru-pro
echo "--- smoke outputs ---"; for f in "$OUT"/*.md; do echo "== $f =="; head -8 "$f"; done
cat "$OUT/_run_stats.json" | python -c "import json,sys;d=json.load(sys.stdin);print('engine',d['engine'],'ok',d['ok'],'fail',d['fail'])"
