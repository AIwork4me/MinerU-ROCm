#!/usr/bin/env bash
# One-command pipeline smoke demo for MinerU-ROCm.
#
# Runs the REAL pipeline backend (mineru[all] in-process) on examples/sample.png
# (a real OmniDocBench PPT page — examples/demo.png is a 1x1 placeholder, kept
# only so the examples/ dir is non-empty for conformance). Targets GPU 3 of the
# 4x gfx1100 host (override with HIP_VISIBLE_DEVICES).
#
# Requirements (Task 4 provisioning):
#   - venv: set MINERU_ROCM_VENV (torch 2.14.0.dev+rocm7.2, mineru 3.4.4)
#   - weights: a HF cache holding opendatalab/PDF-Extract-Kit-1.0
#   - see docs/spike-mineru-api.md for the full env contract
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(dirname "$HERE")"
SAMPLE="${HERE}/sample.png"

# --- run-time env (set BEFORE python starts; pipeline_adapter also setdefaults
# the same vars inside the process for belt-and-braces) ---
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-3}"
export MINERU_DEVICE_MODE=cuda
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
# The dispatcher (run_adapter.py) uses local mineru_rocm.types — no engine import.
# If optional omnidocbench-rocm platform integration is needed, put it on PYTHONPATH.
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}<workspace>/omnidocbench-rocm/engine"

# Activate the mineru venv (overlay ROCm torch already applied).
# shellcheck disable=SC1091
source "${MINERU_ROCM_VENV:?set MINERU_ROCM_VENV to the mineru-rocm venv path}/bin/activate"

OUT_DIR="${MINERU_DEMO_OUT:-/tmp/mineru-demo-out}"
mkdir -p "$OUT_DIR"

# Stage only the real sample into a fresh img dir — the dispatcher iterates
# every image in --img-dir, so we exclude demo.png (1x1 placeholder) to avoid
# wasting a GPU page on it. Keep demo.png in examples/ for conformance non-empty.
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
cp "$SAMPLE" "$STAGE/"

# Run the dispatcher on the single real sample image.
python "$REPO/adapter/run_adapter.py" \
  --img-dir "$STAGE" \
  --out-dir "$OUT_DIR" \
  --platform linux-rocm \
  --backend pipeline

echo "--- demo output ($OUT_DIR/sample.md) ---"
sed -n '1,60p' "$OUT_DIR/sample.md"
echo "--- _run_stats.json ---"
cat "$OUT_DIR/_run_stats.json"
