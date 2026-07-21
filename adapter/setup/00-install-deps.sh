#!/usr/bin/env bash
# MinerU-ROCm — Linux/ROCm provisioning. Venv + weights on a host-local path.
#
# Idempotent: safe to re-run. Creates an isolated venv at
# $MINERU_ROCM_VENV (REQUIRED: set to a host-local path with enough space for
# multi-GB weights; see docs/spike-mineru-api.md) and installs mineru[all] +
# the ROCm torch wheel that matches the system (ROCm 7.2).
#
# Environment caveats (see docs/spike-mineru-api.md):
#   - Pick a disk with enough room for the venv + multi-GB weights; a small
#     NFS PVC will not work.
#   - mineru[all] on Linux pulls mineru[vllm] -> torch>=2.6; that wheel is the
#     CUDA build and torch.cuda.is_available() will be False on ROCm. We
#     therefore REINSTALL the ROCm torch wheel afterwards from the pytorch
#     nightly/rocm7.2 index, which restores cuda.is_available()==True.
#   - HF_ENDPOINT defaults to huggingface.co; if you need a mirror (e.g. because
#     huggingface.co is blocked in your env), export HF_ENDPOINT before running.
set -euo pipefail

VENV="${MINERU_ROCM_VENV:?set MINERU_ROCM_VENV to a host-local path with enough space for multi-GB weights}"
# Prefer python3.11 (matches the platform eval-venv); fall back to python3.
PY="${PYTHON:-python3.11}"
[[ -x "$(command -v "$PY")" ]] || PY="python3"

echo "[00-install-deps] creating venv at $VENV using $("$PY" --version 2>&1)"
"$PY" -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -U pip

echo "[00-install-deps] installing mineru[all] (may pull a CUDA torch; we fix it next)"
pip install -U "mineru[all]"

# Verify torch + ROCm. mineru[all]->[vllm]->torch is the CUDA wheel on Linux; if
# cuda.is_available() is False, overlay the ROCm wheel from the pytorch index.
TORCH_OK="$(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo False)"
echo "[00-install-deps] pre-fix torch.cuda.is_available()=$TORCH_OK"
if [[ "$TORCH_OK" != "True" ]]; then
  echo "[00-install-deps] reinstalling ROCm torch (rocm7.2) over the CUDA wheel"
  # --force-reinstall --no-deps: pip sees the same version string (2.10.0) on
  # both indexes and would otherwise consider it satisfied; --no-deps avoids
  # disturbing the rest of the mineru[vllm] dep graph.
  pip install --pre torch torchvision --force-reinstall --no-deps \
    --index-url https://download.pytorch.org/whl/nightly/rocm7.2
fi

echo "[00-install-deps] verifying torch.cuda.is_available()"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'hip', torch.version.hip)"

# Pipeline weights (PP-DocLayoutV2, UniMERNet, PP-OCRv6, SLANet/UNet/PP-LCNet).
# HF_ENDPOINT defaults to the public endpoint; export it to a mirror if
# huggingface.co is blocked in your env.
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export MINERU_MODEL_SOURCE="${MINERU_MODEL_SOURCE:-huggingface}"
echo "[00-install-deps] downloading pipeline weights from $HF_ENDPOINT"
mineru-models-download -s huggingface -m pipeline

echo "[00-install-deps] done. Activate: source $VENV/bin/activate"

# ---------------------------------------------------------------------------
# VLM env (separate from this mineru-pipeline venv) — DOCUMENTATION ONLY.
# This script provisions $MINERU_ROCM_VENV (mineru pipeline + scorer-side
# deps above). The VLM adapter runs in a DIFFERENT, pre-built env:
#
#   VLM env   = a separate Py3.12 venv (e.g. created during a Plan-2 spike)
#     - vllm 0.16.1.dev0+rocm (gfx1100), mineru_vl_utils==1.0.5
#     - pip install -e <workspace>/omnidocbench-rocm   # so the engine-invoked
#       adapter subprocess resolves mineru_vl_utils AND omnidocbench_rocm.types
#     - GPU 0 only (ROCm single-card)
#
#   Scoring   = a separate Py3.11 venv with OmniDocBench v1.6
#     (set via OMNIDOCBENCH_VENV / OMNIDOCBENCH_REPO; see mineru-rocm score --help)
#
#   LD_LIBRARY_PATH must include /opt/rocm/lib before invoking either the VLM
#   venv python or vllm, so ROCm libs resolve on gfx1100.
#
# Verify (run by hand after provisioning, substituting your VLM venv python):
#   <vlm-venv>/bin/python -c "import vllm, mineru_vl_utils, omnidocbench_rocm; \
#       print('vlm env ok', vllm.__version__)"
#   # -> vlm env ok 0.16.1.dev0+...
# ---------------------------------------------------------------------------
