#!/usr/bin/env bash
# MinerU-ROCm — Linux/ROCm provisioning. Venv + weights on /root (NOT /workspace).
#
# Idempotent: safe to re-run. Creates an isolated venv at
# $MINERU_ROCM_VENV (default /root/ocr-eval/mineru-rocm-venv) and installs
# mineru[all] + the ROCm torch wheel that matches the system (ROCm 7.2).
#
# Environment caveats (see docs/spike-mineru-api.md):
#   - /workspace is a 10GB NFS PVC; the venv + multi-GB weights MUST live on /root.
#   - mineru[all] on Linux pulls mineru[vllm] -> torch>=2.6; that wheel is the
#     CUDA build and torch.cuda.is_available() will be False on ROCm. We
#     therefore REINSTALL the ROCm torch wheel afterwards from the pytorch
#     nightly/rocm7.2 index, which restores cuda.is_available()==True.
#   - huggingface.co is blocked in this env; use the hf-mirror endpoint for
#     weight downloads.
set -euo pipefail

VENV="${MINERU_ROCM_VENV:-/root/ocr-eval/mineru-rocm-venv}"
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

# Pipeline weights (PP-DocLayoutV2, UniMERNet, PP-OCRv6, SLANet/UNet/PP-LCNet)
# via hf-mirror (huggingface.co is blocked in this env).
export HF_ENDPOINT="${HF_ENDPOINT:-http://134.199.133.77}"
export MINERU_MODEL_SOURCE="${MINERU_MODEL_SOURCE:-huggingface}"
echo "[00-install-deps] downloading pipeline weights from $HF_ENDPOINT"
mineru-models-download -s huggingface -m pipeline

echo "[00-install-deps] done. Activate: source $VENV/bin/activate"
