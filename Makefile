PLATFORM     ?= linux-rocm
VERSION      ?= v16
REVISION     ?= 2b161d0
BACKEND      ?= pipeline
GT_JSON      ?= OmniDocBench.json
IMAGES_DIR   ?= images
PRED_DIR     ?= $(IMAGES_DIR)-preds-$(BACKEND)
SCORER_VENV  ?= $(VIRTUAL_ENV)
RESULTS_ROOT ?= results/omnidocbench/$(VERSION)/linux-rocm

# Clean full run by default: CDM scoring ON, --skip-existing OFF.
#   make eval-mineru2.5-linux             # clean CDM run
#   make eval-mineru2.5-linux RESUME=1    # resume an interrupted run
#   make eval-mineru2.5-linux CDM=0       # quick debug score without CDM
CDM ?= 1
RESUME ?= 0
CDM_FLAG = $(if $(filter 1,$(CDM)),--cdm,)
RESUME_FLAG = $(if $(filter 1,$(RESUME)),--skip-existing,)

# ── provisioning ──────────────────────────────────────────────────────────────

setup-linux:
	bash adapter/setup/00-install-deps.sh

setup-windows:
	powershell -ExecutionPolicy Bypass -File adapter\setup\00-install-deps.ps1

# ── demo / smoke ──────────────────────────────────────────────────────────────

demo:
	OUT=$$(mktemp -d); python adapter/run_adapter.py --img-dir examples --out-dir $$OUT --platform $(PLATFORM) --backend smoke; ls $$OUT

smoke-test:
	python -m pytest

# ── mineru-rocm CLI (developer tools, standalone) ─────────────────────────────

predict:
	mineru-rocm predict --backend $(BACKEND) \
	  --gt-json $(GT_JSON) --images-dir $(IMAGES_DIR) \
	  --pred-dir $(PRED_DIR) --platform $(PLATFORM)

score:
	mineru-rocm score --gt-json $(GT_JSON) --pred-dir $(PRED_DIR) \
	  --label $(BACKEND) --venv-python $(SCORER_VENV)/bin/python

# ── OmniDocBench-ROCm platform evaluation ─────────────────────────────────────

# MinerU2.5-Pro VLM (primary model card)
eval-mineru2.5-linux:
	omnidocbench-rocm run \
	  --stage all \
	  --platform linux-rocm \
	  --version v16 \
	  --revision $(REVISION) \
	  --adapter adapter/run_adapter.py \
	  --model-id mineru2.5 \
	  --backend vlm-vllm \
	  --server-url http://127.0.0.1:8265/v1 \
	  --api-model-name mineru-pro \
	  --git-commit "$$(git rev-parse HEAD)" \
	  --results-dir $(RESULTS_ROOT) \
	  $(CDM_FLAG) \
	  $(RESUME_FLAG)

# MinerU 3.4 Pipeline (supplementary)
eval-pipeline-linux:
	omnidocbench-rocm run \
	  --stage all \
	  --platform linux-rocm \
	  --version v16 \
	  --revision $(REVISION) \
	  --adapter adapter/run_adapter.py \
	  --model-id mineru-pipeline \
	  --backend pipeline \
	  --git-commit "$$(git rev-parse HEAD)" \
	  --results-dir $(RESULTS_ROOT) \
	  $(CDM_FLAG) \
	  $(RESUME_FLAG)

# ── conformance ───────────────────────────────────────────────────────────────

conformance:
	omnidocbench-rocm conformance . && echo CONFORMANT

.PHONY: setup-linux setup-windows demo smoke-test predict score \
        eval-mineru2.5-linux eval-pipeline-linux conformance
