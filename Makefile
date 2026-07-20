PLATFORM ?= linux-rocm
BACKEND  ?= pipeline
GT_JSON  ?= OmniDocBench.json
IMAGES_DIR ?= images
PRED_DIR ?= $(IMAGES_DIR)-preds-$(BACKEND)
SCORER_VENV ?= $(VIRTUAL_ENV)

setup-linux:
	bash adapter/setup/00-install-deps.sh
setup-windows:
	powershell -ExecutionPolicy Bypass -File adapter\setup\00-install-deps.ps1

demo:
	OUT=$$(mktemp -d); python adapter/run_adapter.py --img-dir examples --out-dir $$OUT --platform $(PLATFORM) --backend smoke; ls $$OUT

predict:
	mineru-rocm predict --backend $(BACKEND) \
	  --gt-json $(GT_JSON) --images-dir $(IMAGES_DIR) \
	  --pred-dir $(PRED_DIR) --platform $(PLATFORM)

score:
	mineru-rocm score --gt-json $(GT_JSON) --pred-dir $(PRED_DIR) \
	  --label $(BACKEND) --venv-python $(SCORER_VENV)/bin/python

# Full OmniDocBench v1.6 eval = predict + score for both backends (linux-rocm).
# Override paths via env: make eval-linux GT_JSON=x IMAGES_DIR=y PRED_DIR=z SCORER_VENV=/path
eval-linux eval-windows:
	$(MAKE) predict BACKEND=pipeline
	$(MAKE) score    BACKEND=pipeline
	$(MAKE) predict BACKEND=vlm-vllm
	$(MAKE) score    BACKEND=vlm-vllm

publish:
	omnidocbench-amd conformance . && echo CONFORMANT

smoke-test:
	python -m pytest
