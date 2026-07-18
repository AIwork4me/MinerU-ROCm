PLATFORM ?= linux-rocm
VERSION  ?= v16
REVISION ?= v1.6
MODEL_ID ?= mineru2.5
# Clean 1651-image dir (GT only; the legacy shared images/ dir has 1742 entries —
# 91 surplus images the scorer silently ignores). Override per-run if needed.
OMNIDOCBENCH_IMG_DIR ?= /root/ocr-eval/OmniDocBench_v16_images

setup-linux:
	bash adapter/setup/00-install-deps.sh
setup-windows:
	powershell -ExecutionPolicy Bypass -File adapter\setup\00-install-deps.ps1

demo:
	OUT=$$(mktemp -d); omnidocbench-amd infer --adapter adapter/run_adapter.py --img-dir examples --out-dir $$OUT --platform $(PLATFORM); ls $$OUT

eval-linux eval-windows:
	omnidocbench-amd run --stage all --platform $(PLATFORM) --version $(VERSION) --revision $(REVISION) \
	  --adapter adapter/run_adapter.py --model-id $(MODEL_ID) --img-dir $(OMNIDOCBENCH_IMG_DIR) \
	  --git-commit $$(git rev-parse HEAD) --results-dir results/omnidocbench/$(VERSION)/$(PLATFORM)

publish:
	omnidocbench-amd conformance . && echo CONFORMANT

smoke-test:
	python -m pytest
