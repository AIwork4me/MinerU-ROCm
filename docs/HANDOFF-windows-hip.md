# Windows-hip verification handoff — MinerU-ROCm pipeline

> **Who does what.** The **Linux / `linux-rocm`** side is verified by Claude on
> this org's 4× AMD gfx1100 (Radeon PRO W7900) host. The **Windows /
> `windows-hip`** side is handed off to a colleague to run in parallel on
> **Ryzen AI MAX+ 395 (Strix Halo)**. This document is self-contained: everything
> you need to reproduce the MinerU 3.4 **pipeline** result on Windows is here.
>
> Scope of *this* handoff: the **MinerU 3.4 `pipeline` backend** (target
> OmniDocBench v1.6 Overall **≈ 86.47**, within ±1.0 pp). The **MinerU2.5-Pro
> VLM** (`mineru2.5`) is a separate model card / Plan 2 — its Windows path
> (transformers-DirectML or llama.cpp-GGUF) is noted at the end as future work,
> not part of this handoff.

---

## 0. What you are verifying, and the honest framing

You are reproducing OpenDataLab's MinerU 3.4 pipeline **OmniDocBench v1.6 Overall
= 86.47** on AMD hardware (Strix Halo), producing real prediction artifacts +
provenance that this repo can publish under
`results/omnidocbench/v1.6/windows-hip/`.

**The bar is "evaluation-backed", not "bit-exact-CUDA".** MinerU's official 86.47
is upstream's number; our claim is "we reproduce that number on AMD on the full
1651-page set, with provenance naming the engine/hardware we used." A small
cross-device delta (GPU-vs-CPU, bf16 kernels) is expected and acceptable within
**±1.0 pp**. If your Windows Overall lands outside ±1.0 pp of 86.47, stop and
report — don't publish as verified.

**Important expectation (Strix Halo / Windows):** MinerU's pipeline is now
end-to-end pure-PyTorch + 3 ONNX table models. On Linux/ROCm the PyTorch models
run on the GPU. On **Windows without a CUDA/ROCm-runtime GPU**,
`torch.cuda.is_available()` is `False`, so mineru's `get_device()` defaults to
**CPU** for the PyTorch models (layout / OCR / UniMERNet). The OmniDocBench
**score is device-independent to within tolerance** (same BF16 weights, same
inputs), so a CPU run still reproduces ≈86.47 — it is just **slower**, not wrong.
DirectML acceleration of the ONNX table models is an optional speed-up (§3),
not a correctness requirement. Set your expectation: **correctness first, speed
second.**

---

## 1. Target environment

- **Hardware:** AMD Ryzen AI MAX+ 395 (Strix Halo) — integrated Radeon (RDNA 3.5).
- **OS:** Windows 11.
- **Acceleration:** DirectML (`onnxruntime-directml`, `DmlExecutionProvider`) for
  the ONNX table models. Reference:
  https://ryzenai.docs.amd.com/en/latest/gpu/ryzenai_gpu.html
- **Python:** 3.11 (preferred; matches the Linux venv) or 3.12. `mineru` requires
  `>=3.10,<3.14`.

---

## 2. Install (Windows, PowerShell)

```powershell
# 1. Clone the repo
git clone https://github.com/AIwork4me/MinerU-ROCm.git
cd MinerU-ROCm
# checkout the verified commit recorded in docs/reproducibility.md (Linux side)

# 2. Create a venv (keep it OFF any small/Network drive — MinerU + weights are GBs)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip

# 3. MinerU pipeline (pure-PyTorch on Windows; runs on CPU by default — see §0)
pip install -U "mineru[all]"

# 4. ONNX table models on DirectML (optional speed-up; correct on CPU otherwise)
pip install onnxruntime-directml     # provides DmlExecutionProvider

# 5. The platform engine (for score / publish / conformance)
pip install omnidocbench-amd
pip install -e .                     # this repo (adapter + pyproject)

# 6. Pipeline weights (PP-DocLayoutV2, UniMERNet-small, PP-OCRv6, SLANet/UNet)
#    On Windows, ModelScope is usually faster than HF; pick one:
$env:MINERU_MODEL_SOURCE="modelscope"; mineru-models-download -s modelscope -m pipeline
#  or:
$env:HF_ENDPOINT="https://hf-mirror.com"; mineru-models-download -s huggingface -m pipeline
```

Sanity check:
```powershell
python -c "import mineru, onnxruntime as ort; print('mineru ok'); print('EPs:', ort.get_available_providers())"
```
You should see `mineru ok` and `DmlExecutionProvider` in the provider list
(confirming DirectML is available for the table models).

---

## 3. (Optional) DirectML for the ONNX table models

MinerU selects the ONNX execution provider in
`mineru/model/table/rec/onnxruntime_provider.py`, which only enables
`CUDAExecutionProvider`. On Windows/DirectML the table models therefore fall
back to `CPUExecutionProvider` (correct, slow) **unless** you patch the selector
to also accept `DmlExecutionProvider`. This is a **speed optimization only** —
skip it for the first verification run; tables on CPU are correct.

If you want the speed-up, a one-line patch: in
`onnxruntime_provider.py`, add `'DmlExecutionProvider'` to the list of providers
the selector tries (before `CPUExecutionProvider`). Record the patch in your
reproducibility notes.

The PyTorch models (layout/OCR/UniMERNet) **cannot** use DirectML without
`torch-directml` + a `MINERU_DEVICE_MODE` override that mineru doesn't ship —
leave them on CPU for this handoff. (CPU bf16 is correct.)

---

## 4. Get the OmniDocBench v1.6 dataset

```powershell
omnidocbench-amd dataset download --version v16 --revision v1.6
```
This lands the 1651 page images + ground-truth manifest. Note the images
directory path (the engine resolves it from the manifest; if your layout differs
from the Linux side, set it via the eval config or engine flags).

---

## 5. Run the adapter (Windows-hip, full 1651 pages)

The adapter implements the platform contract: it writes
`<out-dir>/<image_stem>.md` per page + `<out-dir>/_run_stats.json`. The
dispatcher routes `--backend pipeline` to `pipeline_adapter`, which drives
MinerU's in-process pipeline (`do_parse` over the shared `ModelSingleton`).

```powershell
# Pin to the iGPU if you set up DirectML; otherwise leave default (CPU).
$env:MINERU_MODEL_SOURCE="modelscope"   # or huggingface + HF_ENDPOINT

python adapter\run_adapter.py `
  --img-dir  <OmniDocBench v1.6 page images dir> `
  --out-dir  results\omnidocbench\v1.6\windows-hip\pipeline `
  --platform windows-hip `
  --backend  pipeline
```

Notes:
- `--platform windows-hip` is **required** (the adapter branches on it; never
  infer the platform from the OS).
- Per-page failures are caught and recorded in `_run_stats.json` as `failed`;
  the run continues (a missing page scores zero, a crash scores nothing). Expect
  `ok ≈ 1651, fail ≈ 0, limit_pages: null`.
- On CPU this is slow (hours). You can validate the path first on a 10-page
  subset (copy 10 images into a temp `--img-dir`) before the full run — **do
  this first** to confirm the wiring.
- The Linux side found `examples/demo.png` is a 1×1 placeholder — use **real**
  OmniDocBench images for any smoke.

---

## 6. Score (engine, eval-venv)

Once `_run_stats.json` exists with `limit_pages: null`:

```powershell
# Provision the formula-CDM metric model (needed for Formula_CDM in Overall)
omnidocbench-amd cdm setup --platform windows-hip

# Score
omnidocbench-amd score `
  --platform windows-hip `
  --predictions-dir results\omnidocbench\v1.6\windows-hip\pipeline `
  --version v16 `
  --cdm `
  --run-stats   results\omnidocbench\v1.6\windows-hip\pipeline\_run_stats.json
```
This produces `metric_result.json` with `Edit_dist` (text), `TEDS` (table),
`CDM` (formula), and reading-order `Edit_dist`.

**Compute Overall** = `((1 − Text_EditDist) × 100 + Table_TEDS + Formula_CDM) / 3`.
**PASS** if within **±1.0 pp of 86.47** (i.e. ≥ 85.47).

---

## 7. Publish + land artifacts

```powershell
omnidocbench-amd publish `
  --model-id mineru-pipeline `
  --platform windows-hip `
  --version v16 `
  --cdm `
  --run-stats <..._run_stats.json> `
  --metric-result <...metric_result.json> `
  --results-dir results\omnidocbench\v1.6\windows-hip `
  --git-commit <your commit sha> `
  --adapter-command "python adapter\run_adapter.py --backend pipeline --platform windows-hip" `
  --dataset-revision v1.6
```

Then:
1. **Commit** the engine-assembled artifacts under
   `results/omnidocbench/v1.6/windows-hip/` to the repo (provenance +
   `metric_result.json` + a prediction sample; follow the repo's `.gitignore` /
   LFS policy for bulk `.md`).
2. **Update** `model_card.pipeline.json`: set
   `badge.windows-hip` → `"community"` (self-attested by you) and fill `eval_date`
   + `overall` + `submetrics` + `hardware` (`gpu: "AMD Ryzen AI MAX+ 395 (Strix
   Halo)"`). If a maintainer later Docker-reproduces it on both platforms, it can
   move to `"verified"`.
3. Run `python scripts\check_conformance.py .` (or `omnidocbench-amd conformance .`)
   — must be CONFORMANT.
4. **Open a PR** (or send the `metric_result.json` + commit sha to Claude) so the
   `linux-rocm` owner can update `hub/registry.yaml`.

---

## 8. What Claude (the Linux side) provides vs. what you provide

| Provided by Claude (already in the repo) | Provided by you (colleague) |
|---|---|
| Platform-aware dispatcher (`adapter/run_adapter.py`, branches on `--platform`) | The Windows run + real artifacts |
| `pipeline_adapter.py` (drives mineru `do_parse` + `ModelSingleton`) | `model_card.pipeline.json` Windows badge + result |
| `adapter/setup/00-install-deps.ps1` (Windows provisioning stub) | Any Windows-specific patch (e.g. DirectML EP) + repro notes |
| `eval/configs/omnidocbench_v16.yaml` + the engine contract | The `metric_result.json` / Overall number |
| Badge mechanics + `hub/registry.yaml` update | PR / report-back |

---

## 9. Expected result & escalation

- **Expected:** pipeline Overall **≈ 86.47** (±1.0 pp) on Strix Halo, likely via
  a CPU-bound PyTorch run + (optionally) DirectML ONNX tables. The number is
  device-independent to within tolerance.
- **If Overall drifts > 1.0 pp from 86.47:** don't publish as verified. Likely
  culprits: table HTML scoring (check a few table pages' `.md`), formula CDM
  provisioning, or a page-set/manifest mismatch. Open an issue with your
  `metric_result.json` + the Linux `metric_result.json` (in
  `results/omnidocbench/v1.6/linux-rocm/`) side by side.

---

## 10. Future: the VLM (`mineru2.5`) on Windows

Out of scope for this handoff (separate model card, Plan 2). When the VLM lands
on Linux, its Windows path will be either **transformers-DirectML** (stock
`Qwen2VLForConditionalGeneration` via `torch-directml` / Olive) or
**llama.cpp-GGUF** (HIP/Vulkan build). The `vlm_adapter` will branch on
`--platform` the same way; a second handoff doc will cover it.
