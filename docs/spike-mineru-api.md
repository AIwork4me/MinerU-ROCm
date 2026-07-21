# MinerU 3.4 in-process API spike (Task 4)

Findings verified by running code in the provisioned venv
(`<eval-root>/mineru-rocm-venv`, mineru 3.4.4, torch 2.14.0.dev20260717+rocm7.2)
on GPU 3 of this 4× gfx1100 host. All signatures below are real (read from source
and exercised), not assumed. This doc is the input to Task 5 — it tells
`MineruPipelineRunner.load()`/`.extract()` exactly what to call.

## 0. Provisioning outcome (already run)

- Venv: `<eval-root>/mineru-rocm-venv` (Python **3.11.15**).
- `mineru` 3.4.4 (`Requires-Python: <3.14,>=3.10`). `[all]` on Linux pulls
  `[vllm]` → `vllm==0.19.1` → CUDA torch. After install, `torch.cuda.is_available()`
  was False (came up as `2.10.0+cu128`). The script then overlays the ROCm wheel
  from `https://download.pytorch.org/whl/nightly/rocm7.2` with
  `--force-reinstall --no-deps` (the `--force-reinstall` is required — pip sees the
  same version string `2.10.0` on both indexes and otherwise considers it
  satisfied). Result: `torch 2.14.0.dev20260717+rocm7.2`, `cuda.is_available()==True`,
  `hip==7.2.53211`. **Use this exact flag set if re-running.**
- Weights downloaded via `mineru-models-download -s huggingface -m pipeline` with
  `HF_ENDPOINT=<hf-mirror>`. Landed at
  `/root/.cache/huggingface/models--opendatalab--PDF-Extract-Kit-1.0/snapshots/ed6b654c018d742e65a17671e379c5e6ecc87ec9/`.
  Config written to `/root/mineru.json` (points at that snapshot).
  Sub-models present: `Layout/PP-DocLayoutV2`, `MFR/unimernet_hf_small_2503`,
  `OCR/paddleocr_torch` (PP-OCRv6 small det+rec safetensors),
  `TabRec/SlanetPlus/slanet-plus.onnx`, `TabRec/UnetStructure/unet.onnx`,
  `TabCls/paddle_table_cls/PP-LCNet_x1_0_table_cls.onnx`,
  `MFR/pp_formulanet_plus_m` (formula fallback; not used when `formula_enable=true`
  with unimernet path).
- Coexists with the platform eval-venv `<eval-root>/omnidocbench-rocm-venv`
  (which is actually **Python 3.12.3**, not 3.11) — separate venvs, separate
  site-packages, no collision.

## 1. Python requirement

- `Requires-Python: <3.14,>=3.10`. We picked **3.11** (`/usr/bin/python3.11`,
  version 3.11.15). 3.12 also works; 3.11 was chosen to match the rest of the
  platform where possible. The eval-venv is 3.12 — both coexist.

## 2. CLI image input + output layout

Invoked:
```
HIP_VISIBLE_DEVICES=3 MINERU_DEVICE_MODE=cuda HF_ENDPOINT=<hf-mirror> \
  mineru -p /tmp/mineru-spike-inputs -o /tmp/mineru-spike -b pipeline
```
on a 3-image dir (`img3.jpg`, `newspaper_…_1.jpg`, `page-….png`).

**Behavior:** the CLI (`mineru.cli.client:main`) **does not call `do_parse`
directly**. It spawns a temporary local FastAPI server (`LocalAPIServer`) and
submits the directory as ONE batch via HTTP. For 3 single-page images the planner
produces 1 task with all 3 documents, and the server-side pipeline runs
`doc_analyze_streaming` once over all 3 (single model load, `total_batches=1`,
window_size=64). Log line:
> `Pipeline processing-window multi-file run. doc_count=3, total_pages=3, window_size=64, total_batches=1`

**Output layout** (verified with `find`):
```
<out_dir>/
  <stem>/
    auto/
      <stem>.md                 <-- Markdown (this is the deliverable)
      <stem>_content_list.json
      <stem>_content_list_v2.json
      <stem>_layout.pdf
      <stem>_middle.json
      <stem>_model.json
      <stem>_origin.pdf
      <stem>_span.pdf
      images/
        <sha256>.jpg            <-- cropped figures/tables, content-addressed
```
So each input `<stem>.jpg` produces **exactly one `<out_dir>/<stem>/auto/<stem>.md`**.
`auto` is the `parse_method` subdir (set by `-m auto`, the default).
Naming is 1:1 with the input stem (the CLI also runs `uniquify_task_stems` to
dedupe collisions, suffixing `_2`, `_3`, …).

`examples/demo.png` is a 1×1 px placeholder — do **not** use it for spikes; use
real OmniDocBench images (e.g. `<eval-root>/OmniDocBench_data/images/`).

## 3. In-process API (PREFERRED for 1651 pages)

The CLI spawns a subprocess + HTTP server per invocation — far too heavy for
1651 pages. The in-process entry is `do_parse` in `mineru.cli.common`. The CLI's
server-side handler calls this same function. Verified working end-to-end on the
3-image dir (identical Markdown output to the CLI run).

### Exact import + signature

```python
from mineru.cli.common import do_parse, read_fn
```

`read_fn(path) -> bytes`: reads an image/pdf file and, for images, wraps it into
PDF bytes via `images_bytes_to_pdf_bytes` (the pipeline renders PDFs, not raw
images). Returns `bytes` that `do_parse` consumes.

`do_parse(...)` — full signature (defaults shown):
```python
do_parse(
    output_dir,                          # str: where <stem>/auto/ lands
    pdf_file_names: list[str],           # list of stems (NO extension, NO path)
    pdf_bytes_list: list[bytes],         # PDF bytes per doc (use read_fn)
    p_lang_list: list[str],              # OCR lang per doc, e.g. ["ch"]*N
    backend="pipeline",                  # MUST be "pipeline" for this adapter
    parse_method="auto",                 # "auto" | "ocr" | "txt"
    formula_enable=True,
    table_enable=True,
    server_url=None,
    f_draw_layout_bbox=True,             # set False (skip layout.pdf)
    f_draw_span_bbox=True,               # set False
    f_dump_md=True,                      # MUST be True — this is our deliverable
    f_dump_middle_json=True,             # set False unless debugging
    f_dump_model_output=True,            # set False unless debugging
    f_dump_orig_pdf=True,                # set False
    f_dump_content_list=True,            # set False
    f_make_md_mode=MakeMode.MM_MD,       # default is fine
    start_page_id=0,
    end_page_id=None,
    image_analysis=True,
    client_side_output_generation=False,
    effort=DEFAULT_HYBRID_EFFORT,        # hybrid-only
    **kwargs,
) -> None                                # writes files; returns None
```

`do_parse` calls `_process_pipeline` → `pipeline_doc_analyze_streaming` (in
`mineru.backend.pipeline.pipeline_analyze`), which loads models **once via
`ModelSingleton`** (a module-level singleton keyed by `(lang, formula_enable,
table_enable)` — persists across calls within the same process), then batches all
docs into windows of 64 pages (`get_processing_window_size(default=64)`).

### How Markdown comes back per image

`do_parse` does NOT return Markdown — it writes files to `output_dir`. After the
call, read `<output_dir>/<stem>/auto/<stem>.md` from disk. The `<stem>` you pass
in `pdf_file_names` is exactly the subdir + filename stem.

### CRITICAL — `if __name__ == "__main__":` guard

`mineru.utils.pdf_image_tools` renders pages via a `ProcessPoolExecutor` with
`multiprocessing.get_context("spawn")`. Spawn re-imports the entry module in each
worker. **If the `do_parse(...)` call is at module top level (no `main()` guard),
spawn workers re-execute it on import and the pool dies with
`BrokenProcessPool: A process in the process pool was terminated abruptly`.**
The fix is the standard guard:

```python
if __name__ == "__main__":
    ...do_parse(...)...
```
Inside `MineruPipelineRunner.extract` this is not an issue (the worker module is
imported, not run as `__main__`). The guard matters only for standalone scripts.

## 4. Decision: in-process `do_parse`, single-call batching

**Chosen path: in-process `do_parse`, called ONCE per batch of N images** (not
per-image, not CLI subprocess). Reasons:
- Avoids CLI's per-invocation FastAPI server spawn + HTTP overhead.
- Single `ModelSingleton` load amortized across all 1651 pages.
- `doc_analyze_streaming` batches up to 64 pages per inference window → much
  better GPU utilization than 1-page-at-a-time.

### Caveat for the dispatcher contract

`adapter/run_adapter.py` currently loops `sub.infer_page(img, platform, cfg)`
**per image** and expects each call to return the Markdown string for that image
(written by the dispatcher as `out_dir/<stem>.md`). Two ways to honour this
contract while keeping the single-load benefit:

**Option A (simplest, fits the existing contract) — per-image call, shared
singleton.** Because `ModelSingleton` is module-level, the first `infer_page`
loads the model and every subsequent call reuses it. We lose the 64-page window
batching (each call is its own 1-page window) but model load is still amortized.
This requires NO change to `run_adapter.py`.

**Option B (optimal, ~better throughput) — override the loop to batch.** Change
`run_adapter.py` (or add a `batch=True` path in `pipeline_adapter.py`) to gather
all images first, call `do_parse` once, then read the per-stem `.md` files.
Requires dispatcher changes; out of scope for Task 5 unless we want the speedup.

**Recommendation for Task 5: Option A.** It honours the existing
`infer_page(img, ...) -> str` contract, keeps the singleton load, and is a
5-line `load()`/`extract()` fill. Option B can be a follow-up optimization.

### Concrete code sketch for Task 5 (Option A)

```python
# adapter/pipeline_adapter.py
import os
from pathlib import Path

# These must be set BEFORE mineru imports anything that reads device mode.
os.environ.setdefault("MINERU_DEVICE_MODE", "cuda")  # HIP_VISIBLE_DEVICES set by the launcher
os.environ.setdefault("HF_ENDPOINT", "<hf-mirror>")

_runner = None

def infer_page(img: Path, platform: str, cfg: dict) -> str:
    global _runner
    if _runner is None:
        _runner = MineruPipelineRunner(platform=platform, cfg=cfg)
        _runner.load()
    return _runner.extract(img)


class MineruPipelineRunner:
    def __init__(self, platform: str, cfg: dict):
        self.platform = platform
        self.cfg = cfg
        self._lang = cfg.get("lang", "ch")
        self._tmp_out = Path(cfg.get("mineru_out_root", "/tmp/mineru-adapter-out"))
        self._call_idx = 0

    def load(self):
        # Import inside load() so the env vars above are set first.
        from mineru.backend.pipeline.pipeline_analyze import ModelSingleton
        # Force model init now (puts layout/OCR/UniMERNet on cuda:0).
        ModelSingleton().get_model(
            lang=self._lang, formula_enable=True, table_enable=True)

    def extract(self, img: Path) -> str:
        from mineru.cli.common import do_parse, read_fn
        stem = img.stem
        out_dir = self._tmp_out / f"run-{os.getpid()}"
        out_dir.mkdir(parents=True, exist_ok=True)
        do_parse(
            output_dir=str(out_dir),
            pdf_file_names=[stem],
            pdf_bytes_list=[read_fn(img)],
            p_lang_list=[self._lang],
            backend="pipeline",
            parse_method="auto",
            formula_enable=True, table_enable=True,
            f_draw_layout_bbox=False, f_draw_span_bbox=False,
            f_dump_md=True,                    # the deliverable
            f_dump_middle_json=False, f_dump_model_output=False,
            f_dump_orig_pdf=False, f_dump_content_list=False,
        )
        md_path = out_dir / stem / "auto" / f"{stem}.md"
        return md_path.read_text(encoding="utf-8")
```

Notes for Task 5:
- `read_fn` wraps the image into PDF bytes; `do_parse` renders it back to pixels
  via pdfium in a spawn pool. Keep `infer_page` importable (no top-level
  `do_parse(...)` call) so spawn workers don't re-enter it.
- `ModelSingleton` survives across `extract()` calls — model load is paid once
  in `load()`, not per page.
- Output dir per-run (`run-<pid>`) so parallel adapter processes don't collide.
  The dispatcher writes the final `out_dir/<stem>.md`; mineru's intermediate tree
  can be cleaned up later if disk is a concern (each page is a few hundred KB).
- For 1651 pages, consider Option B (batch) as a follow-up: collect all stems +
  pdf_bytes, one `do_parse` call, then read all `.md` files. The contract change
  is in `run_adapter.py`, not here.

## 5. GPU placement (verified)

With `HIP_VISIBLE_DEVICES=3 MINERU_DEVICE_MODE=cuda`, on the loaded
`MineruPipelineModel`:

| Sub-model attr | Class | Device | Evidence |
|---|---|---|---|
| `layout_model` | `PPDocLayoutV2LayoutModel` | **cuda:0** | `params[0].device == cuda:0`, n_params=557 |
| `mfr_model` | `UnimernetModel` (UniMERNet-small) | **cuda:0** | `params[0].device == cuda:0`, n_params=736 |
| `ocr_model.text_detector.net` | `TextDetector` | **cuda:0** | `.net.parameters()[0].device == cuda:0` |
| `ocr_model.text_recognizer.net` | `TextRecognizer` | **cuda:0** | `.net.parameters()[0].device == cuda:0` |
| `table_cls_model.sess` | `PaddleTableClsModel` (PP-LCNet ONNX) | **CPU** | `get_providers()==['CPUExecutionProvider']` |
| `wired_table_model` | `UnetTableModel` (UNet ONNX) | **CPU** | ONNX session, CPUExecutionProvider |
| `wireless_table_model` | `PaddleTableModel` (SLANet-Plus ONNX) | **CPU** | ONNX session, CPUExecutionProvider |
| `img_orientation_cls_model` | `MineruTableOrientationClsModel` | CPU | ONNX |

Why ONNX stays on CPU: `mineru/model/table/rec/onnxruntime_provider.py` only
enables `CUDAExecutionProvider` when `get_device()=="cuda"` **AND** onnxruntime
 advertises `CUDAExecutionProvider`. The installed `onnxruntime==1.27.0` exposes
only `['AzureExecutionProvider', 'CPUExecutionProvider']` — no CUDA/ROCm/DML EP —
so the table models fall back to `CPUExecutionProvider`. This matches the brief's
expectation. (There is no ROCm ONNX-EP wheel on PyPI; do not try to "fix" this —
the table models are tiny and CPU is fine.)

**rocm-smi evidence during a 3-page run** (sampled at 1s):
```
GPU[3]: GPU use (%): 14
GPU[3]: GPU use (%): 43
GPU[3]: GPU use (%): 17
GPU[3]: GPU use (%): 31
GPU[3]: GPU use (%): 40
```
GPU 3 is the active device (peaks at 43% during layout/OCR/formula inference);
the ONNX table pass adds no GPU load (runs on host CPU), as expected.

## Appendix: environment requirements summary

For any mineru run (CLI or in-process) on this host:
```bash
export HIP_VISIBLE_DEVICES=3
export MINERU_DEVICE_MODE=cuda
export HF_ENDPOINT=<hf-mirror>     # only needed for weight download
source <eval-root>/mineru-rocm-venv/bin/activate
```
The venv already has the ROCm torch overlay applied. `MINERU_FORMULA_CH_SUPPORT`
must be left unset/false (it pulls in PaddlePaddle, which we do NOT want).
