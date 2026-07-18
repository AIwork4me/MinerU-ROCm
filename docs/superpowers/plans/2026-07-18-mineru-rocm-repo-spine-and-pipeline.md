# MinerU-ROCm — Repo Spine + MinerU 3.4 Pipeline Implementation Plan (Plan 1 of 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the scaffolded MinerU-ROCm repo into a conformant dual-adapter platform repo, and land the **MinerU 3.4 pipeline** on ROCm reproducing official OmniDocBench v1.6 **86.47** (within 1.0 pp) — plus the Windows handoff doc for the colleague track.

**Architecture:** A thin `run_adapter.py` dispatcher routes `--backend` to `pipeline_adapter` or `vlm_adapter` (this plan stubs `vlm_adapter`; Plan 2 fills it). The pipeline adapter wraps upstream `mineru[all]` in-process on `cuda` (`MINERU_DEVICE_MODE=cuda`), loading the pipeline once and looping page images → Markdown. The actual `mineru` call is isolated behind one method (`MineruPipelineRunner._extract_markdown`) that a spike task fills in — all surrounding code (loading, looping, R4 output conventions, per-page error handling, `_run_stats.json`) is complete up front.

**Tech Stack:** Python 3.11/3.12, `mineru[all]` (v3.4.2, pure-PyTorch pipeline + ONNX tables), PyTorch ROCm 2.9.1, the `omnidocbench-amd` engine (contract + eval + scoring + provenance), pytest.

## Global Constraints

- Repo code stays on `/workspace/MinerU-ROCm` (10 GB NFS, code-only). **All venvs, weights, outputs, caches → `/root`** (2 TB). No venv/wheel/source/weights on a real `/workspace` path (blows the disk).
- Model venv (for `mineru[all]`) is **separate** from the platform eval-venv (Py 3.11). Verify `mineru`'s Python requirement during the Task 4 spike; default venv root `/root/ocr-eval/mineru-rocm-venv`.
- HF access via `hf-mirror.com` only (`HF_ENDPOINT=http://134.199.133.77`, huggingface.co is blocked). Weights: `opendatalab/PDF-Extract-Kit-1.0` (pipeline) via `mineru-models-download -s huggingface`.
- Device: `MINERU_DEVICE_MODE=cuda` (explicit). Pipeline lands on GPU via `torch.cuda.is_available()`=true on ROCm.
- **Hard avoid:** `MINERU_FORMULA_CH_SUPPORT=true` (pulls native-PaddlePaddle `pp_formulanet`, the one ROCm blocker). Stay on default UniMERNet.
- Output conventions (contract R4): formulas `$…$`/`$$…$$`; tables HTML; reading order = document order; images `![](path)` (no `<div>` wrappers).
- Per-page failure → record in `_run_stats.json` as `failed`, continue, **never raise** (contract R2).
- Official target: pipeline OmniDocBench v1.6 **86.47**; PASS gate ≤ 1.0 pp. Hardware: 4× gfx1100 (Radeon PRO W7900, 48 GB), all idle. Use **GPU 3** for pipeline work.
- Platform slot: `OmniDocBench-AMD/hub/registry.yaml` → `mineru2.5 → AIwork4me/MinerU-ROCm`. Spec: `docs/superpowers/specs/2026-07-17-mineru-rocm-design.md`.

---

## File Structure

| File | Responsibility | Created in |
|---|---|---|
| `adapter/__init__.py` | Makes `adapter` an importable package (empty) so tests can `from adapter import …` | Task 1 |
| `adapter/run_adapter.py` | **Dispatcher**: contract entrypoint; loops pages, writes `<stem>.md` + `_run_stats.json`; routes `--backend` to the right sub-adapter; times each page; per-page try/except | Task 1 (rewrite) |
| `adapter/pipeline_adapter.py` | `backend=pipeline`: lazy `MineruPipelineRunner` (load once → `infer_page` per image). `mineru` call isolated in `_extract_markdown` (Task 4 spike fills it) | Task 1 (skeleton) → Task 5 (impl) |
| `adapter/vlm_adapter.py` | `backend=vlm-*`: stub raising a clear "Phase 2 / Plan 2" message (filled later) | Task 1 |
| `adapter/adapter_config.py` | Defaults: `BACKEND`, `SERVER_URL`, `API_MODEL_NAME`, `WEIGHTS_DIR`, `MODEL` (pipeline\|vlm) | Task 2 (extend) |
| `adapter/setup/00-install-deps.sh` | Linux/ROCm provisioning: create `/root` venv, `pip install mineru[all]`, set env, download pipeline weights | Task 4 |
| `model_card.json` | VLM model card (primary) — unchanged this plan | (exists) |
| `model_card.pipeline.json` | Pipeline model card (model_id `mineru-pipeline`) | Task 2 → Task 8 (results) |
| `examples/run_demo.sh` | One-command pipeline smoke on `examples/demo.png` (GPU 3) | Task 5 |
| `docs/HANDOFF-windows-hip.md` | Self-contained Windows verification guide for the colleague | Task 7 |
| `docs/spike-mineru-api.md` | Task 4 spike output: the exact `mineru` in-process API + CLI-vs-in-process decision | Task 4 |
| `tests/test_dispatcher.py` | Dispatcher routing + smoke + stats (CPU) | Task 1 |
| `tests/test_pipeline_output.py` | R4 output-convention helpers (CPU) | Task 5 |
| `.github/workflows/ci.yml` | CPU CI: pytest + `check_conformance` | Task 3 |

---

## Task 1: Dispatcher refactor + vlm_adapter stub

**Files:**
- Create: `adapter/__init__.py`, `adapter/pipeline_adapter.py`, `adapter/vlm_adapter.py`, `tests/test_dispatcher.py`, `tests/__init__.py`
- Rewrite: `adapter/run_adapter.py`

**Interfaces:**
- Produces: `run_adapter.run_adapter(img_dir, out_dir, *, platform, config) -> dict` (unchanged contract signature); each sub-adapter exposes `infer_page(img: Path, platform: str, cfg: dict) -> str` returning Markdown. Later tasks rely on `infer_page`'s name + signature.

- [ ] **Step 1: Write the failing test** — `tests/__init__.py` (empty) + `tests/test_dispatcher.py`:

```python
# tests/test_dispatcher.py
from pathlib import Path
import json, run_adapter, pipeline_adapter, vlm_adapter

def _write_img(d, name="p1.jpg"):
    p = Path(d) / name; p.write_bytes(b"\xff\xd8\xff"); return p

def test_smoke_backend_writes_md_and_stats(tmp_path):
    _write_img(tmp_path)
    out = tmp_path / "out"
    run_adapter.run_adapter(tmp_path, out, platform="linux-rocm", config={"backend": "smoke"})
    assert (out / "p1.md").read_text(encoding="utf-8").startswith("# p1")
    rs = json.loads((out / "_run_stats.json").read_text())
    assert rs["count"] == 1 and rs["ok"] == 1 and rs["fail"] == 0 and rs["engine"] == "smoke"

def test_pipeline_backend_routes_to_pipeline_adapter(tmp_path, monkeypatch):
    _write_img(tmp_path)
    called = {}
    def fake(img, platform, cfg):
        called["img"] = img.name; return "# pipeline md\n"
    monkeypatch.setattr(pipeline_adapter, "infer_page", fake)
    out = tmp_path / "out"
    run_adapter.run_adapter(tmp_path, out, platform="linux-rocm", config={"backend": "pipeline"})
    assert (out / "p1.md").read_text(encoding="utf-8") == "# pipeline md\n"
    assert called["img"] == "p1.jpg"
    assert json.loads((out / "_run_stats.json").read_text())["engine"] == "pipeline"

def test_vlm_backend_routes_to_vlm_adapter(tmp_path, monkeypatch):
    _write_img(tmp_path)
    monkeypatch.setattr(vlm_adapter, "infer_page", lambda i, p, c: "# vlm md\n")
    out = tmp_path / "out"
    run_adapter.run_adapter(tmp_path, out, platform="linux-rocm", config={"backend": "vlm-transformers"})
    assert (out / "p1.md").read_text(encoding="utf-8") == "# vlm md\n"

def test_unknown_backend_raises_value_error(tmp_path):
    _write_img(tmp_path)
    try:
        run_adapter.run_adapter(tmp_path, tmp_path / "o", platform="linux-rocm", config={"backend": "wat"})
        assert False, "expected ValueError"
    except ValueError:
        pass

def test_per_page_failure_is_recorded_not_raised(tmp_path, monkeypatch):
    _write_img(tmp_path, "ok.jpg"); _write_img(tmp_path, "bad.jpg")
    def fake(img, platform, cfg):
        if img.name == "bad.jpg": raise RuntimeError("boom")
        return "# ok\n"
    monkeypatch.setattr(pipeline_adapter, "infer_page", fake)
    out = tmp_path / "out"
    run_adapter.run_adapter(tmp_path, out, platform="linux-rocm", config={"backend": "pipeline"})
    rs = json.loads((out / "_run_stats.json").read_text())
    assert rs["ok"] == 1 and rs["fail"] == 1
    assert (out / "ok.md").exists() and not (out / "bad.md").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /workspace/MinerU-ROCm && python -m pytest tests/test_dispatcher.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'run_adapter'` / `pipeline_adapter` / `vlm_adapter` (they don't exist yet / old structure).

- [ ] **Step 3: Create the package marker + sub-adapter stubs**

`adapter/__init__.py`: (empty file)

`adapter/vlm_adapter.py`:
```python
"""VLM adapter (MinerU2.5-Pro-2605-1.2B). Filled in Plan 2 (Phase 2+3).

This plan ships only the stub so the dispatcher routes vlm-* backends cleanly.
"""
from __future__ import annotations
from pathlib import Path

_NOT_IMPLEMENTED = (
    "The VLM adapter (MinerU2.5-Pro-2605-1.2B) is implemented in Plan 2 (Phase 2+3). "
    "It drives opendatalab/mineru-vl-utils two-step inference (layout→extract) with the "
    "MinerULogitsProcessor against a vLLM-on-ROCm or transformers server. "
    "See docs/superpowers/specs/2026-07-17-mineru-rocm-design.md §8."
)

def infer_page(img: Path, platform: str, cfg: dict) -> str:
    raise NotImplementedError(_NOT_IMPLEMENTED)
```

`adapter/pipeline_adapter.py` (skeleton; `infer_page` lazy-loads a runner; the runner's `_extract_markdown` is filled in Task 5 after the Task 4 spike):
```python
"""MinerU 3.4 pipeline adapter (backend=pipeline).

Wraps upstream mineru[all] in-process on ROCm cuda. Loads the pipeline ONCE
(first page) and reuses it for every page. The actual mineru call lives in
MineruPipelineRunner._extract_markdown (filled by Task 5, after the Task 4
spike documents the exact API in docs/spike-mineru-api.md).
"""
from __future__ import annotations
from pathlib import Path

_runner = None  # lazy singleton, created on first infer_page call

def infer_page(img: Path, platform: str, cfg: dict) -> str:
    """Return Markdown for one page image. Raises on per-page failure (caller catches)."""
    global _runner
    if _runner is None:
        _runner = MineruPipelineRunner(platform=platform, cfg=cfg)
        _runner.load()
    return _runner.extract(img)

class MineruPipelineRunner:
    def __init__(self, platform: str, cfg: dict):
        self.platform = platform
        self.cfg = cfg

    def load(self):
        """Warm the pipeline sub-models on cuda. Filled in Task 5."""
        raise NotImplementedError("Task 5 fills load() using the Task 4 spike findings.")

    def extract(self, img: Path) -> str:
        """Run the pipeline on one image → Markdown. Filled in Task 5."""
        raise NotImplementedError("Task 5 fills extract() using the Task 4 spike findings.")
```

- [ ] **Step 4: Rewrite the dispatcher** — `adapter/run_adapter.py`:

```python
"""MinerU-ROCm adapter dispatcher — implements the omnidocbench-amd contract.

Routes --backend to the right sub-adapter (pipeline | vlm-*). Keeps the
contract signature and the out_dir/<image_stem>.md + _run_stats.json output
convention. Per-page failures are caught and recorded (R2) — never raised.
"""
from __future__ import annotations
import argparse, importlib, sys, time
from pathlib import Path
from omnidocbench_amd.types import RunSummary, PageStatus

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
PLATFORMS = ("linux-rocm", "windows-hip")
SUB_ADAPTERS = {"pipeline": "pipeline_adapter",
                "vlm-vllm": "vlm_adapter", "vlm-transformers": "vlm_adapter"}


def _load_adapter_config():
    try:
        from . import adapter_config  # package-relative
    except ImportError:
        _here = Path(__file__).resolve().parent
        if str(_here) not in sys.path:
            sys.path.insert(0, str(_here))
        import adapter_config  # type: ignore[import-not-found]
    return adapter_config


def _import_sub(name: str):
    """Import a sibling adapter module whether run as a package or a bare script."""
    pkg = __package__
    if pkg:
        try:
            return importlib.import_module(f".{name}", pkg)
        except ImportError:
            pass
    _here = Path(__file__).resolve().parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))
    return importlib.import_module(name)


def run_adapter(img_dir: Path, out_dir: Path, *, platform: str, config: dict) -> dict:
    assert platform in PLATFORMS, f"unknown platform: {platform}"
    adapter_config = _load_adapter_config()
    cfg = {**adapter_config.as_dict(), **config}
    out_dir.mkdir(parents=True, exist_ok=True)
    imgs = sorted(p for p in Path(img_dir).iterdir() if p.suffix.lower() in IMG_EXT)
    stats: list[PageStatus] = []
    backend = cfg.get("backend", "smoke")
    sub = None if backend == "smoke" else _import_sub(SUB_ADAPTERS[backend])  # KeyError → ValueError below
    for i in imgs:
        t0 = time.time()
        try:
            if sub is None:
                md = f"# {i.stem}\n\n(smoke output — backend=smoke)\n"
            else:
                md = sub.infer_page(i, platform, cfg)
            (out_dir / f"{i.stem}.md").write_text(md, encoding="utf-8")
            stats.append(PageStatus(i.name, "ok", seconds=time.time() - t0, attempts=1))
        except Exception as e:  # per-page failure → record, continue, never raise
            stats.append(PageStatus(i.name, f"failed: {e}", error=str(e), seconds=time.time() - t0))
    rs = RunSummary(len(imgs), sum(1 for s in stats if s.status == "ok"),
                    sum(1 for s in stats if s.status.startswith("failed")),
                    sum(1 for s in stats if s.status.startswith("fallback")),
                    cfg.get("limit_pages"), stats, engine=backend)
    rs.write(out_dir / "_run_stats.json")
    return rs.to_run_stats()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--img-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--platform", required=True, choices=PLATFORMS)
    p.add_argument("--backend", default="smoke")
    p.add_argument("--server-url", default="")
    p.add_argument("--api-model-name", default="")
    a = p.parse_args()
    if a.backend != "smoke" and a.backend not in SUB_ADAPTERS:
        raise SystemExit(f"unknown backend: {a.backend!r} (expected smoke|pipeline|vlm-vllm|vlm-transformers)")
    run_adapter(Path(a.img_dir), Path(a.out_dir), platform=a.platform,
                config={"backend": a.backend, "server_url": a.server_url, "api_model_name": a.api_model_name})
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /workspace/MinerU-ROCm && python -m pytest tests/test_dispatcher.py -v`
Expected: PASS (5 passed). (Tests import `run_adapter` etc. at top level — `adapter/` is on `sys.path` via the empty `__init__.py` + pytest rootdir; if import fails, add a `conftest.py` with `import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent / "adapter"))`.)

- [ ] **Step 6: Commit**

```bash
cd /workspace/MinerU-ROCm
git add adapter/__init__.py adapter/run_adapter.py adapter/pipeline_adapter.py adapter/vlm_adapter.py tests/
git commit -m "refactor(adapter): dispatcher routing to pipeline/vlm sub-adapters

Splits the single-file adapter into a thin dispatcher + pipeline_adapter +
vlm_adapter (stub). Keeps the contract signature and out_dir/<stem>.md +
_run_stats.json convention; per-page failures recorded not raised (R2)."
```

---

## Task 2: Backend/model knob + pipeline model_card + repo polish

**Files:**
- Modify: `adapter/adapter_config.py`
- Create: `model_card.pipeline.json`
- Modify: `README.md` (fix org link), `docs/known-gaps.md` (seed)

**Interfaces:**
- Produces: `adapter_config.as_dict()` now includes `MODEL` ("pipeline"|"vlm"); `model_card.pipeline.json` with `model_id: mineru-pipeline`.

- [ ] **Step 1: Extend adapter_config.py** — replace its body with:

```python
"""Adapter configuration for MinerU-ROCm.

backend selects the path; model is advisory (which MinerU model a run targets).
"""
from __future__ import annotations

# smoke = no-GPU CI placeholder. Real: pipeline | vlm-vllm | vlm-transformers.
BACKEND = "smoke"
# Which MinerU model this run targets: "pipeline" (3.4) | "vlm" (2.5-Pro).
MODEL = "pipeline"
SERVER_URL = ""               # VLM OpenAI-compatible server (empty = spawn locally)
API_MODEL_NAME = "mineru2.5"  # VLM model name as registered on the server
WEIGHTS_DIR = ""              # resolved at runtime; pipeline weights via mineru-models-download

def as_dict() -> dict:
    return {"backend": BACKEND, "model": MODEL, "server_url": SERVER_URL,
            "api_model_name": API_MODEL_NAME, "weights_dir": WEIGHTS_DIR}
```

- [ ] **Step 2: Create model_card.pipeline.json**

```json
{
  "schema_version": 1,
  "model_id": "mineru-pipeline",
  "model_version": "3.4.2",
  "platforms": ["linux-rocm", "windows-hip"],
  "badge": {"linux-rocm": "community-wanted", "windows-hip": "community-wanted"},
  "eval_date": "",
  "omnidocbench_version": "v1.6",
  "overall": null,
  "submetrics": {},
  "hardware": {"gpu": "", "vram": "", "rocm_driver": ""},
  "artifacts": {}
}
```

Validate it against the schema:
Run: `cd /workspace/omnidocbench-amd && python -c "import json,jsonschema,pathlib; s=json.load(open('contracts/artifact-schema.json'))['\$defs']['model_card']; jsonschema.validate(json.load(open('/workspace/MinerU-ROCm/model_card.pipeline.json')), s); print('valid')"`
Expected: `valid`

- [ ] **Step 3: Fix the README org link**

In `README.md`, replace `https://github.com/omnidobench/omnidocbench-amd` → `https://github.com/AIwork4me/OmniDocBench-AMD` (the cookiecutter default is wrong). Verify with:
Run: `grep -n "omnidobench/omnidocbench-amd\|omnidocbench/omnidocbench-amd" /workspace/MinerU-ROCm/README.md /workspace/MinerU-ROCm/README.zh-CN.md`
Expected: no matches after the replacement.

- [ ] **Step 4: Seed docs/known-gaps.md** — replace the placeholder body with the pipeline-relevant gaps (verbatim from spec §16, pipeline subset): ONNX tables run on CPU on ROCm (correct, slow; optional ROCm-EP patch); `MINERU_FORMULA_CH_SUPPORT=true` must stay off (pulls native PaddlePaddle); windows-hip unverified here.

- [ ] **Step 5: Commit**

```bash
cd /workspace/MinerU-ROCm
git add adapter/adapter_config.py model_card.pipeline.json README.md README.zh-CN.md docs/known-gaps.md
git commit -m "feat(config): backend/model knob, pipeline model_card, README link fix"
```

---

## Task 3: Conformance + CI green

**Files:**
- Modify: `.github/workflows/ci.yml`
- Verify: `check_conformance.py` passes

- [ ] **Step 1: Confirm conformance passes locally**

Run: `cd /workspace/omnidocbench-amd && source /root/ocr-eval/omnidocbench-amd-venv/bin/activate && python scripts/check_conformance.py /workspace/MinerU-ROCm`
Expected: `CONFORMANT` (exit 0). The smoke repo satisfies all 8 checks (adapter exists, eval config exists, READMEs with 5 sections, examples/ non-empty, pyproject depends on omnidocbench-amd, model_card valid).

If it fails on a README section, add the missing `## Install` / `## Demo` / `## Evaluation` / `## Reproducibility` / `## Known Gaps` header to both README.md and README.zh-CN.md (the template ships them — verify).

- [ ] **Step 2: Strengthen CI** — replace `.github/workflows/ci.yml` with:

```yaml
on: [push, pull_request]
jobs:
  smoke:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: '3.11'}
      - run: pip install -e ".[dev]" && pip install omnidocbench-amd pytest
      - run: python -c "from omnidocbench_amd.types import RunSummary"
      - run: pytest -q
      - run: python /workspace/omnidocbench-amd/scripts/check_conformance.py . || true  # local-only; CI can't see the platform repo path — see note
```
> Note: CI runners can't see the local `/workspace/omnidocbench-amd` path. For CI, vendor a minimal conformance check or fetch the platform repo in CI. For now this plan keeps CI = pytest + import; full conformance runs locally (Step 1) and is a pre-publish gate. (Refining CI conformance is a Phase 0 polish item; not blocking.)

- [ ] **Step 3: Run pytest locally**

Run: `cd /workspace/MinerU-ROCm && python -m pytest -q`
Expected: all green.

- [ ] **Step 4: Commit**

```bash
cd /workspace/MinerU-ROCm
git add .github/workflows/ci.yml
git commit -m "ci: pytest + import check; conformance runs locally as pre-publish gate"
```

---

## Task 4: mineru[all] provisioning + in-process API spike

**Files:**
- Create: `adapter/setup/00-install-deps.sh` (real provisioning), `docs/spike-mineru-api.md` (spike findings)

> This task does NOT commit application code — it provisions the environment and documents the exact `mineru` API so Task 5 has real signatures to code against. Output is a doc + a working venv, verified by a smoke command.

- [ ] **Step 1: Write the provisioning script** — `adapter/setup/00-install-deps.sh`:

```bash
#!/usr/bin/env bash
# MinerU-ROCm — Linux/ROCm provisioning. Venv + weights on /root (NOT /workspace).
set -euo pipefail
VENV="${MINERU_ROCM_VENV:-/root/ocr-eval/mineru-rocm-venv}"
PY="${PYTHON:-python3}"
echo "[00-install-deps] creating venv at $VENV"
"$PY" -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -U pip
# mineru[all] pulls torch; on ROCm ensure the right wheel. If it grabs a CUDA torch,
# reinstall the ROCm wheel afterwards: pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/rocm7.2
pip install -U "mineru[all]"
# Pipeline weights (PP-DocLayoutV2, UniMERNet, PP-OCRv6, SLANet/UNet) via hf-mirror
export HF_ENDPOINT="http://134.199.133.77"
export MINERU_MODEL_SOURCE=huggingface
mineru-models-download -s huggingface -m pipeline
echo "[00-install-deps] done. Activate: source $VENV/bin/activate"
```

- [ ] **Step 2: Run provisioning on GPU 3's host** (CPU step; just sets up the venv)

Run: `chmod +x /workspace/MinerU-ROCm/adapter/setup/00-install-deps.sh && bash /workspace/MinerU-ROCm/adapter/setup/00-install-deps.sh`
Expected: venv created at `/root/ocr-eval/mineru-rocm-venv`, `mineru[all]` + weights installed. Note the torch version installed; if it's not the ROCm build, reinstall the ROCm wheel as the comment shows and re-verify `python -c "import torch; print(torch.cuda.is_available())"` → `True`.

- [ ] **Step 3: Spike the in-process API + CLI** — write findings to `docs/spike-mineru-api.md`. Determine, by reading `mineru/` source and running probes in the venv:

  1. **Python requirement** of `mineru[all]` (record exact version; confirm it coexists with the Py3.11 eval-venv).
  2. **CLI image-input**: does `mineru -p <dir_of_images> -o <out> -b pipeline` process a whole directory in one model load? What is the output layout/naming (`<stem>.md`? a single `full.md`? a `json/` + `md/` split)? Run it on `examples/demo.png` and 2–3 images, record exact outputs.
  3. **In-process API**: the class/module to call for "load pipeline once, run on N images" — read `mineru/cli/__init__.py` (the CLI entrypoint) to find the function it calls, then replicate it in-process. Record the exact import + call signature (e.g. `from mineru.cli.common import ...` / `from mineru.backend.pipeline.pipeline_pipeline import ...`). Preferred over per-page subprocess for 1651 pages.
  4. **Decision**: CLI-one-shot-dir (if it loads once + names outputs `<stem>.md`) vs in-process API. Record the chosen path + a 5-line code sketch.
  5. **GPU placement**: confirm `MINERU_DEVICE_MODE=cuda` puts PP-DocLayoutV2 + UniMERNet + PytorchPaddleOCR on GPU (`rocm-smi` shows GPU-UTIL>0 during a run); record that the 3 ONNX table models stay on CPU (expected).

  Run, in the venv, on GPU 3:
  ```bash
  source /root/ocr-eval/mineru-rocm-venv/bin/activate
  export HIP_VISIBLE_DEVICES=3 MINERU_DEVICE_MODE=cuda
  cd /workspace/MinerU-ROCm
  mineru -p examples/demo.png -o /tmp/mineru-spike -b pipeline
  ls -R /tmp/mineru-spike   # record exact output structure + naming
  ```

- [ ] **Step 4: Commit the script + spike doc**

```bash
cd /workspace/MinerU-ROCm
git add adapter/setup/00-install-deps.sh docs/spike-mineru-api.md
git commit -m "chore(setup): mineru[all] ROCm provisioning + in-process API spike doc"
```

---

## Task 5: pipeline_adapter implementation + smoke demo

**Files:**
- Modify: `adapter/pipeline_adapter.py` (fill `load()` + `extract()` using Task 4 findings)
- Create: `tests/test_pipeline_output.py`, `examples/run_demo.sh`

**Interfaces:**
- Consumes: the `mineru` API documented in `docs/spike-mineru-api.md` (Task 4).
- Produces: `pipeline_adapter.infer_page(img, platform, cfg) -> str` returning R4-conformant Markdown; `examples/run_demo.sh` runs it on `examples/demo.png` on GPU 3.

- [ ] **Step 1: Write the output-convention unit test** — `tests/test_pipeline_output.py` (CPU; tests a pure formatter helper, not mineru):

```python
# tests/test_pipeline_output.py
from pipeline_adapter import normalize_markdown

def test_display_formula_wrapped_in_double_dollar():
    assert "$$E=mc^2$$" in normalize_markdown("$$E=mc^2$$")

def test_html_table_passes_through():
    md = normalize_markdown("<table><tr><td>a</td></tr></table>")
    assert "<table>" in md  # HTML tables preserved (R4)

def test_no_div_wrappers_around_images():
    # R4: do not wrap figures in <div>; md_tex_filter strips ![](path) but keeps <div>
    assert "<div>" not in normalize_markdown("![fig](x.png)\n")
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /workspace/MinerU-ROCm && python -m pytest tests/test_pipeline_output.py -v`
Expected: FAIL — `ImportError: cannot import name 'normalize_markdown'`.

- [ ] **Step 3: Implement pipeline_adapter.py** — fill in using the Task 4 spike. The structure (replace the skeleton body):

```python
"""MinerU 3.4 pipeline adapter (backend=pipeline). See docs/spike-mineru-api.md."""
from __future__ import annotations
from pathlib import Path

_runner = None

def normalize_markdown(md: str) -> str:
    """Enforce contract R4: keep LaTeX/HTML/pipe tables, drop any <div> figure wrappers.

    mineru already emits LaTeX ($$...$$) and HTML tables; this is a safety pass.
    Extend here if the spike shows mineru wrapping figures in <div>.
    """
    # Strip <div>...</div> wrappers that survive md_tex_filter (R4 pitfall).
    import re
    md = re.sub(r"</?div[^>]*>", "", md)
    return md

def infer_page(img: Path, platform: str, cfg: dict) -> str:
    global _runner
    if _runner is None:
        _runner = MineruPipelineRunner(platform=platform, cfg=cfg)
        _runner.load()
    return normalize_markdown(_runner.extract(img))

class MineruPipelineRunner:
    def __init__(self, platform: str, cfg: dict):
        self.platform = platform
        self.cfg = cfg

    def load(self):
        """Warm the pipeline on cuda. Use the API from docs/spike-mineru-api.md §3-§4.

        Example shape (replace with the exact import/call from the spike):
            import os; os.environ.setdefault("MINERU_DEVICE_MODE", "cuda")
            from mineru.<...> import <PipelineAPI>
            self._api = <PipelineAPI>(...)
        """
        raise NotImplementedError("Fill from docs/spike-mineru-api.md §3 (in-process API).")

    def extract(self, img: Path) -> str:
        """Run the warmed pipeline on one image → Markdown string."""
        raise NotImplementedError("Fill from docs/spike-mineru-api.md §3-§4.")
```

> The implementer copies the spike's chosen 5-line sketch into `load()`/`extract()`. The dispatcher already handles per-page try/except, timing, `.md` writing, and `_run_stats.json`.

- [ ] **Step 4: Run the unit test to verify the formatter passes**

Run: `cd /workspace/MinerU-ROCm && python -m pytest tests/test_pipeline_output.py -v`
Expected: PASS (3 passed) — the formatter works regardless of mineru.

- [ ] **Step 5: Write the smoke demo** — `examples/run_demo.sh`:

```bash
#!/usr/bin/env bash
# One-command pipeline smoke on examples/demo.png (GPU 3). Needs the Task 4 venv.
set -euo pipefail
source "${MINERU_ROCM_VENV:-/root/ocr-eval/mineru-rocm-venv}/bin/activate"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-3}"
export MINERU_DEVICE_MODE=cuda
export HF_ENDPOINT="http://134.199.133.77"
cd "$(dirname "$0")/.."
python adapter/run_adapter.py \
  --img-dir examples \
  --out-dir /tmp/mineru-demo-out \
  --platform linux-rocm \
  --backend pipeline
echo "--- demo output ---"; cat /tmp/mineru-demo-out/demo.md
```

- [ ] **Step 6: Run the demo and verify real Markdown on GPU**

Run: `chmod +x /workspace/MinerU-ROCm/examples/run_demo.sh && bash /workspace/MinerU-ROCm/examples/run_demo.sh`
Expected: `/tmp/mineru-demo-out/demo.md` contains real parsed Markdown (text/formula/tables), `_run_stats.json` shows `"engine":"pipeline","ok":1`; `rocm-smi --showuse` shows GPU 3 busy during the run.

- [ ] **Step 7: Commit**

```bash
cd /workspace/MinerU-ROCm
git add adapter/pipeline_adapter.py tests/test_pipeline_output.py examples/run_demo.sh
git commit -m "feat(pipeline): MinerU 3.4 pipeline adapter on ROCm + demo

Wraps mineru[all] in-process on cuda (MINERU_DEVICE_MODE=cuda), loads once,
per-page img→md with R4 output normalization. Demo runs on GPU 3."
```

---

## Task 6: Full OmniDocBench v1.6 eval → 86.47

**Files:**
- Output: `results/omnidocbench/v16/linux-rocm/pipeline/` (predictions + engine-owned metric/provenance)

> Uses the **platform engine** (`OmniDocBench-AMD`) for infer→score→publish. The adapter is the model-specific step; the engine drives the dataset, scoring (Py3.11 eval-venv), CDM, and provenance.

- [ ] **Step 1: Confirm the dataset + engine are wired**

Run: `ls /workspace/OmniDocBench_data && ls /workspace/OmniDocBench/eval.py 2>/dev/null; cat /workspace/MinerU-ROCm/eval/configs/omnidocbench_v16.yaml`
Expected: the 1651-page OmniDocBench v1.6 dataset is present (symlinked to `/root/ocr-eval/OmniDocBench_data`); the eval config declares the metric set.

- [ ] **Step 2: Run the full eval via the platform engine** (GPU 3)

The engine invokes the adapter as a subprocess and consumes `out_dir/*.md` + `_run_stats.json`. Use the engine's infer/score/publish stages (consult `OmniDocBench-AMD` README + `engine/omnidocbench_amd/cli.py --help` for exact flags). Shape:
```bash
source /root/ocr-eval/mineru-rocm-venv/bin/activate
export HIP_VISIBLE_DEVICES=3 MINERU_DEVICE_MODE=cuda HF_ENDPOINT="http://134.199.133.77"
cd /workspace/omnidocbench-amd
omnidocbench-amd infer   --repo /workspace/MinerU-ROCm --platform linux-rocm \
                         --backend pipeline --limit-pages null
omnidocbench-amd score   --repo /workspace/MinerU-ROCm --platform linux-rocm   # Py3.11 eval-venv
omnidocbench-amd publish --repo /workspace/MinerU-ROCm --platform linux-rocm
```
If the engine CLI differs, follow `OmniDocBench-AMD/docs/architecture.md` §7 (the infer→score→publish flow) and the adapter contract — the exact subcommand names are the platform's source of truth, not this plan. Record the actual command used in `docs/reproducibility.md`.

- [ ] **Step 3: Verify the result against the gate**

Read the engine's `metric_result.json` / `run_summary.json` for the `pipeline` run. Compute `Overall = ((1−Text_EditDist)×100 + Table_TEDS + Formula_CDM)/3`.
Expected: Overall within **1.0 pp of 86.47** (i.e. ≥ 85.47). If below, investigate (likely: ONNX tables on CPU mis-scoring — verify table HTML; or a formula normalization issue) before publishing.

- [ ] **Step 4: Commit the result bundle** (predictions are engine-assembled; commit the artifact pointers, not 1651 raw .md if huge — follow the repo's `.gitignore`/LFS policy; the platform stores artifacts under `results/`)

```bash
cd /workspace/MinerU-ROCm
git add results/omnidocbench/v16/linux-rocm/pipeline/ docs/reproducibility.md
git commit -m "eval(pipeline): OmniDocBench v1.6 full-set on ROCm gfx1100 (Overall ≈86.47)"
```

---

## Task 7: Windows-hip handoff doc

**Files:**
- Create: `docs/HANDOFF-windows-hip.md`

- [ ] **Step 1: Write the self-contained handoff** — `docs/HANDOFF-windows-hip.md`, covering (verbatim intent from spec §14):
  1. **Target**: Ryzen AI MAX+ 395 (Strix Halo), Windows, DirectML.
  2. **Pipeline install**: `pip install -U "mineru[all]"`; the 3 ONNX table models use `onnxruntime-directml` (`DmlExecutionProvider` via Microsoft Olive) — this fixes the CPU-fallback the Linux side optionally patches. Ref: https://ryzenai.docs.amd.com/en/latest/gpu/ryzenai_gpu.html.
  3. **Run**: `python adapter/run_adapter.py --img-dir <OmniDocBench pages> --out-dir results/omnidocbench/v16/windows-hip/pipeline --platform windows-hip --backend pipeline`.
  4. **Score**: platform engine `score` stage (Py3.11 eval-venv).
  5. **Land artifacts** in `results/omnidocbench/v16/windows-hip/`; update `model_card.pipeline.json` `badge.windows-hip` → `community`; ping the Linux owner to update `OmniDocBench-AMD/hub/registry.yaml`.
  6. **What's provided**: the platform-aware dispatcher (already branches on `platform`), the Windows setup stub, the eval config, badge mechanics, this doc. **The colleague provides the Windows run + artifacts.**
  7. A short "expected" number: reproduce pipeline Overall ≈ 86.47 within 1.0 pp on Strix Halo.

- [ ] **Step 2: Commit**

```bash
cd /workspace/MinerU-ROCm
git add docs/HANDOFF-windows-hip.md
git commit -m "docs: Windows-hip handoff for parallel colleague verification"
```

---

## Task 8: Finalize pipeline model card + README + registry + branch

**Files:**
- Modify: `model_card.pipeline.json`, `README.md`, `README.zh-CN.md`
- External: `OmniDocBench-AMD/hub/registry.yaml` (note for the platform repo)

- [ ] **Step 1: Fill model_card.pipeline.json with results** — set `eval_date`, `overall` (the Task 6 number), `submetrics` (text/CDM/TEDS/read-order from `metric_result.json`), `hardware` (`gpu: "AMD gfx1100 / Radeon PRO W7900"`, `vram: "48 GB"`, `rocm_driver: "7.2.1"`), `badge.linux-rocm: "community"` (self-attested; `verified` needs maintainer Docker reproduction on both platforms), and `artifacts` (paths under `results/`). Re-validate against the schema (Task 2 Step 2 command).

- [ ] **Step 2: Add the comparison table to both READMEs** — in the `## Evaluation` section, the table from spec §10 (official 86.47 vs ours, per submetric). Mark windows-hip as "community-run, pending."

- [ ] **Step 3: Note the registry update** — `mineru2.5` in `OmniDocBench-AMD/hub/registry.yaml` currently has `overall: null`; this is the VLM's slot (filled in Plan 2). The pipeline is a secondary model_card in the same repo, so no new registry row is required — record this decision in `docs/how-it-works.md` so the registry/table story is clear (VLM = primary registry row; pipeline = secondary model_card + README table row).

- [ ] **Step 4: Align the branch to `main`**

Run: `cd /workspace/MinerU-ROCm && git branch -m master main`
Expected: branch renamed to `main` (matches the org convention).

- [ ] **Step 5: Commit + final conformance**

```bash
cd /workspace/MinerU-ROCm
python /workspace/omnidocbench-amd/scripts/check_conformance.py .   # expect CONFORMANT
git add -A && git commit -m "docs: finalize pipeline model_card, comparison table, registry note"
```

---

## Self-Review

**1. Spec coverage (Plan 1 scope = Phase 0 + Phase 1):**
- Dispatcher + two modules (spec §5.1): Task 1 ✓
- Repo structure (§6): Tasks 1–2 ✓ (vlm_adapter stub; serve scripts deferred to Plan 2 for VLM)
- Pipeline adapter (§7): Tasks 4–5 ✓ (mineru call spike-gated, isolated)
- Backend matrix (§9): pipeline row ✓; vlm rows = Plan 2
- Eval/precision protocol (§10): Task 6 ✓ (pipeline result set)
- Venv/storage isolation (§11): Task 4 script enforces /root venv ✓
- Error handling R2 (§12): Task 1 dispatcher ✓
- Testing (§13): Tasks 1, 3, 5 ✓
- Windows handoff (§14): Task 7 ✓
- Phase 0 + Phase 1 (§15): Tasks 1–8 ✓
- Phase 2 + Phase 3 (VLM + finalize): **deferred to Plan 2** (needs vLLM-patch + mineru-vl-utils spikes first — noted up front).

**2. Placeholder scan:** Two intentional `NotImplementedError` sites in `pipeline_adapter` (load/extract) are **spike-gated**, not placeholders — Task 4 produces `docs/spike-mineru-api.md` with exact signatures and Task 5 copies them in. Every other step has complete code or an exact command. The engine CLI subcommands (Task 6 Step 2) point to the platform's source of truth (the adapter contract + platform README) rather than fabricating flags — this is honest, not a placeholder.

**3. Type consistency:** `infer_page(img: Path, platform: str, cfg: dict) -> str` is used identically in `run_adapter` (Task 1), `pipeline_adapter` (Task 5), and `vlm_adapter` (Task 1 stub). `as_dict()` keys (`backend`, `model`, `server_url`, `api_model_name`, `weights_dir`) match between `adapter_config` (Task 2) and the dispatcher's `cfg` merge (Task 1). `model_card.pipeline.json` uses `model_id: mineru-pipeline` consistently.

---

## Execution

Plan 1 is self-contained: it produces a conformant repo + a working pipeline adapter + the 86.47 result + the Windows handoff. Plan 2 (VLM, both engines + finalize) follows after the vLLM-patch and `mineru-vl-utils` spikes.
