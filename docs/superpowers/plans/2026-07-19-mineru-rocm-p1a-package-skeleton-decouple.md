# MinerU-ROCm P1a — Package Skeleton & Dependency Decoupling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create the `src/mineru_rocm/` importable package, port a local `types.py` so the dispatcher no longer imports `omnidocbench_amd.types`, move the inference backends into the package, and prove the test suite passes with **only `[dev]` installed (no `omnidocbench-amd`)** — the foundational decoupling that all of P1b–P1d builds on. **No inference behavior or score change.**

**Architecture:** A new `src/mineru_rocm/` package (GPU-free core, mirroring Hunyuan's `src/hunyuan_ocr/` layout) holds the real logic. `adapter/run_adapter.py` stays as a **thin platform-shim script** (the omnidocbench-amd engine invokes it as a subprocess) that delegates into the package. The omnidocbench-amd `types` import — the single coupling line — is replaced by a local verbatim port (~30 LOC, stdlib-only). The three existing tests are rewritten to import from the package.

**Tech Stack:** Python 3.11+, setuptools (src-layout), pytest.

## Global Constraints

(From the approved spec `docs/superpowers/specs/2026-07-19-mineru-rocm-standalone-port-design.md` §3.2; every task implicitly includes these.)

- **No behavior change / no score change.** The dispatcher's per-page R2 contract (record failure, continue, never raise), the `_run_stats.json` shape (8 keys: `schema_version, count, ok, fail, fallback, limit_pages, engine, stats`), and the `infer_page(img, platform, cfg) -> str` signatures MUST stay identical. Existing test assertions must still hold.
- **Core package is GPU-free and platform-free.** `src/mineru_rocm/` must import with **no** `omnidocbench-amd`, `torch`, `mineru`, `mineru_vl_utils`, `openai`, or `transformers` installed — all heavy deps stay lazy inside the backend modules.
- **Engine contract preserved.** `adapter/run_adapter.py` MUST remain a runnable script with the same argparse CLI (`--img-dir --out-dir --platform --backend --server-url --api-model-name --skip-existing`) — the omnidocbench-amd engine invokes it as `python adapter/run_adapter.py …`.
- **Field order is load-bearing.** `PageStatus(image, status, error="", seconds=0.0, attempts=0)` and `RunSummary(count, ok, fail, fallback, limit_pages, stats, engine="")` — exact field declaration order (positional construction at call sites).
- **One concern per commit; commit after every task's validation passes.** Branch: `feat/p1-standalone-package-cli` (off `main` @ the P0 merge).

---

## File Structure (P1a scope)

| File | Action | Responsibility |
|---|---|---|
| `src/mineru_rocm/__init__.py` | Create | Package identity + `__version__` |
| `src/mineru_rocm/types.py` | Create | Local `RunSummary` / `PageStatus` (verbatim port of engine `types.py`); stdlib-only |
| `src/mineru_rocm/backends/__init__.py` | Create | Backends subpackage marker |
| `src/mineru_rocm/backends/pipeline.py` | Move from `adapter/pipeline_adapter.py` | In-process MinerU 3.4 pipeline (logic unchanged) |
| `src/mineru_rocm/backends/vlm.py` | Move from `adapter/vlm_adapter.py` | vLLM/transformers VLM (logic unchanged) |
| `src/mineru_rocm/dispatcher.py` | Create (logic from `adapter/run_adapter.py`) | The real `run_adapter(...)` + routing; imports `mineru_rocm.types` + `mineru_rocm.backends` |
| `adapter/run_adapter.py` | Rewrite → thin shim | Engine subprocess entry; delegates to `mineru_rocm.dispatcher` (same CLI) |
| `adapter/adapter_config.py` | Keep (referenced by dispatcher) | Stdlib-only config; unchanged |
| `pyproject.toml` | Modify | Add `[tool.setuptools.packages.find] where=["src"]`; keep version 0.1.0 |
| `conftest.py` | Modify | Drop the `sys.path.insert(.../adapter)` hack; tests import the installed package |
| `tests/test_types.py` | Create | New: assert the 8-key JSON shape + positional ctors + `.write()` round-trip |
| `tests/test_dispatcher.py` | Modify | Rewrite imports to `mineru_rocm.dispatcher` / `mineru_rocm.backends.{pipeline,vlm}` |
| `tests/test_pipeline_output.py` | Modify | `from mineru_rocm.backends.pipeline import normalize_markdown` |
| `tests/test_vlm_adapter.py` | Modify | `from mineru_rocm.backends.vlm import normalize_vlm_markdown` |
| `scripts/check_deps.py` | Modify (P0 file) | Extend: assert `src/mineru_rocm` has NO `omnidocbench_amd` import |

Out of P1a (later sub-phases): `validation`/`scoring`/`omnidocbench`/`preflight` (P1b), `runner`/`endpoint_pool`/drivers (P1c), `cli`/`check_repo` (P1d).

---

## Task 1: Create the `src/mineru_rocm/` package + wire setuptools src-layout

**Files:**
- Create: `src/mineru_rocm/__init__.py`
- Modify: `pyproject.toml` (add packages.find)
- Modify: `scripts/check_deps.py` (add package + no-engine-import assertions)
- Test: `tests/test_types.py` is Task 2; here just import-smoke.

**Interfaces:**
- Consumes: nothing (foundational).
- Produces: an importable `mineru_rocm` package; later tasks add submodules.

- [ ] **Step 1: Write the failing check (extends P0 validator)**

Append to `scripts/check_deps.py` (after the existing assertions, before the final `print`):

```python
# P1a: src/mineru_rocm package exists, is src-layout, and core has no engine import.
import subprocess
root = Path(__file__).resolve().parents[1]
pkg = root / "src" / "mineru_rocm" / "__init__.py"
assert pkg.is_file(), f"package missing: {pkg}"
assert (root / "pyproject.toml").read_text().find('[tool.setuptools.packages.find]') != -1, "src-layout not declared"
# core package must not import the platform engine at module top level
for py in (root / "src" / "mineru_rocm").rglob("*.py"):
    src = py.read_text()
    assert "omnidocbench_amd" not in src, f"engine import leaked into package: {py}"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `/opt/venv/bin/python scripts/check_deps.py`
Expected: FAIL — `AssertionError: package missing: .../src/mineru_rocm/__init__.py`.

- [ ] **Step 3: Create `src/mineru_rocm/__init__.py`**

```python
"""mineru_rocm — evaluation-backed AMD ROCm port of opendatalab/MinerU.

Benchmark infrastructure for running the MinerU 3.4 pipeline and the
MinerU2.5-Pro VLM on AMD gfx1100 (RDNA3) and scoring them on OmniDocBench v1.6.
Evaluation-backed, not precision-aligned. See the project README and
docs/superpowers/specs/.
"""

__version__ = "0.1.0"
```

- [ ] **Step 4: Declare the src-layout in `pyproject.toml`**

Add at the end of `pyproject.toml` (after the `[project.urls]` block):

```toml
[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 5: Run the validator + import smoke**

Run: `/opt/venv/bin/python scripts/check_deps.py`
Expected: `P0 pyproject OK`

Run: `/opt/venv/bin/pip install -e . -q && /opt/venv/bin/python -c "import mineru_rocm; print('mineru_rocm', mineru_rocm.__version__)"`
Expected: `mineru_rocm 0.1.0`

Run: `/opt/venv/bin/python -m pytest -q`
Expected: the existing 15 tests still pass (they still import via the old `adapter/` paths + conftest; unchanged this task).

- [ ] **Step 6: Commit**

```bash
git add src/mineru_rocm/__init__.py pyproject.toml scripts/check_deps.py
git commit -m "build(p1a): src/mineru_rocm package skeleton (src-layout); validator asserts no engine import"
```

---

## Task 2: Local `types.py` — port `RunSummary` / `PageStatus` (TDD)

**Files:**
- Create: `src/mineru_rocm/types.py`
- Test: `tests/test_types.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `mineru_rocm.types.RunSummary` / `PageStatus` with the EXACT engine contract (positional ctor order, 8-key `_run_stats.json`, `.write()`/`.to_run_stats()`). Task 3's dispatcher imports these.

- [ ] **Step 1: Write the failing test**

Create `tests/test_types.py`:

```python
import json
from pathlib import Path
from mineru_rocm.types import RunSummary, PageStatus


def test_page_status_positional_and_kwargs():
    # ok-path call shape (from the dispatcher)
    a = PageStatus("p1.jpg", "ok", seconds=1.25, attempts=1)
    assert a.image == "p1.jpg" and a.status == "ok" and a.seconds == 1.25 and a.attempts == 1
    # failed-path call shape (attempts omitted -> default 0)
    b = PageStatus("p2.jpg", "failed: boom", error="boom", seconds=0.5)
    assert b.status.startswith("failed") and b.error == "boom" and b.attempts == 0


def test_run_summary_positional_engine_kwarg():
    stats = [PageStatus("p.jpg", "ok")]
    rs = RunSummary(1, 1, 0, 0, None, stats, engine="smoke")
    assert (rs.count, rs.ok, rs.fail, rs.fallback, rs.limit_pages, rs.engine) == (1, 1, 0, 0, None, "smoke")


def test_to_run_stats_emits_eight_keys():
    rs = RunSummary(2, 1, 1, 0, None, [PageStatus("a", "ok"), PageStatus("b", "failed: x")], engine="pipeline")
    d = rs.to_run_stats()
    assert set(d) == {"schema_version", "count", "ok", "fail", "fallback", "limit_pages", "engine", "stats"}
    assert d["schema_version"] == 1
    assert (d["count"], d["ok"], d["fail"]) == (2, 1, 1)
    assert d["engine"] == "pipeline"
    assert d["stats"][0]["image"] == "a" and d["stats"][1]["status"] == "failed: x"


def test_write_round_trips(tmp_path):
    rs = RunSummary(1, 1, 0, 0, None, [PageStatus("p.jpg", "ok", seconds=0.1)], engine="vlm-vllm")
    out = rs.write(tmp_path / "_run_stats.json")
    assert out.exists()
    d = json.loads((tmp_path / "_run_stats.json").read_text())
    assert d["count"] == 1 and d["ok"] == 1 and d["engine"] == "vlm-vllm"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `/opt/venv/bin/python -m pytest tests/test_types.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'mineru_rocm.types'`.

- [ ] **Step 3: Implement `src/mineru_rocm/types.py`** (verbatim contract of the engine `types.py`)

```python
"""Run summary + per-page status types for mineru_rocm.

Local verbatim port of the omnidocbench_amd.types contract (RunSummary /
PageStatus), so the dispatcher no longer imports the platform engine. Same
field order and the same 8-key _run_stats.json shape the engine and the
existing tests expect. Stdlib-only.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

SCHEMA_VERSION = 1


@dataclass
class PageStatus:
    image: str
    status: str  # "ok" | "failed: <reason>" | "fallback: <reason>"
    error: str = ""
    seconds: float = 0.0
    attempts: int = 0


@dataclass
class RunSummary:
    count: int
    ok: int
    fail: int
    fallback: int
    limit_pages: int | None
    stats: list[PageStatus] = field(default_factory=list)
    engine: str = ""

    def to_run_stats(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "count": self.count,
            "ok": self.ok,
            "fail": self.fail,
            "fallback": self.fallback,
            "limit_pages": self.limit_pages,
            "engine": self.engine,
            "stats": [asdict(s) for s in self.stats],
        }

    def write(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_run_stats(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    @classmethod
    def from_run_stats(cls, path: Path) -> "RunSummary":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            count=d["count"], ok=d["ok"], fail=d["fail"], fallback=d["fallback"],
            limit_pages=d.get("limit_pages"),
            stats=[PageStatus(**s) for s in d.get("stats", [])],
            engine=d.get("engine", ""),
        )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `/opt/venv/bin/python -m pytest tests/test_types.py -q`
Expected: `4 passed`.

Run: `/opt/venv/bin/python scripts/check_deps.py`
Expected: still `P0 pyproject OK` (no engine import in types.py).

- [ ] **Step 5: Commit**

```bash
git add src/mineru_rocm/types.py tests/test_types.py
git commit -m "feat(p1a): local types.py (RunSummary/PageStatus) — verbatim engine contract port; 4 tests"
```

---

## Task 3: Move backends into the package + refactor the dispatcher + rewrite test imports

**Files:**
- Create: `src/mineru_rocm/backends/__init__.py`
- Move: `adapter/pipeline_adapter.py` → `src/mineru_rocm/backends/pipeline.py` (logic unchanged)
- Move: `adapter/vlm_adapter.py` → `src/mineru_rocm/backends/vlm.py` (logic unchanged; the vendored `qwen2vl_chat_template.jinja` moves alongside or is located via `__file__`)
- Create: `src/mineru_rocm/dispatcher.py` (logic from `adapter/run_adapter.py`, but `from mineru_rocm.types import ...` and package-relative backend imports)
- Rewrite: `adapter/run_adapter.py` → thin shim (engine subprocess entry; same CLI)
- Modify: `conftest.py` (drop the sys.path hack)
- Modify: `tests/test_dispatcher.py`, `tests/test_pipeline_output.py`, `tests/test_vlm_adapter.py` (package imports)

**Interfaces:**
- Consumes: `mineru_rocm.types` (Task 2).
- Produces: `mineru_rocm.dispatcher.run_adapter(img_dir, out_dir, *, platform, config, skip_existing=False) -> dict` (identical signature/behavior to today); `mineru_rocm.backends.pipeline.infer_page` / `mineru_rocm.backends.vlm.infer_page` (identical signatures); `adapter/run_adapter.py` remains a runnable engine shim.

- [ ] **Step 1: Move the backend modules**

Move `adapter/pipeline_adapter.py` → `src/mineru_rocm/backends/pipeline.py` and `adapter/vlm_adapter.py` → `src/mineru_rocm/backends/vlm.py` (preserve their content byte-for-byte — only the file location changes). Create `src/mineru_rocm/backends/__init__.py`:

```python
"""Inference backends for mineru_rocm.

- pipeline: in-process MinerU 3.4 pipeline on ROCm cuda.
- vlm: MinerU2.5-Pro VLM via vLLM-on-ROCm (http-client) or transformers.

Each backend exposes infer_page(img: Path, platform: str, cfg: dict) -> str.
Heavy deps (mineru, mineru_vl_utils, transformers, PIL) are imported lazily
inside methods so the package imports with no GPU deps installed.
"""
```

Move the vendored `adapter/qwen2vl_chat_template.jinja` → `src/mineru_rocm/backends/qwen2vl_chat_template.jinja` and update the one `__file__`-relative reference inside `vlm.py` (it already resolves the template as a sibling of `__file__`, so no logic change — just confirm the path resolves after the move).

- [ ] **Step 2: Create `src/mineru_rocm/dispatcher.py`**

Copy the logic of `adapter/run_adapter.py` verbatim, with exactly three edits:
1. Replace `from omnidocbench_amd.types import RunSummary, PageStatus` with `from mineru_rocm.types import RunSummary, PageStatus`.
2. Replace `SUB_ADAPTERS = {"pipeline": "pipeline_adapter", "vlm-vllm": "vlm_adapter", "vlm-transformers": "vlm_adapter"}` with `SUB_ADAPTERS = {"pipeline": "pipeline", "vlm-vllm": "vlm", "vlm-transformers": "vlm"}`.
3. Simplify `_import_sub`/`_load_adapter_config` to package-relative imports only (the bare-script fallback is no longer needed inside the package). The `__main__` argparse block stays, so the module is also runnable as `python -m mineru_rocm.dispatcher`.

Keep `adapter/adapter_config.py` referenced (the dispatcher still loads it). Since `adapter_config.py` is stdlib-only and engine-free, leave it where it is and import it via the existing dual-mode loader OR move it into the package as `mineru_rocm.config`. **Decision: move it** — copy `adapter/adapter_config.py` → `src/mineru_rocm/config.py`, and in `dispatcher.py` replace `_load_adapter_config()` with a direct `from mineru_rocm import config as adapter_config`.

- [ ] **Step 3: Rewrite `adapter/run_adapter.py` as the engine shim**

```python
"""omnidocbench-amd platform shim — the engine invokes this as a subprocess.

Thin entry that delegates to mineru_rocm.dispatcher, preserving the engine's
adapter contract (runnable script, same CLI). The real logic lives in the
mineru_rocm package; this shim exists only so `python adapter/run_adapter.py …`
keeps working for the optional [platform] integration.
"""
import sys
from pathlib import Path

# Allow running as a bare script (no parent package): put src/ on sys.path.
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mineru_rocm.dispatcher import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
```

(This relies on `dispatcher.main(argv=None) -> int` existing — add it: refactor `dispatcher.py`'s `__main__` block into a `def main(argv=None) -> int:` that builds the parser, parses, calls `run_adapter`, returns 0. Keep the exact argparse surface.)

- [ ] **Step 4: Rewrite the tests + conftest**

Replace `conftest.py` with:

```python
# conftest.py — the package is installed (pip install -e .); no sys.path hacks.
```

Rewrite imports in the three test files:
- `tests/test_dispatcher.py`: `from mineru_rocm import dispatcher` and `from mineru_rocm.backends import pipeline, vlm`; replace `run_adapter.run_adapter(...)` → `dispatcher.run_adapter(...)`; replace `monkeypatch.setattr(pipeline_adapter, ...)` → `monkeypatch.setattr(pipeline, ...)`; replace `vlm_adapter` → `vlm`. The 5 test bodies' assertions stay identical.
- `tests/test_pipeline_output.py`: `from mineru_rocm.backends.pipeline import normalize_markdown` (drop its internal sys.path insert).
- `tests/test_vlm_adapter.py`: `from mineru_rocm.backends.vlm import normalize_vlm_markdown`.

- [ ] **Step 5: Validate — full suite + the decoupling proof**

Run: `/opt/venv/bin/pip install -e . -q && /opt/venv/bin/python -m pytest -q`
Expected: all tests pass (the original dispatcher/pipeline/vlm tests, now via the package, + the 4 new `test_types`).

**The decoupling proof** — prove the package needs no engine:
Run: `/opt/venv/bin/python -c "import mineru_rocm, mineru_rocm.types, mineru_rocm.dispatcher, mineru_rocm.backends.pipeline, mineru_rocm.backends.vlm; print('package imports with no engine dep')"`
Expected: `package imports with no engine dep` (the backends import lazily; nothing pulls `omnidocbench_amd`).

Run: `/opt/venv/bin/python scripts/check_deps.py`
Expected: `P0 pyproject OK` (validator now also confirms no `omnidocbench_amd` anywhere in `src/mineru_rocm/`).

Confirm the engine shim still parses:
Run: `/opt/venv/bin/python adapter/run_adapter.py --help`
Expected: argparse usage text with `--img-dir --out-dir --platform --backend ...` (the engine contract is intact).

- [ ] **Step 6: Commit**

```bash
git add src/mineru_rocm/ adapter/run_adapter.py conftest.py tests/
git rm adapter/pipeline_adapter.py adapter/vlm_adapter.py adapter/qwen2vl_chat_template.jinja adapter/adapter_config.py 2>/dev/null || true
# (if git mv was used instead of cp+rm, the moves are already staged)
git commit -m "refactor(p1a): move backends + dispatcher into mineru_rocm package; adapter/ -> engine shim; decouple omnidocbench_amd.types"
```

---

## Definition of Done (P1a)

- [ ] `import mineru_rocm` (+ `.types`, `.dispatcher`, `.backends.pipeline`, `.backends.vlm`) succeeds with NO `omnidocbench-amd` installed.
- [ ] `src/mineru_rocm/` contains zero `omnidocbench_amd` references (asserted by `scripts/check_deps.py`).
- [ ] `python -m pytest -q` green: the original 5 dispatcher + 6 pipeline-output + 3 vlm tests (now via package imports) + 4 new `test_types`.
- [ ] `adapter/run_adapter.py --help` shows the unchanged engine CLI (platform shim contract intact).
- [ ] No `_run_stats.json` shape change (8 keys, same field order); no `infer_page` signature change; no score/behavior change.
- [ ] `scripts/check_deps.py` → `P0 pyproject OK`.

## Follow-on sub-phases (separate plans)

- **P1b** — eval plumbing port: `omnidocbench.py`, `validation.py`, `preflight.py`, `scoring.py` (+ per-model `data/eval_config.yaml` templates) from Hunyuan; CPU unit tests.
- **P1c** — runner + endpoint_pool + drivers: port `runner.py` (atomic/resume/`run_manifest.json` conservation laws/RunLock — the resume-`ok_pages` fix), `endpoint_pool.py` (VLM track), write the pipeline in-process driver + the VLM http-client driver.
- **P1d** — CLI + `check_repo.py`: `mineru-rocm doctor|validate|predict|score|canary|manifest verify`; `scripts/check_repo.py` (lock/README consistency); `[project.scripts]` entry.
