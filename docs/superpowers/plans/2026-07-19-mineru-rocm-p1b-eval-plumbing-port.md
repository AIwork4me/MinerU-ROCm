# MinerU-ROCm P1b — Eval-Plumbing Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the five eval-plumbing modules (`omnidocbench`, `preflight`, `validation`, `scoring`, `canary`) from the HunyuanOCR-ROCm reference into `src/mineru_rocm/`, reclaiming dataset iteration, pre-score validation, OmniDocBench scoring, and canary materialization into this repo — with CPU-only unit tests and **no score/behavior change**.

**Architecture:** A faithful, near-verbatim port of `hunyuan_ocr.{omnidocbench,preflight,validation,scoring,canary}` (the reference is importable as `hunyuan-ocr 0.1.1` in `/opt/venv`, source at `/workspace/HunyuanOCR-ROCm/src/hunyuan_ocr/`). All five are stdlib-only (only `scoring` needs `yaml`, lazy-imported), so they land cleanly in the GPU-free core established by P1a. Three deliberate adaptations (each justified inline): (1) `validation.py` owns `ERROR_PREFIX`/`_OWN_ARTIFACTS` locally (Hunyuan imports them from `runner`, a P1c module — localizing keeps P1b self-contained; P1c's runner imports them back); (2) `scoring.py` lazy-imports `yaml` inside `write_eval_config` (keeps `dependencies=[]` and matches P1a's lazy discipline); (3) `scoring.py` drops Hunyuan's repo-relative `_REPO_TEMPLATE` fallback (the bundled package-data template is sufficient under editable + wheel installs).

**Tech Stack:** Python 3.11+, setuptools src-layout, pytest, stdlib + `yaml` (lazy).

## Global Constraints

(From the approved spec `docs/superpowers/specs/2026-07-19-mineru-rocm-standalone-port-design.md` §3.2; every task implicitly includes these.)

- **No behavior / no score change.** The OmniDocBench v1.6 Overall formula MUST stay `((1-text_edit)*100 + cdm*100 + teds*100)/3` (reading-order Edit_dist reported separately, NOT in Overall). `scoring.run_scorer` MUST invoke the same `pdf_validation.py --config <cfg>` in the pinned OmniDocBench venv — verified: Hunyuan's `scoring` is itself a wrapper around that same scorer, so a verbatim port reproduces the recorded numbers. (Hunyuan source: `scoring.py:82-83`.)
- **Core package stays GPU-free and platform-free.** The new modules must import with **no** `torch`, `mineru`, `mineru_vl_utils`, `omnidocbench_amd`, `openai`, or `transformers` at module top level. `yaml` is imported lazily inside `write_eval_config`. `pyproject.toml` `dependencies` stays `[]` (the P0 validator asserts this; do not change it).
- **Engine subprocess contract untouched.** P1b does NOT modify `adapter/run_adapter.py`, `dispatcher.py`, `backends/*`, or `types.py`. `adapter/run_adapter.py --help` stays byte-identical.
- **Verbatim ports.** Each module's public API (function/class names, signatures, return shapes, error classes) MUST match `hunyuan_ocr` byte-for-byte except the three documented adaptations. The implementer READS the reference source (at `/workspace/HunyuanOCR-ROCm/src/hunyuan_ocr/<module>.py`, or `/opt/venv/bin/python -c "import hunyuan_ocr.<module>, inspect; print(inspect.getsource(hunyuan_ocr.<module>))"`) and ports it; this plan specifies the adaptations, the signatures cross-task contracts rely on, and the tests — not a re-paste of the reference body.
- **One concern per commit; commit after every task's validation passes.** Branch: `feat/p1b-eval-plumbing` (off `main` @ `609eacb`, which has P1a).
- **Validation environment:** `/opt/venv/bin/python` and `/opt/venv/bin/pip` ONLY (py3.12; has `hunyuan-ocr`, `omnidocbench_amd`, `mineru-rocm` editable, `pytest`, `PyYAML`). The real OmniDocBench scorer (only needed for an end-to-end score smoke, deferred) lives at `/root/ocr-eval/OmniDocBench` with venv `/root/ocr-eval/OmniDocBench/.venv/bin/python` — these are already `scoring.py`'s env-overridable defaults.

---

## File Structure (P1b scope)

| File | Action | Responsibility |
|---|---|---|
| `src/mineru_rocm/omnidocbench.py` | Create (port from `hunyuan_ocr.omnidocbench`) | Dataset iteration + prediction filename mapping (stdlib-only) |
| `src/mineru_rocm/preflight.py` | Create (port from `hunyuan_ocr.preflight`) | Fail-fast input validation + sharding, before model load (stdlib-only) |
| `src/mineru_rocm/validation.py` | Create (port from `hunyuan_ocr.validation`) | Pre-score prediction-dir validation; **owns** `ERROR_PREFIX` + `_OWN_ARTIFACTS` |
| `src/mineru_rocm/canary.py` | Create (port from `hunyuan_ocr.canary`) | Byte-identical canary materialization from full GT (stdlib-only) |
| `src/mineru_rocm/scoring.py` | Create (port from `hunyuan_ocr.scoring`) | OmniDocBench eval-config writer + scorer + result parser; lazy `yaml` |
| `src/mineru_rocm/data/eval_config.yaml` | Create (port verbatim) | Bundled OmniDocBench v1.6 metric template (model-agnostic) |
| `pyproject.toml` | Modify | Add `[tool.setuptools.package-data]` so the template ships in the wheel |
| `tests/test_omnidocbench.py` | Create | TDD for `omnidocbench` |
| `tests/test_preflight.py` | Create | TDD for `preflight` |
| `tests/test_validation.py` | Create | TDD for `validation` (incl. localized constants) |
| `tests/test_canary.py` | Create | TDD for `canary` |
| `tests/test_scoring.py` | Create | TDD for `scoring` (monkeypatched; no real scorer needed) |

Out of P1b: `runner.py` / `endpoint_pool.py` / drivers (P1c); `cli.py` / `check_repo.py` (P1d). `canary.py` is IN P1b (it is standalone stdlib, no P1c deps — revised up from an implicit P1d placement).

---

## Task 1: Port `omnidocbench.py` (dataset iteration + filename mapping)

**Files:**
- Create: `src/mineru_rocm/omnidocbench.py`
- Test: `tests/test_omnidocbench.py`

**Interfaces:**
- Consumes: nothing (foundational; pure stdlib).
- Produces (exact signatures, must match `hunyuan_ocr.omnidocbench`):
  - `derive_prediction_filename(image_path: str | Path) -> str`  → `"<stem>.md"`
  - `iter_page_images(gt_json: str | Path, images_dir: str | Path) -> Iterator[tuple[str, Path]]`  → yields `(image_stem, abs_image_path)`

- [ ] **Step 1: Write the failing test**

Create `tests/test_omnidocbench.py`:

```python
from pathlib import Path
from mineru_rocm.omnidocbench import derive_prediction_filename, iter_page_images


def test_derive_prediction_filename_strips_dir_and_ext():
    assert derive_prediction_filename("a/b/c/page-001.png") == "page-001.md"
    assert derive_prediction_filename(Path("/x/yo.JPG")) == "yo.md"
    assert derive_prediction_filename("noext") == "noext.md"


def test_iter_page_images_yields_stem_and_abs_path(tmp_path):
    gt = tmp_path / "gt.json"
    gt.write_text(
        '[{"page_info": {"image_path": "page-001.png"}}, '
        '{"page_info": {"image_path": "sub/page-002.jpg"}}]',
        encoding="utf-8",
    )
    images = tmp_path / "images"
    (images / "sub").mkdir(parents=True)
    (images / "page-001.png").write_text("x")
    (images / "sub" / "page-002.jpg").write_text("y")
    out = list(iter_page_images(gt, images))
    assert [stem for stem, _ in out] == ["page-001", "page-002"]
    assert out[0][1] == images / "page-001.png"
    assert out[1][1] == images / "sub" / "page-002.jpg"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `/opt/venv/bin/python -m pytest tests/test_omnidocbench.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'mineru_rocm.omnidocbench'`.

- [ ] **Step 3: Port `hunyuan_ocr.omnidocbench` verbatim**

Read the reference (`/workspace/HunyuanOCR-ROCm/src/hunyuan_ocr/omnidocbench.py`) and create `src/mineru_rocm/omnidocbench.py` as a verbatim port: same module docstring intent (retitled to mineru_rocm), `from __future__ import annotations`, `import json`, `from pathlib import Path`, `from typing import Iterator`, and the two functions above with identical bodies. **Zero adaptations** — this module is pure stdlib and has no Hunyuan-local coupling. Keep the SPDX header (`Apache-2.0`).

- [ ] **Step 4: Run the test to verify it passes**

Run: `/opt/venv/bin/python -m pytest tests/test_omnidocbench.py -q`
Expected: `2 passed`.

- [ ] **Step 5: Validate (no engine leak + full suite)**

Run: `/opt/venv/bin/python scripts/check_deps.py` → `P0 pyproject OK`.
Run: `/opt/venv/bin/python -c "import mineru_rocm.omnidocbench as m; import sys; print('engine:', 'omnidocbench_amd' in sys.modules)"` → `engine: False`.
Run: `/opt/venv/bin/python -m pytest -q` → 22 passed (20 from P1a + 2 new).

- [ ] **Step 6: Commit**

```bash
git add src/mineru_rocm/omnidocbench.py tests/test_omnidocbench.py
git commit -m "feat(p1b): port omnidocbench (dataset iteration + filename mapping) from Hunyuan; 2 tests"
```
(End the message body with a blank line + `Co-Authored-By: Claude <noreply@anthropic.com>`.)

---

## Task 2: Port `preflight.py` (fail-fast input validation + sharding)

**Files:**
- Create: `src/mineru_rocm/preflight.py`
- Test: `tests/test_preflight.py`

**Interfaces:**
- Consumes: nothing.
- Produces (exact signatures, must match `hunyuan_ocr.preflight`):
  - `class PreflightError(ValueError)` — `.errors: list[str]`
  - `load_gt(gt_json) -> list[dict]`
  - `pages_with_images(gt_json, images_dir) -> list[tuple[str, str]]`
  - `shard(items: list, n: int) -> list[list]`
  - `check_prediction_inputs(*, gt_json, images_dir, ports, gpu_ids, concurrency, max_retries, retry_backoff, max_pixels, model, pred_dir, backend_name=None, allowed_backends=None) -> list[tuple[str, str]]`
  - `assert_ok(problems: list[tuple[str, str]]) -> None`
  - (`_split_ids` is private; port it too.)

- [ ] **Step 1: Write the failing test**

Create `tests/test_preflight.py`:

```python
import json
import pytest
from mineru_rocm.preflight import (
    PreflightError, load_gt, pages_with_images, shard,
    check_prediction_inputs, assert_ok,
)


def _write_gt(tmp_path, pages):
    gt = tmp_path / "gt.json"
    gt.write_text(json.dumps(pages), encoding="utf-8")
    return gt


def test_load_gt_valid(tmp_path):
    gt = _write_gt(tmp_path, [{"page_info": {"image_path": "a.png"}}])
    assert load_gt(gt) == [{"page_info": {"image_path": "a.png"}}]


def test_load_gt_errors(tmp_path):
    with pytest.raises(PreflightError):  # missing file
        load_gt(tmp_path / "nope.json")
    bad = tmp_path / "bad.json"; bad.write_text("{}", encoding="utf-8")
    with pytest.raises(PreflightError):  # not a list
        load_gt(bad)
    empty = tmp_path / "empty.json"; empty.write_text("[]", encoding="utf-8")
    with pytest.raises(PreflightError):  # empty list
        load_gt(empty)
    nopage = tmp_path / "np.json"; nopage.write_text("[{}]", encoding="utf-8")
    with pytest.raises(PreflightError):  # page missing page_info
        load_gt(nopage)


def test_pages_with_images_missing_image(tmp_path):
    gt = _write_gt(tmp_path, [{"page_info": {"image_path": "a.png"}}])
    (tmp_path / "images").mkdir()
    with pytest.raises(PreflightError):
        pages_with_images(gt, tmp_path / "images")


def test_pages_with_images_ok(tmp_path):
    gt = _write_gt(tmp_path, [{"page_info": {"image_path": "a.png"}}])
    imgd = tmp_path / "images"; imgd.mkdir()
    (imgd / "a.png").write_text("x")
    assert pages_with_images(gt, imgd) == [("a", str(imgd / "a.png"))]


def test_shard_returns_exactly_n_buckets_some_empty():
    buckets = shard(["a", "b", "c"], n=5)
    assert len(buckets) == 5            # exactly n, even though len < n
    assert sum(len(b) for b in buckets) == 3


def test_shard_negative_raises():
    with pytest.raises(ValueError):
        shard(["a"], n=0)


def test_check_prediction_inputs_clean(tmp_path):
    probs = check_prediction_inputs(
        gt_json="x", images_dir="x", ports="8000,8001", gpu_ids="0,1",
        concurrency=2, max_retries=3, retry_backoff=1.5, max_pixels=1000,
        model="m", pred_dir=str(tmp_path / "pred"),
    )
    assert probs == []


def test_check_prediction_inputs_bad_ranges(tmp_path):
    probs = check_prediction_inputs(
        gt_json="x", images_dir="x", ports="", gpu_ids=None,
        concurrency=0, max_retries=0, retry_backoff=-1, max_pixels=-5,
        model="", pred_dir=str(tmp_path / "pred"),
    )
    fields = {f for f, _ in probs}
    assert {"ports", "concurrency", "max-retries", "max-pixels", "retry-backoff", "model"} <= fields


def test_assert_ok_raises_on_problems():
    assert_ok([])  # no-op
    with pytest.raises(PreflightError):
        assert_ok([("ports", "is empty")])
```

- [ ] **Step 2: Run it to verify it fails**

Run: `/opt/venv/bin/python -m pytest tests/test_preflight.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'mineru_rocm.preflight'`.

- [ ] **Step 3: Port `hunyuan_ocr.preflight` verbatim**

Read `/workspace/HunyuanOCR-ROCm/src/hunyuan_ocr/preflight.py` and create `src/mineru_rocm/preflight.py` as a verbatim port: `PreflightError`, `load_gt`, `pages_with_images`, `shard`, `_split_ids`, `check_prediction_inputs`, `assert_ok` — identical bodies. **Zero adaptations** (pure stdlib). Keep the SPDX header.

- [ ] **Step 4: Run the test to verify it passes**

Run: `/opt/venv/bin/python -m pytest tests/test_preflight.py -q`
Expected: `9 passed`.

- [ ] **Step 5: Validate**

Run: `/opt/venv/bin/python scripts/check_deps.py` → `P0 pyproject OK`.
Run: `/opt/venv/bin/python -m pytest -q` → 31 passed (22 + 9).

- [ ] **Step 6: Commit**

```bash
git add src/mineru_rocm/preflight.py tests/test_preflight.py
git commit -m "feat(p1b): port preflight (fail-fast input validation + sharding) from Hunyuan; 9 tests"
```
(+ `Co-Authored-By` trailer.)

---

## Task 3: Port `validation.py` (+ localize `ERROR_PREFIX` / `_OWN_ARTIFACTS`)

**Files:**
- Create: `src/mineru_rocm/validation.py`
- Test: `tests/test_validation.py`

**Interfaces:**
- Consumes: nothing. (Hunyuan's `from .runner import ERROR_PREFIX, _OWN_ARTIFACTS` is REPLACED — define both locally; see adaptation below.)
- Produces (exact signatures, must match `hunyuan_ocr.validation`):
  - `@dataclass class Problem` — `severity: str`, `code: str`, `message: str`, `detail: object = None`
  - `@dataclass class Report` — `expected: int`, `valid: int`, `problems: list = field(default_factory=list)`; properties `.ok`, `.ok_strict`; methods `.errors()`, `.warnings()`
  - `validate_predictions(gt_json, pred_dir, *, strict: bool = True) -> Report`
  - Module constants `ERROR_PREFIX = "ERROR:"` and `_OWN_ARTIFACTS = {"_errors", "_errors.jsonl", "run_manifest.json", "_metadata", "_eval_config.yaml", ".run.lock"}` — **owned here** (P1c's `runner` port will `from mineru_rocm.validation import ERROR_PREFIX, _OWN_ARTIFACTS`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_validation.py`:

```python
from pathlib import Path
from mineru_rocm.validation import (
    Report, Problem, validate_predictions, ERROR_PREFIX, _OWN_ARTIFACTS,
)


def _gt(tmp_path, stems):
    import json
    gt = tmp_path / "gt.json"
    gt.write_text(
        json.dumps([{"page_info": {"image_path": f"{s}.png"}} for s in stems]),
        encoding="utf-8",
    )
    return gt


def test_constants_exact():
    assert ERROR_PREFIX == "ERROR:"
    assert "_errors" in _OWN_ARTIFACTS and "run_manifest.json" in _OWN_ARTIFACTS


def test_clean(tmp_path):
    gt = _gt(tmp_path, ["a", "b"])
    pred = tmp_path / "pred"; pred.mkdir()
    (pred / "a.md").write_text("# a"); (pred / "b.md").write_text("# b")
    r = validate_predictions(gt, pred)
    assert r.expected == 2 and r.valid == 2
    assert r.ok and r.ok_strict


def test_missing(tmp_path):
    gt = _gt(tmp_path, ["a", "b"])
    pred = tmp_path / "pred"; pred.mkdir()
    (pred / "a.md").write_text("# a")  # b missing
    r = validate_predictions(gt, pred)
    assert not r.ok
    assert any(p.code == "missing" for p in r.errors())


def test_empty_and_error_marker(tmp_path):
    gt = _gt(tmp_path, ["a", "b"])
    pred = tmp_path / "pred"; pred.mkdir()
    (pred / "a.md").write_text("")                 # empty -> error
    (pred / "b.md").write_text("ERROR: boom")      # error marker
    r = validate_predictions(gt, pred)
    codes = {p.code for p in r.errors()}
    assert "empty" in codes and "error_marker" in codes


def test_partial_and_unexpected(tmp_path):
    gt = _gt(tmp_path, ["a"])
    pred = tmp_path / "pred"; pred.mkdir()
    (pred / "a.md").write_text("# a")
    (pred / "leftover.partial").write_text("x")    # partial -> error
    (pred / "strange.txt").write_text("z")          # unexpected file -> warning
    r = validate_predictions(gt, pred)
    assert any(p.code == "partial" for p in r.errors())
    assert any(p.code == "unexpected_file" and p.severity == "warning" for p in r.problems)


def test_own_artifacts_tolerated(tmp_path):
    gt = _gt(tmp_path, ["a"])
    pred = tmp_path / "pred"; pred.mkdir()
    (pred / "a.md").write_text("# a")
    (pred / "run_manifest.json").write_text("{}")  # owned -> no warning
    (pred / "_errors").mkdir()                      # owned dir -> no warning
    r = validate_predictions(gt, pred)
    assert r.ok_strict
```

- [ ] **Step 2: Run it to verify it fails**

Run: `/opt/venv/bin/python -m pytest tests/test_validation.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'mineru_rocm.validation'`.

- [ ] **Step 3: Port `hunyuan_ocr.validation` with ONE adaptation**

Read `/workspace/HunyuanOCR-ROCm/src/hunyuan_ocr/validation.py` and create `src/mineru_rocm/validation.py`. Bodies of `Problem`, `Report`, `_gt_stems`, `validate_predictions` are verbatim. **The single adaptation:** replace Hunyuan's
```python
from .runner import ERROR_PREFIX, _OWN_ARTIFACTS
```
with local definitions placed near the top of the module (after the stdlib imports):
```python
ERROR_PREFIX = "ERROR:"
# Files/dirs this project may write into a prediction directory. The validator
# treats these as expected (never flags them as unexpected_file/dir). Owned here
# so validation is self-contained; P1c's runner imports these back.
_OWN_ARTIFACTS = {
    "_errors",            # per-page error records dir
    "_errors.jsonl",      # aggregated error log
    "run_manifest.json",  # per-run manifest
    "_metadata",          # project metadata dir
    "_eval_config.yaml",  # legacy scorer artifact; tolerated if present
    ".run.lock",          # writer mutual-exclusion lock
}
```
(These values are copied verbatim from `hunyuan_ocr.runner:25-35`.) Keep the SPDX header.

- [ ] **Step 4: Run the test to verify it passes**

Run: `/opt/venv/bin/python -m pytest tests/test_validation.py -q`
Expected: `6 passed`.

- [ ] **Step 5: Validate**

Run: `/opt/venv/bin/python scripts/check_deps.py` → `P0 pyproject OK`.
Run: `/opt/venv/bin/python -m pytest -q` → 37 passed (31 + 6).

- [ ] **Step 6: Commit**

```bash
git add src/mineru_rocm/validation.py tests/test_validation.py
git commit -m "feat(p1b): port validation (pre-score prediction-dir checks); own ERROR_PREFIX/_OWN_ARTIFACTS; 6 tests"
```
(+ `Co-Authored-By` trailer.)

---

## Task 4: Port `canary.py` (byte-identical canary materialization)

**Files:**
- Create: `src/mineru_rocm/canary.py`
- Test: `tests/test_canary.py`

**Interfaces:**
- Consumes: nothing (pure stdlib).
- Produces (exact signatures, must match `hunyuan_ocr.canary`):
  - `class CanaryError(ValueError)`
  - `materialize(full_gt, manifest_path, out_path) -> str`  → returns the written SHA256 hex

- [ ] **Step 1: Write the failing test**

Create `tests/test_canary.py`:

```python
import hashlib
import json
import pytest
from mineru_rocm.canary import CanaryError, materialize


def _full_gt(tmp_path):
    pages = [
        {"page_info": {"image_path": "alpha.png"}, "data": 1},
        {"page_info": {"image_path": "beta.png"}, "data": 2},
        {"page_info": {"image_path": "gamma.png"}, "data": 3},
    ]
    gt = tmp_path / "full.json"
    gt.write_text(json.dumps(pages), encoding="utf-8")
    return gt, pages


def _manifest(pages_order, full_pages, expected_count=None, sha=None):
    # Compute the expected SHA exactly as materialize() will: subset in MANIFEST
    # order (not full-GT order). Skip pages absent from full_pages so the helper
    # doesn't KeyError on the missing-page test (materialize raises before the
    # SHA check there anyway).
    by_img = {p["page_info"]["image_path"]: p for p in full_pages}
    subset = [by_img[ip] for ip in pages_order if ip in by_img]
    blob = json.dumps(subset, ensure_ascii=False).encode("utf-8")
    return {
        "expected_count": expected_count if expected_count is not None else len(pages_order),
        "pages": [{"image_path": ip, "stem": ip.rsplit(".", 1)[0]} for ip in pages_order],
        "source_json_sha256": sha if sha is not None else hashlib.sha256(blob).hexdigest(),
    }


def test_materialize_round_trips_sha(tmp_path):
    gt, pages = _full_gt(tmp_path)
    manifest = _manifest(["gamma.png", "alpha.png"], pages)  # reordered subset
    mf = tmp_path / "manifest.json"; mf.write_text(json.dumps(manifest), encoding="utf-8")
    out = tmp_path / "out" / "canary.json"
    digest = materialize(gt, mf, out)
    assert out.exists()
    assert hashlib.sha256(out.read_bytes()).hexdigest() == digest
    # subset is in manifest order, compact serialization
    sub = json.loads(out.read_text())
    assert [p["page_info"]["image_path"] for p in sub] == ["gamma.png", "alpha.png"]


def test_materialize_sha_mismatch(tmp_path):
    gt, pages = _full_gt(tmp_path)
    manifest = _manifest(["alpha.png"], pages, sha="0" * 64)  # wrong sha
    mf = tmp_path / "manifest.json"; mf.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(CanaryError):
        materialize(gt, mf, tmp_path / "out.json")


def test_materialize_missing_page(tmp_path):
    gt, pages = _full_gt(tmp_path)
    manifest = _manifest(["alpha.png", "nope.png"], pages)
    mf = tmp_path / "manifest.json"; mf.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(CanaryError):
        materialize(gt, mf, tmp_path / "out.json")


def test_materialize_duplicate_paths(tmp_path):
    gt, pages = _full_gt(tmp_path)
    manifest = _manifest(["alpha.png", "alpha.png"], pages, expected_count=2)
    mf = tmp_path / "manifest.json"; mf.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(CanaryError):
        materialize(gt, mf, tmp_path / "out.json")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `/opt/venv/bin/python -m pytest tests/test_canary.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'mineru_rocm.canary'`.

- [ ] **Step 3: Port `hunyuan_ocr.canary` verbatim**

Read `/workspace/HunyuanOCR-ROCm/src/hunyuan_ocr/canary.py` and create `src/mineru_rocm/canary.py` as a verbatim port: `CanaryError`, `materialize` — identical bodies (`hashlib`, `json`, `pathlib`). **Zero adaptations** (pure stdlib). Keep the SPDX header.

- [ ] **Step 4: Run the test to verify it passes**

Run: `/opt/venv/bin/python -m pytest tests/test_canary.py -q`
Expected: `4 passed`.

- [ ] **Step 5: Validate**

Run: `/opt/venv/bin/python scripts/check_deps.py` → `P0 pyproject OK`.
Run: `/opt/venv/bin/python -m pytest -q` → 41 passed (37 + 4).

- [ ] **Step 6: Commit**

```bash
git add src/mineru_rocm/canary.py tests/test_canary.py
git commit -m "feat(p1b): port canary (byte-identical subset materialization) from Hunyuan; 4 tests"
```
(+ `Co-Authored-By` trailer.)

---

## Task 5: Port `scoring.py` + bundle `data/eval_config.yaml` + wire package-data

**Files:**
- Create: `src/mineru_rocm/scoring.py`
- Create: `src/mineru_rocm/data/eval_config.yaml` (verbatim from Hunyuan)
- Modify: `pyproject.toml` (add `[tool.setuptools.package-data]`)
- Test: `tests/test_scoring.py`

**Interfaces:**
- Consumes: `mineru_rocm.validation.validate_predictions` (Task 3).
- Produces (exact signatures, must match `hunyuan_ocr.scoring` except the documented adaptations):
  - `class ScoringError(RuntimeError)`
  - `DEFAULT_VENV_PYTHON`, `DEFAULT_OMNIDOCBENCH_REPO` (env-overridable; keep Hunyuan's defaults verbatim — they match this benchmark env)
  - `overall_score(metrics: dict) -> float | None`
  - `write_eval_config(*, gt_json: str, pred_dir: str, out_yaml: Path) -> None`
  - `run_scorer(*, omnidocbench_repo: str, config_yaml: str, venv_python: str | None = None) -> subprocess.CompletedProcess`
  - `parse_run_summary(result_dir: str | Path, save_name: str) -> dict`
  - `score_directory(*, gt_json: str, pred_dir: str, omnidocbench_repo: str | None = None, venv_python: str | None = None, skip_validation: bool = False, strict: bool = True) -> dict`
  - `format_score_table(label: str, metrics: dict) -> str`

- [ ] **Step 1: Write the failing test**

Create `tests/test_scoring.py` (all hermetic — the real scorer is monkeypatched; no OmniDocBench venv needed):

```python
import json
import subprocess
from pathlib import Path
import pytest
from mineru_rocm import scoring
from mineru_rocm.scoring import (
    ScoringError, overall_score, write_eval_config, parse_run_summary,
    score_directory, format_score_table,
)


def test_overall_score_formula():
    # ((1-text)*100 + cdm*100 + teds*100)/3
    assert overall_score({"text_edit_dist": 0.05, "formula_cdm": 0.95, "table_teds": 0.90}) == \
        pytest.approx(((1 - 0.05) * 100 + 0.95 * 100 + 0.90 * 100) / 3)


def test_overall_score_none_when_metric_missing():
    assert overall_score({"text_edit_dist": 0.05, "formula_cdm": None, "table_teds": 0.90}) is None


def test_write_eval_config_substitutes_paths_and_keeps_metrics(tmp_path):
    out = tmp_path / "cfg" / "_eval_config.yaml"
    write_eval_config(gt_json="/gt/full.json", pred_dir="/pred/vlm", out_yaml=out)
    assert out.is_file()
    import yaml
    cfg = yaml.safe_load(out.read_text())
    assert cfg["end2end_eval"]["dataset"]["ground_truth"]["data_path"] == "/gt/full.json"
    assert cfg["end2end_eval"]["dataset"]["prediction"]["data_path"] == "/pred/vlm"
    # metric structure intact (model-agnostic)
    assert cfg["end2end_eval"]["metrics"]["display_formula"]["metric"] == ["Edit_dist", "CDM"]
    assert cfg["end2end_eval"]["metrics"]["table"]["metric"] == ["TEDS", "Edit_dist"]


def _write_run_summary(result_dir, save_name, *, text=0.0566, cdm=0.9755, teds=0.8204, order=0.1240):
    result_dir = Path(result_dir); result_dir.mkdir(parents=True, exist_ok=True)
    ms = {
        "text_block_Edit_dist": {"raw": text},
        "display_formula_CDM": {"raw": cdm},
        "table_TEDS": {"raw": teds},
        "reading_order_Edit_dist": {"raw": order},
    }
    blob = {"notebook_metric_summary": {"metrics": ms}}
    (result_dir / f"{save_name}_run_summary.json").write_text(json.dumps(blob), encoding="utf-8")


def test_parse_run_summary(tmp_path):
    _write_run_summary(tmp_path / "result", "mypred_quick_match")
    m = parse_run_summary(tmp_path / "result", "mypred_quick_match")
    assert m["text_edit_dist"] == 0.0566 and m["formula_cdm"] == 0.9755
    assert m["table_teds"] == 0.8204 and m["reading_order_edit"] == 0.1240
    assert m["overall"] == pytest.approx(((1 - 0.0566) * 100 + 0.9755 * 100 + 0.8204 * 100) / 3)


def test_score_directory_success(monkeypatch, tmp_path):
    # hermetic: don't call the real scorer; fake a clean result
    pred = tmp_path / "mypred"; pred.mkdir()
    (pred / "a.md").write_text("# a")
    gt = tmp_path / "gt.json"; gt.write_text(json.dumps([{"page_info": {"image_path": "a.png"}}]), encoding="utf-8")

    def fake_run(*, omnidocbench_repo, config_yaml, venv_python=None):
        save = f"{pred.name}_quick_match"
        _write_run_summary(Path(omnidocbench_repo) / "result", save)
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(scoring, "run_scorer", fake_run)
    out = score_directory(gt_json=str(gt), pred_dir=str(pred), omnidocbench_repo=str(tmp_path / "repo"))
    assert out["validation_report"].ok
    assert out["metrics"]["overall"] is not None


def test_score_directory_validation_failure(tmp_path):
    pred = tmp_path / "mypred"; pred.mkdir()  # empty -> missing prediction
    gt = tmp_path / "gt.json"; gt.write_text(json.dumps([{"page_info": {"image_path": "a.png"}}]), encoding="utf-8")
    with pytest.raises(ScoringError):
        score_directory(gt_json=str(gt), pred_dir=str(pred), omnidocbench_repo=str(tmp_path))


def test_score_directory_scorer_nonzero(monkeypatch, tmp_path):
    pred = tmp_path / "mypred"; pred.mkdir(); (pred / "a.md").write_text("# a")
    gt = tmp_path / "gt.json"; gt.write_text(json.dumps([{"page_info": {"image_path": "a.png"}}]), encoding="utf-8")
    monkeypatch.setattr(scoring, "run_scorer",
                        lambda **kw: subprocess.CompletedProcess(args=[], returncode=2, stdout="", stderr="boom"))
    with pytest.raises(ScoringError):
        score_directory(gt_json=str(gt), pred_dir=str(pred), omnidocbench_repo=str(tmp_path))


def test_format_score_table_renders_overall():
    m = {"overall": 95.56, "text_edit_dist": 0.0566, "formula_cdm": 0.9755,
         "table_teds": 0.8204, "reading_order_edit": 0.1240}
    s = format_score_table("vlm-vllm", m)
    assert "95.56" in s and "OmniDocBench v1.6" in s
```

- [ ] **Step 2: Run it to verify it fails**

Run: `/opt/venv/bin/python -m pytest tests/test_scoring.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'mineru_rocm.scoring'`.

- [ ] **Step 3: Create the bundled template `src/mineru_rocm/data/eval_config.yaml` (verbatim)**

Create `src/mineru_rocm/data/eval_config.yaml` with EXACTLY this content (the OmniDocBench v1.6 metric template — model-agnostic; the per-model distinction is the `prediction.data_path` substituted at score time):

```yaml
end2end_eval:
  metrics:
    text_block: {metric: [Edit_dist]}
    display_formula: {metric: [Edit_dist, CDM], cdm_workers: 13}
    table: {metric: [TEDS, Edit_dist], teds_workers: 13}
    reading_order: {metric: [Edit_dist]}
  dataset:
    dataset_name: end2end_dataset
    ground_truth: {data_path: REPLACE_WITH_GT_JSON}
    prediction: {data_path: REPLACE_WITH_PREDICTIONS_DIR}
    match_method: quick_match
    match_workers: 13
    quick_match_truncated_timeout_sec: 300
    match_timeout_sec: 420
    timeout_fallback_max_chunk_span: 10
    timeout_fallback_order_penalty: 0.10
```

(Do NOT add `data/__init__.py` — it is resource data, not a subpackage; matches Hunyuan.)

- [ ] **Step 4: Wire package-data in `pyproject.toml`**

Add after the existing `[tool.setuptools.packages.find]` block:

```toml
[tool.setuptools.package-data]
mineru_rocm = ["data/*.yaml"]
```

- [ ] **Step 5: Port `hunyuan_ocr.scoring` with THREE adaptations**

Read `/workspace/HunyuanOCR-ROCm/src/hunyuan_ocr/scoring.py` and create `src/mineru_rocm/scoring.py`. Port `ScoringError`, `DEFAULT_VENV_PYTHON`, `DEFAULT_OMNIDOCBENCH_REPO`, `overall_score`, `run_scorer`, `parse_run_summary`, `score_directory`, `format_score_table` verbatim. Apply exactly these three adaptations:

1. **Lazy `yaml`:** remove the top-level `import yaml`; instead import it inside `write_eval_config` (`def write_eval_config(...): import yaml; ...`). Keeps `dependencies=[]`.
2. **Drop the repo-relative fallback:** remove `_REPO_TEMPLATE` and the `try/except` in `_load_eval_template()`. The function becomes:
   ```python
   def _load_eval_template() -> str:
       from importlib.resources import files
       return (files("mineru_rocm") / "data" / "eval_config.yaml").read_text(encoding="utf-8")
   ```
3. **Internal import:** in `score_directory`, change `from hunyuan_ocr.validation import validate_predictions` → `from mineru_rocm.validation import validate_predictions`.

Keep `DEFAULT_VENV_PYTHON`/`DEFAULT_OMNIDOCBENCH_REPO` and their `os.environ.get(...)` defaults verbatim (they match this env). Keep the SPDX header.

- [ ] **Step 6: Run the test to verify it passes**

Run: `/opt/venv/bin/pip install -e . -q` (re-install to pick up the new module + package-data).
Run: `/opt/venv/bin/python -m pytest tests/test_scoring.py -q`
Expected: `8 passed`.

- [ ] **Step 7: Validate — decoupling proof + engine contract + full suite**

Run: `/opt/venv/bin/python scripts/check_deps.py` → `P0 pyproject OK` (the no-`omnidocbench_amd` sweep now covers the 5 new modules too).
Run: `/opt/venv/bin/python -c "import mineru_rocm.scoring as s; import sys; print('engine:', 'omnidocbench_amd' in sys.modules); print('overall:', s.overall_score({'text_edit_dist':0.0566,'formula_cdm':0.9755,'table_teds':0.8204}))"` → `engine: False` and `overall:` ≈ `91.31`.
Run: `/opt/venv/bin/python -c "from importlib.resources import files; print((files('mineru_rocm')/'data'/'eval_config.yaml').read_text()[:20])"` → prints the first chars of the template (proves package-data wiring).
Run: `/opt/venv/bin/python adapter/run_adapter.py --help | grep -c -- '--'` → `7` (engine CLI byte-identical; P1b didn't touch it).
Run: `/opt/venv/bin/python -m pytest -q` → 49 passed (41 + 8).

- [ ] **Step 8: Commit**

```bash
git add src/mineru_rocm/scoring.py src/mineru_rocm/data/eval_config.yaml pyproject.toml tests/test_scoring.py
git commit -m "feat(p1b): port scoring (OmniDocBench eval-config + scorer + parser) from Hunyuan; bundle eval_config.yaml; lazy yaml; 8 tests"
```
(+ `Co-Authored-By` trailer.)

---

## Definition of Done (P1b)

- [ ] `import mineru_rocm.omnidocbench, mineru_rocm.preflight, mineru_rocm.validation, mineru_rocm.canary, mineru_rocm.scoring` succeeds with NO engine pulled into `sys.modules`.
- [ ] `src/mineru_rocm/` still contains zero `omnidocbench_amd` references (asserted by `scripts/check_deps.py`); `dependencies == []` unchanged.
- [ ] `python -m pytest -q` green: 49 total (20 from P1a + 2 omnidocbench + 9 preflight + 6 validation + 4 canary + 8 scoring).
- [ ] `scoring.overall_score` reproduces the v1.6 formula `((1-text)*100 + cdm*100 + teds*100)/3`; `run_scorer` issues `pdf_validation.py --config <cfg>` (the same scorer that produced 95.56/86.48).
- [ ] The bundled `data/eval_config.yaml` loads via `importlib.resources` (package-data wired).
- [ ] `adapter/run_adapter.py --help` still shows the unchanged 7-flag engine CLI (P1b untouched the dispatcher/backends/types).
- [ ] No `RunSummary`/`PageStatus`/`infer_page`/`_run_stats.json` change; no score/behavior change.

**Deferred (NOT P1b):** a real end-to-end score smoke (run `score_directory` against an actual scored prediction dir + the OmniDocBench venv) — needs a prediction directory + the pinned scorer; belongs to the reproducibility/results phase (P3) or a follow-up. The unit tests prove the formula, config-writing, summary-parsing, and control flow via monkeypatched `run_scorer`.

## Follow-on sub-phases (separate plans)

- **P1c** — `runner.py` (atomic writes, resumability, `run_manifest.json` conservation laws, `.run.lock`; the resume-`ok_pages` fix) + `endpoint_pool.py` (VLM http-client track) + the in-process pipeline driver and the VLM http-client driver. **P1c's `runner` must `from mineru_rocm.validation import ERROR_PREFIX, _OWN_ARTIFACTS`** (the contract localized in P1b Task 3). Also resolve the `dispatcher.py:41,45` `skipped` dead var here.
- **P1d** — `cli.py` (`mineru-rocm doctor|validate|predict|score|canary|manifest verify`) + `scripts/check_repo.py` (lock↔README consistency + a `pip install -e .` smoke as the PEP 639 regression guard + structural `tomllib`/AST no-engine scan).
