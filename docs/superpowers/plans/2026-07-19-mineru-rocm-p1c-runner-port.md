# MinerU-ROCm P1c — Runner Port (Atomic Writes + Run Manifest + RunLock) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port `hunyuan_ocr.runner` into `src/mineru_rocm/runner.py` — the prediction-integrity primitives that **structurally fix the resume-`ok_pages` fragility**: atomic `.md` writes (`.partial` → fsync → `os.replace`), structured per-page error records, resumability that skips only genuinely-complete pages, the conservation-checked `run_manifest.json`, and `fcntl` writer mutual-exclusion. No GPU/model deps — pure filesystem + stdlib.

**Architecture:** A faithful, near-verbatim port of `hunyuan_ocr.runner` (the reference is importable as `hunyuan-ocr 0.1.1` in `/opt/venv`, source at `/workspace/HunyuanOCR-ROCm/src/hunyuan_ocr/runner.py`, tests at `/workspace/HunyuanOCR-ROCm/tests/test_runner.py`). The runner **coexists** with `mineru_rocm.dispatcher` (it does NOT replace it): `dispatcher.run_adapter` stays the omnidocbench-amd engine-subprocess path (writes the 8-key `_run_stats.json`); the runner is the new robust path the future drivers (P1c.2) will use (writes `run_manifest.json`). P1c does NOT touch `dispatcher.py`, the backends, or the engine CLI. One adaptation: runner imports `ERROR_PREFIX` from `mineru_rocm.validation` (P1b localized it there) instead of defining it.

**Tech Stack:** Python 3.11+, stdlib (`json`/`os`/`subprocess`/`sys`/`time`/`pathlib`/`dataclasses`; lazy `fcntl`/`datetime`/`platform`/`socket`), pytest.

## Global Constraints

(From the approved spec `docs/superpowers/specs/2026-07-19-mineru-rocm-standalone-port-design.md` §3.2; every task implicitly includes these.)

- **No behavior / no score change.** P1c does NOT modify `dispatcher.py`, `adapter/run_adapter.py`, `backends/*`, or `types.py`. The engine subprocess contract (8-key `_run_stats.json`, `adapter/run_adapter.py --help`) MUST stay byte-identical. The runner is ADDITIVE — a parallel robust path.
- **Atomic + resumable** (spec §3.2): one `<stem>.md` per page written atomically (`.partial` → fsync → `os.replace`); resumability skips only genuinely-complete pages (non-empty, not `ERROR:`, no `_errors/<stem>.json`); `.run.lock` prevents two writers; the manifest is authoritative, not reconstructed from disk.
- **Core stays GPU-free / platform-free.** `runner.py` imports with **no** `torch`/`mineru`/`omnidocbench_amd`/`transformers`/`openai` at module top level. (`fcntl` is POSIX, imported lazily inside `RunLock.acquire`; `_env_versions` best-effort-imports torch/transformers/vllm inside the function, never raising.)
- **Verbatim port.** Every function/class body MUST match `hunyuan_ocr.runner` byte-for-byte except the ONE documented adaptation (the `ERROR_PREFIX` import). The implementer READS the reference source (`/workspace/HunyuanOCR-ROCm/src/hunyuan_ocr/runner.py`, or `/opt/venv/bin/python -c "import hunyuan_ocr.runner as m, inspect; print(inspect.getsource(m))"`) and ports it.
- **One concern per commit; commit after every task's validation passes.** Branch: `feat/p1c-runner` (off `main` @ `ed1837e`, which has P1a+P1b+the ci-fix).
- **Validation environment:** `/opt/venv/bin/python` and `/opt/venv/bin/pip` ONLY (py3.12; has `hunyuan-ocr`, `mineru-rocm` editable, `pytest`, `PyYAML`). Linux (so `fcntl`/`RunLock` are testable).
- **Standing rule for ports:** the module docstring MAY differ from the reference (retitled to mineru_rocm; strip any Hunyuan-local absolute paths). Function/class bodies stay verbatim.

---

## File Structure (P1c scope)

| File | Action | Responsibility |
|---|---|---|
| `src/mineru_rocm/runner.py` | Create (port from `hunyuan_ocr.runner`, grown across Tasks 1-3) | Prediction-integrity primitives: atomic writes, error records, completion/resume, run-manifest + conservation, RunLock |
| `tests/test_runner_integrity.py` | Create (port from Hunyuan `test_runner.py`) | TDD for the page-level primitives (Task 1) |
| `tests/test_runner_manifest.py` | Create (port from Hunyuan `test_runner.py`) | TDD for manifest schema + conservation laws (Task 2) |
| `tests/test_runner_lock.py` | Create (new — Hunyuan has none) | TDD for RunLock acquire/release/contention (Task 3) |

Out of P1c (separate plan, "P1c.2"): `endpoint_pool.py` + the in-process pipeline driver + the VLM http-client driver — they consume this runner. `dispatcher.py`'s `skipped` dead var is NOT resolved here (runner is additive and never touches dispatcher; `skipped` stays harmless dead code in the frozen engine path — leave it).

---

## Task 1: Page-level integrity primitives (atomic writes + error records + completion/resume)

**Files:**
- Create: `src/mineru_rocm/runner.py` (module header + top imports + the validation-import adaptation + the page-level functions)
- Test: `tests/test_runner_integrity.py`

**Interfaces:**
- Consumes: `mineru_rocm.validation.ERROR_PREFIX` (P1b).
- Produces (exact signatures, must match `hunyuan_ocr.runner`):
  - `_partial_of(path: Path) -> Path`
  - `_fsync_dir(path: Path) -> None`
  - `write_atomic(path: Path, content: str) -> None`
  - `_error_path(pred_dir, stem: str, ext: str = ".md") -> Path`
  - `record_error(pred_dir, stem: str, *, image_path, backend, endpoint, exc, attempt: int, ts: float | None = None) -> None`
  - `commit_success(pred_dir, stem: str, md: str, *, ext: str = ".md") -> Path`
  - `is_complete(pred_dir, stem: str, ext: str = ".md") -> bool`
  - `page_status(pred_dir, stem: str, ext: str = ".md") -> str`
  - `select_todo(items, pred_dir, *, overwrite: bool = False, retry_failed: bool = False, ext: str = ".md")`
  - `detect_stem_conflicts(image_paths) -> list`
  - `decide_run_status(final_failed: int, final_pending: int, worker_errors: int = 0, crashed: int = 0) -> str`
  - `aggregate_errors(pred_dir, out_name: str = "_errors.jsonl") -> Path`

- [ ] **Step 1: Write the failing test**

Create `tests/test_runner_integrity.py` by porting these 14 test functions VERBATIM from `/workspace/HunyuanOCR-ROCm/tests/test_runner.py` (lines 1-153), with the ONE change: the import line `from hunyuan_ocr import runner` → `from mineru_rocm import runner`. Port these named functions exactly (bodies unchanged):
`test_write_atomic_creates_final_and_no_partial`, `test_write_atomic_is_atomic_on_error`, `test_write_atomic_creates_parent_dir`, `test_record_error_writes_structured_record`, `test_commit_success_writes_md_and_clears_stale_error`, `test_is_complete_false_for_missing_empty_error_partial`, `test_is_complete_false_if_partial_only`, `test_page_status_states`, `test_select_todo_default_resumes_and_retries_failed`, `test_select_todo_retry_failed_only`, `test_select_todo_overwrite`, `test_detect_stem_conflicts`, `test_decide_run_status`, `test_aggregate_errors_concatenates_records`. Keep the file header `import json`, `import pytest`, `from mineru_rocm import runner`.

- [ ] **Step 2: Run it to verify it fails**

Run: `/opt/venv/bin/python -m pytest tests/test_runner_integrity.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'mineru_rocm.runner'`.

- [ ] **Step 3: Port the page-level primitives into `src/mineru_rocm/runner.py`**

Create `src/mineru_rocm/runner.py`. Port from `/workspace/HunyuanOCR-ROCm/src/hunyuan_ocr/runner.py`:
- The module docstring (lines 3-13), retitled to mineru_rocm context (strip any Hunyuan-local paths — there are none here, so a light retitle suffices). Keep the SPDX header.
- The top import block (lines 15-23): `from __future__ import annotations`, `import json`, `import os`, `import time`, `from pathlib import Path`. (Task 1's primitives use json/os/time/pathlib only; Task 2 will add `import subprocess, sys`, Task 3 `from dataclasses import asdict, dataclass` — add imports as each task's code needs them, so no unused imports linger.)
- **The ONE adaptation:** replace Hunyuan's local constant block (lines 25-35: `ERROR_PREFIX = "ERROR:"` + the `_OWN_ARTIFACTS = {...}` definition) with a single import:
  ```python
  from mineru_rocm.validation import ERROR_PREFIX  # localized in P1b; runner uses it in is_complete()
  ```
  (Do NOT define `_OWN_ARTIFACTS` here — it is unused in `runner.py` and already owned by `mineru_rocm.validation` where it IS used. Importing only `ERROR_PREFIX` avoids an unused import.)
- Port these functions VERBATIM (bodies byte-identical to the reference): `_partial_of` (38-39), `_fsync_dir` (42-57), `write_atomic` (60-86), `_error_path` (89-90), `record_error` (93-112), `commit_success` (115-128), `is_complete` (131-147), `page_status` (150-156), `select_todo` (159-183), `detect_stem_conflicts` (186-192), `decide_run_status` (195-199), `aggregate_errors` (202-216).

- [ ] **Step 4: Run the test to verify it passes**

Run: `/opt/venv/bin/python -m pytest tests/test_runner_integrity.py -q`
Expected: `14 passed`.

- [ ] **Step 5: Validate (no engine leak + full suite)**

Run: `/opt/venv/bin/python scripts/check_deps.py` → `P0 pyproject OK`.
Run: `/opt/venv/bin/python -c "import mineru_rocm.runner; import sys; print('engine:', 'omnidocbench_amd' in sys.modules)"` → `engine: False`.
Run: `/opt/venv/bin/python -m pytest -q` → 63 passed (49 + 14).

- [ ] **Step 6: Commit**

```bash
git add src/mineru_rocm/runner.py tests/test_runner_integrity.py
git commit -m "feat(p1c): port runner page-integrity primitives (atomic write, error records, completion/resume) from Hunyuan; 14 tests"
```
(End the message body with a blank line + `Co-Authored-By: Claude <noreply@anthropic.com>`.)

---

## Task 2: Run manifest + validation (conservation laws)

**Files:**
- Modify: `src/mineru_rocm/runner.py` (append the manifest section)
- Test: `tests/test_runner_manifest.py`

**Interfaces:**
- Consumes: `write_atomic`, `iso_utc`/`_git_head`/etc. (this task); `mineru_rocm.runner` primitives (Task 1).
- Produces (exact signatures, must match `hunyuan_ocr.runner`):
  - Module constants: `MANIFEST_SCHEMA_VERSION = 2`, `SUPPORTED_SCHEMA_VERSIONS = (1, 2)`, `KNOWN_RUN_STATUSES`, `REQUIRED_RUN_COUNTS`, `REQUIRED_FINAL_STATE`, `RESERVED_MANIFEST_KEYS`, `_SECRET_FLAGS`
  - `_is_nonneg_int(val) -> bool`, `_parse_iso(ts) -> bool`, `safe_argv(argv=None) -> list[str]`
  - `_repo_root() -> Path`, `_git_head(repo=None) -> str | None`, `iso_utc(epoch=None) -> str`, `_platform_info() -> dict`, `_env_versions() -> dict`
  - `validate_manifest(m: dict) -> list[str]`
  - `write_run_manifest(pred_dir, *, backend, model, run_counts, final_state, model_revision=None, backend_provenance=None, command=None, ports=None, gpu_ids=None, max_pixels=None, max_tokens=None, status="ok", extra=None) -> Path`

- [ ] **Step 1: Write the failing test**

Create `tests/test_runner_manifest.py` by porting VERBATIM from `/workspace/HunyuanOCR-ROCm/tests/test_runner.py` (lines 156-389), with the import `from hunyuan_ocr import runner` → `from mineru_rocm import runner`. Port the `_valid_manifest(**overrides)` helper (lines 278-290) AND these 20 test functions (bodies unchanged): `test_safe_argv_redacts_secrets`, `test_safe_argv_no_false_positive_on_monkey`, `test_write_run_manifest_structure_and_no_secret`, `test_write_run_manifest_extra_is_namespaced_and_collision_rejected`, `test_manifest_invariants_hold`, `test_manifest_invariants_violated`, `test_manifest_works_without_torch`, `test_validate_manifest_accepts_valid`, `test_validate_manifest_rejects_non_object`, `test_validate_manifest_unknown_schema_version`, `test_validate_manifest_v1_read_compat`, `test_validate_manifest_missing_run_counts`, `test_validate_manifest_missing_single_count`, `test_validate_manifest_rejects_string_count`, `test_validate_manifest_rejects_float_count`, `test_validate_manifest_rejects_bool_count`, `test_validate_manifest_rejects_empty_backend_or_model`, `test_validate_manifest_rejects_unparseable_timestamp`, `test_validate_manifest_ok_with_failed_is_invalid`, `test_validate_manifest_conservation_with_interrupted`. Keep `import json`, `import pytest`, `from mineru_rocm import runner`.

- [ ] **Step 2: Run it to verify it fails**

Run: `/opt/venv/bin/python -m pytest tests/test_runner_manifest.py -q`
Expected: FAIL — `AttributeError: module 'mineru_rocm.runner' has no attribute 'MANIFEST_SCHEMA_VERSION'` (or `safe_argv`).

- [ ] **Step 3: Port the manifest section into `src/mineru_rocm/runner.py`**

Append to `src/mineru_rocm/runner.py` (after the Task 1 primitives). Add `import subprocess` and `import sys` to the top import block (these are used by `safe_argv`/`_git_head`). Port VERBATIM from the reference (lines 219-547): the `_SECRET_FLAGS` set, `MANIFEST_SCHEMA_VERSION`, `SUPPORTED_SCHEMA_VERSIONS`, `KNOWN_RUN_STATUSES`, `REQUIRED_RUN_COUNTS`, `REQUIRED_FINAL_STATE`, `RESERVED_MANIFEST_KEYS`, `_is_nonneg_int`, `_parse_iso`, `safe_argv`, `_repo_root`, `_git_head`, `iso_utc`, `_platform_info`, `_env_versions`, `validate_manifest`, `write_run_manifest`. **Zero adaptations** (these are pure stdlib + best-effort optional imports). `_repo_root()` uses `Path(__file__)` which resolves correctly under `src/mineru_rocm/runner.py` (walks up to the MinerU `.git`).

- [ ] **Step 4: Run the test to verify it passes**

Run: `/opt/venv/bin/python -m pytest tests/test_runner_manifest.py -q`
Expected: `20 passed`.

- [ ] **Step 5: Validate**

Run: `/opt/venv/bin/python scripts/check_deps.py` → `P0 pyproject OK`.
Run: `/opt/venv/bin/python -c "import mineru_rocm.runner as r; print('engine ok; manifest validates empty:', r.validate_manifest({}) != [])"` → prints `True` (non-list of errors for `{}` is truthy; confirms the function is wired).
Run: `/opt/venv/bin/python -m pytest -q` → 83 passed (63 + 20).

- [ ] **Step 6: Commit**

```bash
git add src/mineru_rocm/runner.py tests/test_runner_manifest.py
git commit -m "feat(p1c): port runner run-manifest + validate_manifest (conservation laws) from Hunyuan; 20 tests"
```
(+ `Co-Authored-By` trailer.)

---

## Task 3: RunLock (fcntl writer mutual-exclusion)

**Files:**
- Modify: `src/mineru_rocm/runner.py` (append the RunLock section)
- Test: `tests/test_runner_lock.py` (NEW — Hunyuan ships no lock tests)

**Interfaces:**
- Consumes: `iso_utc` (Task 2).
- Produces (exact signatures, must match `hunyuan_ocr.runner`):
  - `class RunLockHeld(RuntimeError)`
  - `@dataclass class LockInfo` — `pid: int`, `host: str`, `started_iso: str`, `command: str`; method `is_alive() -> bool`
  - `class RunLock` — `__init__(self, pred_dir, command=None)`, `acquire() -> RunLock`, `release() -> None`, `__enter__`/`__exit__`; attr `LOCK_NAME = ".run.lock"`
  - `_hostname() -> str`
  - `acquire_run_lock(pred_dir, command=None) -> RunLock`

- [ ] **Step 1: Write the failing test (NEW — author these)**

Create `tests/test_runner_lock.py`:

```python
import json
import pytest
from mineru_rocm import runner
from mineru_rocm.runner import RunLock, RunLockHeld, acquire_run_lock


def test_run_lock_acquire_and_release(tmp_path):
    lock = RunLock(tmp_path)
    lock.acquire()
    assert (tmp_path / ".run.lock").is_file()
    # the lock file records holder metadata as JSON
    info = json.loads((tmp_path / ".run.lock").read_text("utf-8"))
    assert {"pid", "host", "started_iso", "command"} <= set(info)
    lock.release()
    assert not (tmp_path / ".run.lock").exists()


def test_run_lock_context_manager(tmp_path):
    with acquire_run_lock(tmp_path, command=["predict", "--x", "1"]) as lock:
        assert (tmp_path / ".run.lock").is_file()
    assert not (tmp_path / ".run.lock").exists()  # released on exit


def test_run_lock_second_acquire_raises(tmp_path):
    with acquire_run_lock(tmp_path):
        second = RunLock(tmp_path)
        with pytest.raises(RunLockHeld):
            second.acquire()


def test_run_lock_reacquire_after_release(tmp_path):
    with acquire_run_lock(tmp_path):
        pass
    # after release, a fresh acquire must succeed (flock auto-released)
    with acquire_run_lock(tmp_path):
        assert (tmp_path / ".run.lock").is_file()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `/opt/venv/bin/python -m pytest tests/test_runner_lock.py -q`
Expected: FAIL — `ImportError: cannot import name 'RunLock' from 'mineru_rocm.runner'`.

- [ ] **Step 3: Port the RunLock section into `src/mineru_rocm/runner.py`**

Add `from dataclasses import asdict, dataclass` to the top import block (used by `LockInfo`'s `@dataclass` decorator and `RunLock.acquire`'s `asdict(info)`). Append VERBATIM from the reference (lines 550-684): `RunLockHeld`, `LockInfo`, `RunLock`, `_hostname`, `acquire_run_lock`. **Zero adaptations** (`fcntl` is imported lazily inside `acquire`/`release`; `os`/`json` already imported). Keep the section's comments (they document the flock auto-release semantics).

- [ ] **Step 4: Run the test to verify it passes**

Run: `/opt/venv/bin/python -m pytest tests/test_runner_lock.py -q`
Expected: `4 passed`.

- [ ] **Step 5: Validate — decoupling proof + engine contract + full runner parity + full suite**

Run: `/opt/venv/bin/python scripts/check_deps.py` → `P0 pyproject OK`.
Run: `/opt/venv/bin/python -c "import mineru_rocm.runner; import sys; print('engine:', 'omnidocbench_amd' in sys.modules)"` → `engine: False`.
Run: `/opt/venv/bin/python -c "import inspect, hunyuan_ocr.runner as h, mineru_rocm.runner as m; print('bodies match (modulo the ERROR_PREFIX import):', inspect.getsource(h)==inspect.getsource(m) or 'see diff')"` — then eyeball: the ONLY differences vs the reference should be (a) the module docstring retitle, (b) `from mineru_rocm.validation import ERROR_PREFIX` replacing the local `ERROR_PREFIX`/`_OWN_ARTIFACTS` block, (c) the import-line grouping. No function body differs.
Run: `/opt/venv/bin/python adapter/run_adapter.py --help | grep -E '^\s+--' | wc -l` → `7` (engine CLI byte-identical — P1c didn't touch it).
Run: `/opt/venv/bin/python -m pytest -q` → 87 passed (83 + 4).

- [ ] **Step 6: Commit**

```bash
git add src/mineru_rocm/runner.py tests/test_runner_lock.py
git commit -m "feat(p1c): port runner RunLock (fcntl writer mutual-exclusion) from Hunyuan; 4 new lock tests"
```
(+ `Co-Authored-By` trailer.)

---

## Definition of Done (P1c — runner)

- [ ] `import mineru_rocm.runner` succeeds with NO engine pulled into `sys.modules` (stdlib + lazy `fcntl`/`datetime`/`platform`/`socket`).
- [ ] `src/mineru_rocm/runner.py` contains zero `omnidocbench_amd` references; `ERROR_PREFIX` imported from `mineru_rocm.validation` (not redefined).
- [ ] `python -m pytest -q` green: 87 total (49 from P1a+P1b + 14 integrity + 20 manifest + 4 lock).
- [ ] Runner bodies are byte-identical to `hunyuan_ocr.runner` modulo the documented adaptation + docstring retitle (verifiable by `inspect.getsource` diff).
- [ ] The 3 conservation laws hold: `attempted == succeeded + failed + interrupted`; `expected == attempted + skipped`; `expected == complete + failed + pending`; `status=="ok"` ⇒ `failed==0 and pending==0` (asserted by `test_manifest_invariants_*`).
- [ ] `RunLock` provides mutual exclusion: a second `acquire()` on a held dir raises `RunLockHeld`; release allows reacquire.
- [ ] **Engine contract untouched**: `adapter/run_adapter.py --help` shows the unchanged 7-flag CLI; `dispatcher.run_adapter` still writes the 8-key `_run_stats.json` (P1c is purely additive).

## Follow-on (separate plan — "P1c.2")

- `endpoint_pool.py` (circuit-breaking OpenAI pool, ~210 LOC, standalone + CPU-testable with injected clock/health-check).
- The in-process **pipeline driver** (orchestrates preflight → `omnidocbench.iter_page_images` → `runner.select_todo` → `backends.pipeline.infer_page` → `runner.commit_success` → `write_run_manifest`). Open question for that plan: in-process loop (simple) vs Hunyuan's multiprocess GPU sharding — recommend in-process (MinerU's `backends.pipeline.infer_page` loads the pipeline once, in-process).
- The **VLM http-client driver** (orchestrates `endpoint_pool` → `backends.vlm.infer_page` → `runner`). Adaptation: Hunyuan's drivers call `infer_one(client, img, prompt, ...)`; MinerU's backends expose `infer_page(img, platform, cfg)` — the MinerU drivers adapt the call signature. Both drivers use injected callables so they're CPU-testable with fakes (P1b scoring's monkeypatch precedent).
- `cli.py` + `scripts/check_repo.py` (P1d): the `mineru-rocm predict|score|...` CLI invokes the drivers; `check_repo.py` adds the `pip install -e .` smoke + structural `tomllib`/AST no-engine scan (the P1a/P1b carry-forward).
