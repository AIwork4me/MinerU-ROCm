# MinerU-ROCm P1d — CLI + check_repo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the user-facing `mineru-rocm` CLI (`cli.py`: `doctor|validate|predict|score|canary|manifest verify`) + `scripts/check_repo.py` (repo-consistency gate wired into CI). This is the FINAL P1 sub-phase — it exposes the P1a–P1c.2 library as a runnable command + adds the regression gate that would have caught the P1a PEP 639 install break.

**Architecture:** The CLI is a thin entry surface mirroring Hunyuan's `hunyuan_ocr/cli.py` structure (argparse subparsers → dispatch dict → per-subcommand handlers), adapted to MinerU's modules: `predict`→`driver.run`, `score`→`scoring.score_directory`, `validate`→`validation.validate_predictions`, `canary materialize`→`canary.materialize`, `manifest verify`→`runner.validate_manifest`, `doctor`→env introspection. `check_repo.py` adapts Hunyuan's repo-consistency checks + adds two MinerU-specific ones (a `pip install -e .` smoke as the PEP 639 guard; an AST-based no-engine-import scan superseding `check_deps.py`'s substring check) and is wired into `ci.yml`. A small Task-1 driver hardening threads an explicit `command=` into the manifest (so the recorded command reflects the user-facing CLI) and adds the two missing resume-mode tests.

**Tech Stack:** Python 3.11+, stdlib (`argparse`/`ast`/`subprocess`/`importlib`/`json`/`tomllib`/`pathlib`), PyYAML (core dep), pytest. Heavy deps (`mineru`/`torch`/`vllm`) stay lazy inside the backends/`doctor`.

## Global Constraints

(From the approved spec `docs/superpowers/specs/2026-07-19-mineru-rocm-standalone-port-design.md` §3.2; every task implicitly includes these.)

- **No behavior / no score change.** The CLI is a NEW entry surface. `dispatcher.py`/`adapter/`/`backends/`/`runner.py`/`types.py` MUST stay byte-identical. `driver.py` is touched ONLY for the Task-1 hardening (additive `command=` param + 2 tests + a docstring wording fix — no behavior change to existing paths). Engine subprocess contract (8-key `_run_stats.json`, 7-flag adapter CLI) byte-identical.
- **`pip install mineru-rocm` runs on plain CPU** (spec §3.2): `cli.py` + `check_repo.py` import with NO `torch`/`mineru`/`mineru_vl_utils`/`omnidocbench_amd`/`openai`/`requests` at module top level. Heavy deps are imported lazily inside the subcommand that needs them (`doctor` imports `mineru`/`torch`/`vllm` best-effort inside try/except; `predict`'s heavy work is in `driver.run` which lazy-imports the backends).
- **check_repo cross-checks README ↔ lock; CI fails on drift** (spec §3.2): the structure/SPDX/script-ref checks land in P1d; the README↔lock *value* cross-check **defers to P3** (the lock is a `not_recorded` skeleton today — cross-checking would be meaningless; documented in Follow-on).
- **One concern per commit; commit after every task's validation passes.** Branch: `feat/p1d-cli` (off `main` @ `8ae58cb`).
- **Validation environment:** `/opt/venv/bin/python` and `/opt/venv/bin/pip` ONLY (py3.12; `mineru-rocm` editable, `pytest`, `PyYAML`).

---

## File Structure (P1d scope)

| File | Action | Responsibility |
|---|---|---|
| `src/mineru_rocm/driver.py` | Modify (Task 1) | Add additive `command=` param to `_orchestrate`/`run` (manifest provenance) + docstring fix |
| `tests/test_driver.py` | Modify (Task 1) | Add `--retry-failed` + `--overwrite` mode tests |
| `src/mineru_rocm/cli.py` | Create (Task 2; mirror Hunyuan `cli.py`) | The `mineru-rocm` CLI: 6 subcommands |
| `pyproject.toml` | Modify (Task 2) | Add `[project.scripts] mineru-rocm = "mineru_rocm.cli:main"` |
| `tests/test_cli.py` | Create (Task 2) | TDD for the CLI (CPU-only; monkeypatch heavy callables) |
| `scripts/check_repo.py` | Create (Task 3; adapt Hunyuan `check_repo.py`) | Repo-consistency gate (install smoke + AST scan + SPDX + README refs + lock structure) |
| `.github/workflows/ci.yml` | Modify (Task 3) | Run `check_repo.py` in the smoke job |
| `tests/test_check_repo.py` | Create (Task 3) | TDD for check_repo |

Out of P1d (Follow-on): `doctor` deep env checks beyond the minimal version/import probe; `benchmark`/`report` subcommands (need a filled lock → P3); README↔lock *value* cross-check (P3, when the lock is populated); the existing `scripts/check_deps.py` stays (check_repo SUPERSEDES its no-engine scan conceptually but both can coexist; consolidating is a later cleanup).

---

## Task 1: Driver hardening — `command=` threading + resume-mode tests + docstring fix

**Files:**
- Modify: `src/mineru_rocm/driver.py` (`_orchestrate` + `run`)
- Modify: `tests/test_driver.py` (append 2 mode tests)

**Interfaces:**
- Consumes: nothing new.
- Produces: `_orchestrate(args, *, infer_page, backend, model, cfg, platform="linux-rocm", command=None) -> int` — the new `command=None` kwarg threads through to `runner.write_run_manifest(command=...)` (None ⇒ runner's `safe_argv()` default, i.e. existing behavior unchanged). `run(args)` gains `command=None` too (passes through). Task 2's CLI `predict` handler passes an explicit `command=["mineru-rocm", "predict", ...]`.

- [ ] **Step 1: Write the 2 failing mode tests (append to `tests/test_driver.py`)**

```python
def test_orchestration_retry_failed_mode(tmp_path):
    """--retry-failed: a previously-failed page is re-attempted; conservation holds."""
    gt, img_dir = _make_gt_and_images(tmp_path, ["a", "b"])
    pred = tmp_path / "pred"
    # pre-seed "a" as FAILED the way the runner does (error record, no .md)
    try:
        raise RuntimeError("earlier crash")
    except RuntimeError as e:
        runner.record_error(pred, "a", image_path="a.png", backend="pipeline", endpoint="in-process", exc=e, attempt=1)

    seen = []
    def fake_infer(img, platform, cfg):
        seen.append(Path(img).stem)
        return f"# {Path(img).stem}\n"  # succeeds this time

    code = driver._orchestrate(
        _args(tmp_path, gt, img_dir, retry_failed=True), infer_page=fake_infer,
        backend="pipeline", model="m", cfg={},
    )
    assert code == 0
    assert seen == ["a"]  # only the failed page re-run; "b" (pending) NOT run under retry_failed
    m = json.loads((pred / "run_manifest.json").read_text("utf-8"))
    assert m["run_counts"]["attempted"] == 1 and m["run_counts"]["skipped"] == 1
    assert m["final_state"]["complete"] == 1 and m["final_state"]["failed"] == 0  # a now complete, b still pending
    assert runner.validate_manifest(m) == []


def test_orchestration_overwrite_mode(tmp_path):
    """--overwrite: even a complete page is re-inferred; conservation holds."""
    gt, img_dir = _make_gt_and_images(tmp_path, ["a"])
    pred = tmp_path / "pred"
    runner.commit_success(pred, "a", "# old output\n")  # already complete

    seen = []
    def fake_infer(img, platform, cfg):
        seen.append(Path(img).stem)
        return f"# {Path(img).stem} NEW\n"

    code = driver._orchestrate(
        _args(tmp_path, gt, img_dir, overwrite=True), infer_page=fake_infer,
        backend="pipeline", model="m", cfg={},
    )
    assert code == 0
    assert seen == ["a"] and (pred / "a.md").read_text("utf-8") == "# a NEW\n"  # re-inferred
    m = json.loads((pred / "run_manifest.json").read_text("utf-8"))
    assert m["run_counts"]["attempted"] == 1 and m["run_counts"]["skipped"] == 0
    assert runner.validate_manifest(m) == []
```

- [ ] **Step 2: Run to verify they fail**

Run: `/opt/venv/bin/python -m pytest tests/test_driver.py::test_orchestration_retry_failed_mode tests/test_driver.py::test_orchestration_overwrite_mode -q`
Expected: FAIL — both pass already? If they pass, the modes already work (the runner supports them); if not, the failure is the test asserting the exact skipped/final_state shape. (They should PASS against the current `_orchestrate` since `select_todo` already honors `retry_failed`/`overwrite` — these are COVERAGE tests for modes the final review flagged as untested. If they pass immediately, that's fine — they lock the behavior.)

- [ ] **Step 3: Add `command=` to `_orchestrate` + `run` (additive; fix the docstring)**

In `src/mineru_rocm/driver.py`:
1. Change the `_orchestrate` signature to add `command: list[str] | None = None` as the last keyword-only param.
2. In the `runner.write_run_manifest(...)` call, add `command=command,` (when `command is None`, `write_run_manifest` keeps its `safe_argv()` default — existing behavior unchanged).
3. Change `run(args) -> int` to `run(args, command=None) -> int` and pass `command=command` through to `_orchestrate`.
4. Soften the `_orchestrate` docstring's last line from "Writes ``run_manifest.json`` on every run that starts (not on a conflict abort)." to "Writes ``run_manifest.json`` on every run that completes the loop body (not on a conflict abort or a mid-run crash; resume recovers partial progress)."

- [ ] **Step 4: Run the tests to verify they pass**

Run: `/opt/venv/bin/python -m pytest tests/test_driver.py -q`
Expected: `11 passed` (9 existing + 2 new).

- [ ] **Step 5: Validate**

Run: `/opt/venv/bin/python scripts/check_deps.py` → `P0 pyproject OK`.
Run: `/opt/venv/bin/python -c "import mineru_rocm.driver; import sys; print('engine:', 'omnidocbench_amd' in sys.modules, 'torch:', 'torch' in sys.modules)"` → `engine: False | torch: False`.
Run: `/opt/venv/bin/python -m pytest -q` → 98 passed (96 + 2).

- [ ] **Step 6: Commit**

```bash
git add src/mineru_rocm/driver.py tests/test_driver.py
git commit -m "feat(p1d): driver hardening — command= manifest provenance + retry-failed/overwrite mode tests + docstring fix"
```
(+ `Co-Authored-By: Claude <noreply@anthropic.com>` trailer.)

---

## Task 2: The `mineru-rocm` CLI (`cli.py`) + `[project.scripts]` entry

**Files:**
- Create: `src/mineru_rocm/cli.py` (mirror Hunyuan's `src/hunyuan_ocr/cli.py` structure, adapted)
- Modify: `pyproject.toml` (add `[project.scripts]`)
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `mineru_rocm.driver.run/parse_args`, `mineru_rocm.scoring.score_directory/format_score_table/ScoringError`, `mineru_rocm.validation.validate_predictions`, `mineru_rocm.canary.materialize/CanaryError`, `mineru_rocm.runner.validate_manifest`.
- Produces: `main(argv=None) -> int` (the `mineru-rocm` entry point). Subcommands: `doctor`, `validate`, `predict`, `score`, `canary` (with `materialize`), `manifest` (with `verify`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cli.py` mirroring Hunyuan's `tests/test_cli.py` patterns (CPU-only; `capsys` for output; `monkeypatch`/`tmp_path` for heavy callables). Exact tests:

```python
import json
import sys
import pytest
from mineru_rocm import cli, runner


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as ei:
        cli.main(["--help"])
    assert ei.value.code == 0
    out = capsys.readouterr().out
    for sub in ("predict", "score", "validate", "canary", "manifest", "doctor"):
        assert sub in out


def test_doctor_advisory_exits_zero(capsys):
    # advisory mode (no --strict): never fails on missing optional deps
    assert cli.main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "mineru_rocm" in out  # always reports the package itself


def test_doctor_json_shape(capsys):
    assert cli.main(["doctor", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert isinstance(data, list) and any(d["label"] == "mineru_rocm" for d in data)


def test_manifest_verify_ok(tmp_path, capsys):
    # write a valid manifest via the runner, then verify it
    runner.write_run_manifest(
        tmp_path, backend="pipeline", model="m",
        run_counts={"attempted": 1, "succeeded": 1, "failed": 0, "skipped": 0},
        final_state={"expected": 1, "complete": 1, "failed": 0, "pending": 0},
        command=["mineru-rocm", "predict"],
    )
    assert cli.main(["manifest", "verify", "--pred-dir", str(tmp_path)]) == 0
    assert "[OK]" in capsys.readouterr().out


def test_manifest_verify_invalid_reports_violations(tmp_path, capsys):
    (tmp_path / "run_manifest.json").write_text(json.dumps({"schema_version": 99}), encoding="utf-8")
    assert cli.main(["manifest", "verify", "--pred-dir", str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "schema_version" in err  # friendly violation report, no traceback


def test_score_invalid_preddir_is_friendly(tmp_path, capsys):
    # a pred-dir with no .md files fails pre-score VALIDATION (the CPU-only path before the
    # scorer subprocess) → ScoringError → friendly message, exit 1, no traceback.
    gt = tmp_path / "gt.json"
    gt.write_text(json.dumps([{"page_info": {"image_path": "a.png"}}]), encoding="utf-8")
    rc = cli.main(["score", "--gt-json", str(gt), "--pred-dir", str(tmp_path / "nope")])
    assert rc == 1
    assert "Traceback" not in capsys.readouterr().err


def test_canary_materialize_missing_inputs_friendly(tmp_path, capsys):
    rc = cli.main(["canary", "materialize", "--full-gt", str(tmp_path / "nope.json"),
                   "--manifest", str(tmp_path / "nope-m.json"), "--out", str(tmp_path / "o.json")])
    assert rc == 1
    assert "Traceback" not in capsys.readouterr().err


def test_validate_clean(tmp_path, capsys):
    gt = tmp_path / "gt.json"
    gt.write_text(json.dumps([{"page_info": {"image_path": "a.png"}}]), encoding="utf-8")
    pred = tmp_path / "pred"; pred.mkdir()
    (pred / "a.md").write_text("# a")
    assert cli.main(["validate", "--gt-json", str(gt), "--pred-dir", str(pred)]) == 0
    assert "[OK]" in capsys.readouterr().out


def test_predict_reaches_driver_arg_check():
    # no extra args → driver.parse_args raises SystemExit (missing required --gt-json/--pred-dir) before any GPU work
    with pytest.raises(SystemExit):
        cli.main(["predict", "--backend", "pipeline"])
```

- [ ] **Step 2: Run to verify they fail**

Run: `/opt/venv/bin/python -m pytest tests/test_cli.py -q`
Expected: FAIL — `ImportError: cannot import name 'cli' from 'mineru_rocm'`.

- [ ] **Step 3: Implement `src/mineru_rocm/cli.py` (mirror Hunyuan's `cli.py`, adapted)**

Read Hunyuan's `/workspace/HunyuanOCR-ROCm/src/hunyuan_ocr/cli.py` and mirror its structure: `build_parser()` (argparse with subparsers) → `main(argv=None) -> int` (parse + dispatch dict + `int(rc or 0)`) → per-subcommand `_doctor/_validate/_predict/_score/_canary_materialize/_manifest_verify` handlers. Adapt to MinerU:

- **Subcommands + flags** (use argparse subparsers; mirror Hunyuan's lines 421-479):
  - `doctor`: `--strict` (store_true), `--json` (store_true).
  - `validate`: `--gt-json` (required), `--pred-dir` (required), `--lenient` (store_true).
  - `manifest` → subsubparser `verify`: `--pred-dir` (required).
  - `canary` → subsubparser `materialize`: `--full-gt` (required), `--manifest` (required), `--out` (required).
  - `predict`: `--backend` (required, choices `pipeline`/`vlm-vllm`), `extra` (`nargs=argparse.REMAINDER` — forwarded to `driver.parse_args`).
  - `score`: `--pred-dir` (required), `--gt-json` (required), `--label` (default "score"), `--omnidocbench-repo` (default None), `--venv-python` (default None), `--skip-validation` (store_true).
- **Handlers** (each returns `int`; friendly errors to stderr, no tracebacks — wrap library exceptions):
  - `_doctor(args)`: build a list of `(label, status, detail)` checks: `mineru_rocm` importable + `__version__` (required); optional deps `mineru`/`torch`/`vllm`/`yaml` via `importlib.import_module` in `try/except` (advisory: status `ok`/`miss`); `config.BACKEND` (info). Print a table (`[OK]`/`[MISS]`/`[INFO]`) by default, or JSON list with `--json`. `--strict` ⇒ exit 1 only if `mineru_rocm` itself is missing (the one required check); otherwise exit 0 (advisory).
  - `_validate(args)`: `from mineru_rocm import validation`; `r = validation.validate_predictions(args.gt_json, args.pred_dir, strict=not args.lenient)`; print `[OK] valid=N/N` or the error list; return 0 if `r.ok` else 1.
  - `_manifest_verify(args)`: load `pred_dir/run_manifest.json`; `errs = runner.validate_manifest(m)`; if `errs == []` print `[OK]` return 0; else print violations to stderr return 1. Hard error (exit 2) if the file is missing/unreadable (friendly message).
  - `_canary_materialize(args)`: `from mineru_rocm import canary`; `sha = canary.materialize(args.full_gt, args.manifest, args.out)`; print `[OK] <out> sha256=<sha>` return 0. Catch `canary.CanaryError` → friendly stderr, return 1.
  - `_predict(args)`: build `drv_argv = ["--backend", args.backend, *args.extra]`; `dargs = driver.parse_args(drv_argv)`; `return driver.run(dargs, command=["mineru-rocm", "predict", args.backend])` (the `command=` threads the user-facing CLI into the manifest — Task 1's hardening). (`driver.run` lazy-imports the backend, so no GPU at import.)
  - `_score(args)`: `from mineru_rocm import scoring`; `try: result = scoring.score_directory(gt_json=args.gt_json, pred_dir=args.pred_dir, omnidocbench_repo=args.omnidocbench_repo, venv_python=args.venv_python, skip_validation=args.skip_validation); print(scoring.format_score_table(args.label, result["metrics"])); return 0` `except scoring.ScoringError as e: print(e, file=sys.stderr); return 1`.
- **Top-level imports**: stdlib only (`argparse`/`importlib`/`json`/`sys`/`pathlib`). Import `mineru_rocm.runner` at top (stdlib-only module). Import `driver`/`scoring`/`validation`/`canary` LAZILY inside their handlers (keeps `import mineru_rocm.cli` GPU-free + fast). Keep an SPDX header.

- [ ] **Step 4: Add the `[project.scripts]` entry to `pyproject.toml`**

Add (after the `[project.urls]` block, before `[tool.setuptools.packages.find]`):

```toml
[project.scripts]
mineru-rocm = "mineru_rocm.cli:main"
```

- [ ] **Step 5: Run the tests + reinstall + CLI smoke**

Run: `/opt/venv/bin/pip install -e . -q` (picks up the new entry point + module).
Run: `/opt/venv/bin/python -m pytest tests/test_cli.py -q` → `9 passed`.
Run: `/opt/venv/bin/python -m mineru_rocm.cli --help` → usage listing the 6 subcommands (exits 0).
Run: `mineru-rocm --help` (the installed console script) → same usage (exits 0). If the console script isn't on PATH, use `/opt/venv/bin/mineru-rocm --help`.

- [ ] **Step 6: Validate**

Run: `/opt/venv/bin/python scripts/check_deps.py` → `P0 pyproject OK`.
Run: `/opt/venv/bin/python -c "import mineru_rocm.cli; import sys; print('engine:', 'omnidocbench_amd' in sys.modules, 'torch:', 'torch' in sys.modules)"` → `engine: False | torch: False` (lazy handler imports).
Run: `/opt/venv/bin/python -m pytest -q` → 107 passed (98 + 9).

- [ ] **Step 7: Commit**

```bash
git add src/mineru_rocm/cli.py pyproject.toml tests/test_cli.py
git commit -m "feat(p1d): mineru-rocm CLI (doctor|validate|predict|score|canary|manifest verify) + [project.scripts] entry; 9 tests"
```
(+ `Co-Authored-By` trailer.)

---

## Task 3: `scripts/check_repo.py` (repo-consistency gate) + CI wiring

**Files:**
- Create: `scripts/check_repo.py` (adapt Hunyuan's `scripts/check_repo.py`)
- Modify: `.github/workflows/ci.yml` (run check_repo in the smoke job)
- Test: `tests/test_check_repo.py`

**Interfaces:**
- Consumes: the repo tree (`pyproject.toml`, `src/mineru_rocm/`, `README.md`, `reproducibility.lock.yaml`).
- Produces: `scripts/check_repo.py` — a standalone script (exit 0 clean / 1 on any finding) + pytest tests.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_check_repo.py`:

```python
import ast
import subprocess
import sys
from pathlib import Path
import pytest

REPO = Path(__file__).resolve().parents[1]


def test_no_engine_imports_in_core_uses_ast():
    """The no-engine scan is AST-based (catches 'import omnidocbench_amd' anywhere in src/mineru_rocm)."""
    import scripts.check_repo as cr  # noqa
    leaks = cr.find_engine_imports(REPO / "src" / "mineru_rocm")
    assert leaks == [], f"engine imports leaked into core: {leaks}"


def test_required_lock_sections_present():
    import scripts.check_repo as cr
    lock = cr._load_lock()
    missing = cr.check_lock_sections(lock)
    assert missing == [], missing


def test_spdx_headers_on_src_and_scripts():
    import scripts.check_repo as cr
    assert cr.check_spdx(REPO) == []


def test_readme_script_references_exist():
    import scripts.check_repo as cr
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    assert cr.check_readme_scripts_exist(readme) == []


def test_check_repo_clean_on_repo(capsys):
    """Integration gate: the FAST checks (no install smoke) all pass on the real repo.

    `check_install_smoke` runs `pip install -e .` (slow + mutates env) so it runs
    in CI via `main()`, not here. This test covers the engine/lock/SPDX/README checks."""
    import scripts.check_repo as cr
    findings = []
    findings += cr.find_engine_imports(REPO / "src" / "mineru_rocm")
    findings += cr.check_lock_sections(cr._load_lock())
    findings += cr.check_spdx()
    readme = (REPO / "README.md").read_text(encoding="utf-8") if (REPO / "README.md").is_file() else ""
    findings += cr.check_readme_scripts_exist(readme)
    assert findings == [], findings
```

- [ ] **Step 2: Run to verify they fail**

Run: `/opt/venv/bin/python -m pytest tests/test_check_repo.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.check_repo'`.

- [ ] **Step 3: Implement `scripts/check_repo.py` (adapt Hunyuan's; add the 2 MinerU-specific checks)**

Read Hunyuan's `/workspace/HunyuanOCR-ROCm/scripts/check_repo.py` for the structure (a `main(argv)` that runs each `check_*` function, collects findings, prints them, exits 1 on any). Adapt to MinerU with these checks (each returns `list[str]` of findings, empty = clean):

```python
#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
"""Repo-consistency gate. Exits 0 clean, 1 on any finding. Run in CI + locally."""
from __future__ import annotations
import ast, json, re, subprocess, sys, tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ENGINE_MODULES = ("omnidocbench_amd", "torch", "mineru", "mineru_vl_utils", "openai", "vllm")
REQUIRED_LOCK_SECTIONS = ("mineru_rocm", "mineru", "model", "omnidocbench", "environment", "benchmark")


def find_engine_imports(pkg_dir: Path) -> list[str]:
    """AST scan: no ENGINE_MODULES imported at module top-level anywhere under pkg_dir.

    (Catches `import omnidocbench_amd`, `from omnidocbench_amd import x`, `import torch as t` —
    more robust than check_deps.py's substring scan, which a comment could trip.)"""
    errs = []
    for py in sorted(pkg_dir.rglob("*.py")):
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except SyntaxError as e:
            errs.append(f"{py}: unparseable: {e}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in ENGINE_MODULES:
                        errs.append(f"{py}:{node.lineno}: top-level `import {alias.name}` (engine/heavy dep must be lazy)")
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module.split(".")[0] in ENGINE_MODULES and node.level == 0:
                    errs.append(f"{py}:{node.lineno}: top-level `from {node.module} import ...` (engine/heavy dep must be lazy)")
    return errs


def _load_lock():
    p = REPO / "reproducibility.lock.yaml"
    if not p.is_file():
        return None
    import yaml  # PyYAML is a core dep
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def check_lock_sections(lock) -> list[str]:
    if lock is None:
        return []
    return [f"reproducibility.lock.yaml missing section: {k}" for k in REQUIRED_LOCK_SECTIONS if k not in lock]


def check_spdx(repo=REPO) -> list[str]:
    errs = []
    for sub in ("src", "scripts"):
        for py in (repo / sub).rglob("*.py"):
            head = "\n".join(py.read_text(encoding="utf-8").splitlines()[:3])
            if "SPDX-License-Identifier" not in head:
                errs.append(f"{py.relative_to(repo)}: missing SPDX-License-Identifier header")
    return errs


_SCRIPT_REF_RE = re.compile(r"scripts/([A-Za-z0-9_]+\.(?:py|sh))")
def check_readme_scripts_exist(readme: str) -> list[str]:
    errs = []
    for name in _SCRIPT_REF_RE.findall(readme):
        if not (REPO / "scripts" / name).exists():
            errs.append(f"README.md references scripts/{name}, which does not exist")
    return errs


def check_install_smoke() -> list[str]:
    """`pip install -e .` succeeds (the PEP 639 / build regression guard)."""
    cp = subprocess.run([sys.executable, "-m", "pip", "install", "-e", ".", "--quiet"],
                        cwd=str(REPO), capture_output=True, text=True)
    if cp.returncode != 0:
        return [f"`pip install -e .` failed (rc={cp.returncode}):\n{(cp.stderr or '')[-800:]}"]
    return []


def main(argv=None) -> int:
    findings = []
    findings += find_engine_imports(REPO / "src" / "mineru_rocm")
    findings += check_lock_sections(_load_lock())
    findings += check_spdx()
    readme = (REPO / "README.md").read_text(encoding="utf-8") if (REPO / "README.md").is_file() else ""
    findings += check_readme_scripts_exist(readme)
    findings += check_install_smoke()
    if findings:
        print("check_repo: " + str(len(findings)) + " finding(s):", file=sys.stderr)
        for f in findings:
            print("  - " + f, file=sys.stderr)
        return 1
    print("check_repo: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Create an empty `scripts/__init__.py` too (so `import scripts.check_repo` resolves in pytest). It is NOT shipped in the wheel — `[tool.setuptools.packages.find] where = ["src"]` keeps `scripts/` out of the installed package; it only makes `scripts/` importable from the repo root during tests/CI.

- [ ] **Step 4: Wire into CI**

In `.github/workflows/ci.yml`, add a step after `pip install -e ".[dev]"` and before `pytest -q`:

```yaml
      - run: python scripts/check_repo.py
```

- [ ] **Step 5: Run the tests + the script + full suite**

Run: `/opt/venv/bin/python -m pytest tests/test_check_repo.py -q` → `5 passed`.
Run: `/opt/venv/bin/python scripts/check_repo.py` → `check_repo: clean` (exit 0).
Run: `/opt/venv/bin/python -m pytest -q` → 112 passed (107 + 5).

- [ ] **Step 6: Commit**

```bash
git add scripts/check_repo.py scripts/__init__.py .github/workflows/ci.yml tests/test_check_repo.py
git commit -m "feat(p1d): scripts/check_repo.py repo-consistency gate (AST no-engine scan + pip-install smoke + SPDX + README refs + lock sections); wired into CI; 5 tests"
```
(+ `Co-Authored-By` trailer.)

---

## Definition of Done (P1d)

- [ ] `mineru-rocm --help` lists the 6 subcommands; each subcommand's `--help` works; `import mineru_rocm.cli` is GPU-free + engine-free.
- [ ] CLI subcommands wrap the library correctly: `predict`→`driver.run` (with `command=` manifest provenance), `score`→`scoring.score_directory`, `validate`→`validation.validate_predictions`, `manifest verify`→`runner.validate_manifest`, `canary materialize`→`canary.materialize`; `doctor` reports the env (advisory). Friendly errors (no tracebacks); exit codes 0/1/2.
- [ ] `[project.scripts] mineru-rocm = "mineru_rocm.cli:main"` in pyproject; the console script is installed.
- [ ] `scripts/check_repo.py` runs clean on the repo; it includes the **`pip install -e .` smoke** (PEP 639 guard) + **AST-based no-engine-import scan** + SPDX + README script-refs + lock-section checks.
- [ ] CI (`ci.yml` smoke job) runs `check_repo.py` between install + pytest.
- [ ] `python -m pytest -q` green: 112 total (96 + 2 driver-mode + 9 cli + 5 check_repo).
- [ ] Engine subprocess contract untouched (`adapter/run_adapter.py --help` = 7 flags); `dispatcher`/`backends`/`runner`/`types` byte-identical to main.

## Follow-on (deferred)

- **README↔lock value cross-check** (spec §3.2 "CI fails on drift"): deferred to **P3** — `reproducibility.lock.yaml` is a `not_recorded` skeleton today; the cross-check becomes meaningful + enforceable once P3 fills the lock with the real re-run SHAs/scores. P1d's `check_repo` ships the *structure* check (required lock sections present); P3 adds the *value* cross-check (README tables match lock values).
- **`doctor` deep checks** (ROCm/HIP version, GPU presence, weights present): the P1d `doctor` is a minimal import/version probe; deeper GPU/weights checks land when the reproduce flow (P3) needs them.
- **`benchmark`/`report` subcommands**: deferred (need a filled lock + release-artifact assembly).
- **Consolidate `check_deps.py` into `check_repo.py`**: check_repo's AST scan supersedes check_deps's substring scan conceptually; both can coexist for now (check_deps is fast + the validator asserts pyproject shape). Full consolidation is a later cleanup.

---

**P1 completion note:** P1d is the last P1 sub-phase. After it lands, `src/mineru_rocm/` is a complete, CLI-usable, GPU-free-core package (12 modules: types/dispatcher/config/backends×2/omnidbench/preflight/validation/canary/scoring/runner/driver/cli) + a repo-consistency gate, with the heavy inference/scoring reachable via `mineru-rocm predict|score` (real backends exercised in P3). The next major phase is **P2/P3** (results re-run + reproducibility lock fill), which will finally prove the real end-to-end `mineru-rocm predict | score` path against actual MinerU weights + OmniDocBench.
