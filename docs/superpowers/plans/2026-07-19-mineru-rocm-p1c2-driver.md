# MinerU-ROCm P1c.2 — Runner-Driven Inference Driver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the P1c runner to the inference backends via a single backend-parameterized driver (`mineru_rocm/driver.py`) that runs the MinerU pipeline **or** VLM over an OmniDocBench page set with atomic writes, resumability, and a conservation-checked `run_manifest.json` — and prove the full resume lifecycle end-to-end with a fake infer (the P1c carry-forward integration smoke).

**Architecture:** NEW MinerU code (not a port). A single driver `_orchestrate(args, *, infer_page, backend, model, cfg, platform)` runs the loop `preflight.pages_with_images → runner.detect_stem_conflicts → runner.acquire_run_lock → runner.select_todo → per-page {infer_page → runner.commit_success | runner.record_error} → runner.aggregate_errors → runner.write_run_manifest`. `run(args)` selects the backend (`pipeline`→`backends.pipeline.infer_page`, `vlm-vllm`→`backends.vlm.infer_page`) and builds `cfg`. Both backends are single-endpoint + in-process (MinerU's reality — Plan 1/2 ran one server on GPU 0), so Hunyuan's `endpoint_pool`/OpenAI-client machinery is intentionally NOT ported (YAGNI per the owner's decision). The driver is CPU-testable: tests inject a fake `infer_page`, exercising the runner end-to-end without GPU.

**Tech Stack:** Python 3.11+, stdlib (`argparse`/`json`/`sys`/`time`/`pathlib`), pytest. Heavy deps (`mineru`/`mineru_vl_utils`/`torch`) stay inside the backends (lazy), so the driver module imports GPU-free.

## Global Constraints

(From the approved spec `docs/superpowers/specs/2026-07-19-mineru-rocm-standalone-port-design.md` §3.2; every task implicitly includes these.)

- **No behavior / no score change.** The driver is a NEW robust path. `dispatcher.run_adapter` (engine subprocess, 8-key `_run_stats.json`) + `adapter/run_adapter.py` CLI MUST stay byte-identical — P1c.2 does NOT touch `dispatcher.py`/`adapter/`/`backends/`/`types.py`. The driver writes `run_manifest.json` (via the P1c runner), NOT `_run_stats.json`.
- **Atomic + resumable** (spec §3.2): every page goes through `runner.commit_success` (atomic `.partial`→`os.replace`); resume skips only genuinely-complete pages (`runner.select_todo`/`is_complete`); `.run.lock` prevents two writers; the manifest is authoritative.
- **Core stays GPU-free / platform-free.** `driver.py` imports with **no** `torch`/`mineru`/`mineru_vl_utils`/`omnidocbench_amd`/`openai`/`requests` at module top level. Backend modules are imported lazily inside `run()` (and the backends themselves lazy-import their heavy deps).
- **Conservation laws hold.** The manifest the driver writes MUST pass `runner.validate_manifest` (empty error list): `attempted == succeeded + failed + interrupted`; `expected == attempted + skipped`; `expected == complete + failed + pending`; `status == "ok"` ⇒ `failed == 0` and `pending == 0`.
- **One concern per commit; commit after every task's validation passes.** Branch: `feat/p1c2-driver` (off `main` @ `78b8851`, which has P1a+P1b+P1c).
- **Validation environment:** `/opt/venv/bin/python` and `/opt/venv/bin/pip` ONLY (py3.12; `mineru-rocm` editable, `pytest`, `PyYAML`).

---

## File Structure (P1c.2 scope)

| File | Action | Responsibility |
|---|---|---|
| `src/mineru_rocm/driver.py` | Create (NEW; grown across Tasks 1-2) | Backend-parameterized inference driver: `_orchestrate` (Task 1) + `parse_args`/`run`/`main` (Task 2) |
| `tests/test_driver.py` | Create (NEW) | TDD for the driver (integration smoke + resume + failure + entry/CLI) |

Out of P1c.2 (deliberate, per owner decision): `endpoint_pool.py` and a multi-endpoint VLM driver — MinerU is single-server today; defer until multi-replica VLM is actually needed. `cli.py` / `check_repo.py` (P1d) — the `mineru-rocm predict` CLI will call `driver.main`.

---

## Task 1: Orchestration core `_orchestrate` + integration smoke

**Files:**
- Create: `src/mineru_rocm/driver.py` (module header + top imports + `_orchestrate`)
- Test: `tests/test_driver.py`

**Interfaces:**
- Consumes: `mineru_rocm.runner` (P1c: `acquire_run_lock`, `select_todo`, `commit_success`, `record_error`, `aggregate_errors`, `page_status`, `decide_run_status`, `write_run_manifest`, `detect_stem_conflicts`) and `mineru_rocm.preflight` (P1b: `pages_with_images`).
- Produces: `_orchestrate(args, *, infer_page, backend, model, cfg, platform="linux-rocm") -> int` — the backend-agnostic run loop (Task 2's `run` calls it; tests call it with a fake `infer_page`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_driver.py`:

```python
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from mineru_rocm import driver, runner


def _make_gt_and_images(tmp_path, stems):
    """Write a minimal OmniDocBench GT json + an images dir with one PNG per stem."""
    gt = tmp_path / "gt.json"
    gt.write_text(
        json.dumps([{"page_info": {"image_path": f"{s}.png"}} for s in stems]),
        encoding="utf-8",
    )
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for s in stems:
        (img_dir / f"{s}.png").write_bytes(b"\x89PNG fake")  # contents irrelevant (fake infer)
    return gt, img_dir


def _args(tmp_path, gt, img_dir, **over):
    base = dict(
        gt_json=str(gt), images_dir=str(img_dir), pred_dir=str(tmp_path / "pred"),
        backend="pipeline", model="m", platform="linux-rocm",
        max_retries=2, retry_backoff=0.0, overwrite=False, retry_failed=False,
    )
    base.update(over)
    return SimpleNamespace(**base)


def test_orchestration_smoke_full_round_trip(tmp_path):
    """Integration smoke (P1c carry-forward): preflight→select_todo→infer→commit_success→write_manifest→validate_manifest."""
    gt, img_dir = _make_gt_and_images(tmp_path, ["a", "b", "c"])

    def fake_infer(img, platform, cfg):
        return f"# {Path(img).stem}\n\n(fake)\n"

    code = driver._orchestrate(
        _args(tmp_path, gt, img_dir), infer_page=fake_infer,
        backend="pipeline", model="m", cfg={},
    )
    assert code == 0
    pred = tmp_path / "pred"
    # every page written atomically
    assert {p.stem for p in pred.glob("*.md")} == {"a", "b", "c"}
    # manifest present + conservation laws hold
    m = json.loads((pred / "run_manifest.json").read_text("utf-8"))
    assert m["status"] == "ok"
    assert m["run_counts"] == {"attempted": 3, "succeeded": 3, "failed": 0, "skipped": 0, "interrupted": 0}
    assert m["final_state"] == {"expected": 3, "complete": 3, "failed": 0, "pending": 0}
    assert m["backend"] == "pipeline" and m["model"] == "m"
    assert runner.validate_manifest(m) == []  # the load-bearing conservation check


def test_orchestration_resume_skips_complete(tmp_path):
    """A genuinely-complete page is skipped on re-run (select_todo), counted as skipped."""
    gt, img_dir = _make_gt_and_images(tmp_path, ["a", "b"])
    pred = tmp_path / "pred"
    # pre-complete page "a" the way the runner does (atomic + valid content)
    runner.commit_success(pred, "a", "# a already done\n")

    seen = []
    def fake_infer(img, platform, cfg):
        seen.append(Path(img).stem)
        return f"# {Path(img).stem}\n"

    code = driver._orchestrate(
        _args(tmp_path, gt, img_dir), infer_page=fake_infer,
        backend="pipeline", model="m", cfg={},
    )
    assert code == 0
    assert seen == ["b"]  # "a" was skipped, never inferred
    m = json.loads((pred / "run_manifest.json").read_text("utf-8"))
    assert m["run_counts"]["attempted"] == 1 and m["run_counts"]["skipped"] == 1
    assert m["final_state"]["complete"] == 2  # a (pre-done) + b (this run)
    assert runner.validate_manifest(m) == []


def test_orchestration_failure_recorded_manifest_failed(tmp_path):
    """A page whose infer always raises is recorded + makes the run status 'failed' (conservation still holds)."""
    gt, img_dir = _make_gt_and_images(tmp_path, ["ok1", "bad"])

    def fake_infer(img, platform, cfg):
        if Path(img).stem == "bad":
            raise RuntimeError("boom")
        return f"# {Path(img).stem}\n"

    code = driver._orchestrate(
        _args(tmp_path, gt, img_dir, max_retries=2, retry_backoff=0.0),
        infer_page=fake_infer, backend="pipeline", model="m", cfg={},
    )
    assert code == 1  # non-ok status
    pred = tmp_path / "pred"
    assert (pred / "ok1.md").exists()
    assert not (pred / "bad.md").exists()  # never committed
    assert (pred / "_errors" / "bad.json").exists()  # error record written
    m = json.loads((pred / "run_manifest.json").read_text("utf-8"))
    assert m["status"] == "failed"
    assert m["run_counts"]["failed"] == 1 and m["run_counts"]["succeeded"] == 1
    assert m["final_state"]["failed"] == 1 and m["final_state"]["complete"] == 1
    assert runner.validate_manifest(m) == []  # conservation holds even on failure


def test_orchestration_conflict_aborts(tmp_path):
    """Two images mapping to the same stem abort before any write (returns 1, no manifest)."""
    gt = tmp_path / "gt.json"
    gt.write_text(
        json.dumps([
            {"page_info": {"image_path": "dir/page.png"}},
            {"page_info": {"image_path": "other/page.png"}},  # same stem "page"
        ]),
        encoding="utf-8",
    )
    img_dir = tmp_path / "images"
    (img_dir / "dir").mkdir(parents=True)
    (img_dir / "other").mkdir(parents=True)
    (img_dir / "dir" / "page.png").write_bytes(b"x")
    (img_dir / "other" / "page.png").write_bytes(b"x")

    code = driver._orchestrate(
        _args(tmp_path, gt, img_dir), infer_page=lambda *a, **k: "# x\n",
        backend="pipeline", model="m", cfg={},
    )
    assert code == 1
    assert not (tmp_path / "pred" / "run_manifest.json").exists()  # nothing written
```

- [ ] **Step 2: Run it to verify it fails**

Run: `/opt/venv/bin/python -m pytest tests/test_driver.py -q`
Expected: FAIL — `ImportError: cannot import name 'driver' from 'mineru_rocm'` (module doesn't exist yet).

- [ ] **Step 3: Implement `_orchestrate` in `src/mineru_rocm/driver.py`**

Create `src/mineru_rocm/driver.py`:

```python
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
"""Backend-parameterized inference driver — the robust run path.

Orchestrates one MinerU backend (pipeline | vlm-vllm, both single-endpoint /
in-process) over an OmniDocBench page set using the ``mineru_rocm.runner``
primitives: atomic per-page writes, structured error records, resumability that
skips only genuinely-complete pages, an exclusive writer lock, and a
conservation-checked ``run_manifest.json``. This is a NEW path parallel to the
omnidocbench-amd engine subprocess (``dispatcher.run_adapter`` writes
``_run_stats.json``); it does not touch that contract.

Heavy backend deps (mineru / mineru_vl_utils / torch) are imported lazily inside
``run()``, so this module imports with no GPU deps installed.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from mineru_rocm import preflight, runner


def _orchestrate(args, *, infer_page, backend: str, model: str, cfg: dict, platform: str = "linux-rocm") -> int:
    """Run ``infer_page`` over the OmniDocBench page set with full runner integrity.

    ``infer_page(img, platform, cfg) -> str`` is injected so the orchestration is
    CPU-testable without a GPU. Returns 0 on a fully-ok run, 1 otherwise (failed
    pages, pending pages, or a pre-run abort). Writes ``run_manifest.json`` on
    every run that starts (not on a conflict abort).
    """
    # --- preflight: GT + images exist (raises PreflightError on bad input) ---
    pages = preflight.pages_with_images(args.gt_json, args.images_dir)  # [(stem, abs_img), ...]

    # --- output-name conflicts: abort before any write ---
    conflicts = runner.detect_stem_conflicts([img for _, img in pages])
    if conflicts:
        sample = ", ".join(stem for stem, _ in conflicts[:3])
        print(f"[fatal] {len(conflicts)} output-name conflict(s); first: {sample}", file=sys.stderr)
        return 1

    pred_dir = Path(args.pred_dir)
    with runner.acquire_run_lock(pred_dir, command=["mineru_rocm.driver", backend, str(pred_dir)]):
        todo, skipped = runner.select_todo(
            pages, pred_dir, overwrite=args.overwrite, retry_failed=args.retry_failed,
        )
        succeeded = 0
        failed = 0
        for stem, img in todo:
            for attempt in range(1, args.max_retries + 1):
                try:
                    md = infer_page(Path(img), platform, cfg)
                    runner.commit_success(pred_dir, stem, md)
                    succeeded += 1
                    break
                except Exception as exc:  # per-page failure → record + continue (R2 contract)
                    if attempt == args.max_retries:
                        runner.record_error(
                            pred_dir, stem, image_path=str(img), backend=backend,
                            endpoint="in-process", exc=exc, attempt=attempt,
                        )
                        failed += 1
                    else:
                        time.sleep(args.retry_backoff * (2 ** (attempt - 1)))

        runner.aggregate_errors(pred_dir)
        final_complete = sum(1 for s, _ in pages if runner.page_status(pred_dir, s) == "complete")
        final_failed = sum(1 for s, _ in pages if runner.page_status(pred_dir, s) == "failed")
        final_pending = len(pages) - final_complete - final_failed
        status = runner.decide_run_status(final_failed, final_pending)

        runner.write_run_manifest(
            pred_dir,
            backend=backend,
            model=model,
            run_counts={
                "attempted": len(todo), "succeeded": succeeded, "failed": failed,
                "skipped": skipped, "interrupted": 0,
            },
            final_state={
                "expected": len(pages), "complete": final_complete,
                "failed": final_failed, "pending": final_pending,
            },
            status=status,
        )
        return 0 if status == "ok" else 1
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `/opt/venv/bin/python -m pytest tests/test_driver.py -q`
Expected: `4 passed`.

- [ ] **Step 5: Validate (no engine leak + engine contract + full suite)**

Run: `/opt/venv/bin/python scripts/check_deps.py` → `P0 pyproject OK`.
Run: `/opt/venv/bin/python -c "import mineru_rocm.driver; import sys; print('engine:', 'omnidocbench_amd' in sys.modules, '| torch:', 'torch' in sys.modules)"` → `engine: False | torch: False` (driver is GPU-free + engine-free).
Run: `/opt/venv/bin/python adapter/run_adapter.py --help | grep -E '^\s+--' | wc -l` → `7` (engine CLI byte-identical; P1c.2 didn't touch it).
Run: `/opt/venv/bin/python -m pytest -q` → 91 passed (87 + 4).

- [ ] **Step 6: Commit**

```bash
git add src/mineru_rocm/driver.py tests/test_driver.py
git commit -m "feat(p1c2): runner-driven inference driver _orchestrate (atomic+resume+manifest); integration smoke + 4 tests"
```
(End the message body with a blank line + `Co-Authored-By: Claude <noreply@anthropic.com>`.)

---

## Task 2: Entry layer — `parse_args` + `run` (backend-switch) + `main`

**Files:**
- Modify: `src/mineru_rocm/driver.py` (append the entry layer)
- Modify: `tests/test_driver.py` (append the entry tests)

**Interfaces:**
- Consumes: `_orchestrate` (Task 1); `mineru_rocm.backends.pipeline.infer_page` / `mineru_rocm.backends.vlm.infer_page`; `mineru_rocm.config` (server_url / api_model_name defaults).
- Produces:
  - `parse_args(argv=None) -> argparse.Namespace`
  - `run(args) -> int` (selects the backend's `infer_page` + builds `cfg`, calls `_orchestrate`)
  - `main(argv=None) -> int` (so `python -m mineru_rocm.driver …` works)

- [ ] **Step 1: Write the failing tests (append to `tests/test_driver.py`)**

```python
def test_parse_args_required_and_defaults():
    a = driver.parse_args(["--gt-json", "g.json", "--images-dir", "i", "--pred-dir", "p", "--backend", "pipeline"])
    assert a.gt_json == "g.json" and a.backend == "pipeline" and a.platform == "linux-rocm"
    assert a.max_retries == 2 and a.retry_backoff == 2.0 and a.lang == "ch"
    assert a.overwrite is False and a.retry_failed is False


def test_parse_args_rejects_unknown_backend():
    import pytest as _pytest
    with _pytest.raises(SystemExit):
        driver.parse_args(["--gt-json", "g", "--images-dir", "i", "--pred-dir", "p", "--backend", "bogus"])


def test_run_routes_to_pipeline_backend(tmp_path, monkeypatch):
    """run(backend=pipeline) calls backends.pipeline.infer_page via _orchestrate (no GPU)."""
    gt, img_dir = _make_gt_and_images(tmp_path, ["a"])
    called = {}
    from mineru_rocm.backends import pipeline as be
    monkeypatch.setattr(be, "infer_page", lambda img, platform, cfg: called.setdefault("hit", str(img)) or f"# {Path(img).stem}\n")
    a = _args(tmp_path, gt, img_dir, backend="pipeline")
    assert driver.run(a) == 0
    assert "hit" in called  # the real backend selector was invoked


def test_run_routes_to_vlm_backend(tmp_path, monkeypatch):
    """run(backend=vlm-vllm) calls backends.vlm.infer_page via _orchestrate (no GPU)."""
    gt, img_dir = _make_gt_and_images(tmp_path, ["a"])
    from mineru_rocm.backends import vlm as be
    monkeypatch.setattr(be, "infer_page", lambda img, platform, cfg: f"# {Path(img).stem}\n")
    a = _args(tmp_path, gt, img_dir, backend="vlm-vllm")
    assert driver.run(a) == 0


def test_module_is_runnable_help():
    """`python -m mineru_rocm.driver --help` exits 0 and shows the flags."""
    import subprocess
    res = subprocess.run(
        ["/opt/venv/bin/python", "-m", "mineru_rocm.driver", "--help"],
        capture_output=True, text=True,
    )
    assert res.returncode == 0
    assert "--backend" in res.stdout and "--pred-dir" in res.stdout
```

- [ ] **Step 2: Run it to verify it fails**

Run: `/opt/venv/bin/python -m pytest tests/test_driver.py -q`
Expected: FAIL — `AttributeError: module 'mineru_rocm.driver' has no attribute 'parse_args'`.

- [ ] **Step 3: Implement the entry layer (append to `src/mineru_rocm/driver.py`)**

```python
def parse_args(argv=None):
    """CLI for `python -m mineru_rocm.driver`. The P1d `mineru-rocm predict` CLI wraps this."""
    import argparse

    p = argparse.ArgumentParser(
        prog="mineru_rocm.driver",
        description="Run a MinerU backend over an OmniDocBench page set (robust: atomic writes + run_manifest + resume).",
    )
    p.add_argument("--gt-json", required=True)
    p.add_argument("--images-dir", required=True)
    p.add_argument("--pred-dir", required=True)
    p.add_argument("--backend", required=True, choices=["pipeline", "vlm-vllm"])
    p.add_argument("--model", default=None, help="advisory model name for the manifest (default per backend)")
    p.add_argument("--platform", default="linux-rocm")
    p.add_argument("--lang", default="ch")
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--retry-backoff", type=float, default=2.0)
    p.add_argument("--overwrite", action="store_true", help="re-run every page (ignore existing complete pages)")
    p.add_argument("--retry-failed", action="store_true", help="re-run only previously-failed pages")
    return p.parse_args(argv)


def run(args) -> int:
    """Select the backend's infer_page + cfg, then orchestrate. Imports backends lazily (GPU deps stay out of module top-level)."""
    from mineru_rocm import config

    if args.backend == "pipeline":
        from mineru_rocm.backends import pipeline as backend_mod
        model = args.model or "mineru-3.4-pipeline"
    elif args.backend == "vlm-vllm":
        from mineru_rocm.backends import vlm as backend_mod
        model = args.model or "mineru-2.5-pro"
    else:  # argparse choices prevents this, but be explicit
        print(f"[fatal] unknown backend: {args.backend!r}", file=sys.stderr)
        return 2

    cfg = {**config.as_dict(), "lang": args.lang, "backend": args.backend}
    return _orchestrate(
        args, infer_page=backend_mod.infer_page, backend=args.backend,
        model=model, cfg=cfg, platform=args.platform,
    )


def main(argv=None) -> int:
    args = parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `/opt/venv/bin/python -m pytest tests/test_driver.py -q`
Expected: `9 passed` (4 from Task 1 + 5 new).

- [ ] **Step 5: Validate — decoupling proof + engine contract + full suite**

Run: `/opt/venv/bin/python scripts/check_deps.py` → `P0 pyproject OK`.
Run: `/opt/venv/bin/python -c "import mineru_rocm.driver; import sys; print('engine:', 'omnidocbench_amd' in sys.modules, '| torch:', 'torch' in sys.modules)"` → `engine: False | torch: False` (lazy backend imports keep the module GPU-free).
Run: `/opt/venv/bin/python -m mineru_rocm.driver --help | grep -E '^\s+--' | wc -l` → `11` (the driver's own flags; distinct from the engine CLI).
Run: `/opt/venv/bin/python adapter/run_adapter.py --help | grep -E '^\s+--' | wc -l` → `7` (engine CLI byte-identical; P1c.2 didn't touch it).
Run: `/opt/venv/bin/python -m pytest -q` → 96 passed (91 + 5).

- [ ] **Step 6: Commit**

```bash
git add src/mineru_rocm/driver.py tests/test_driver.py
git commit -m "feat(p1c2): driver entry layer — parse_args + run (backend switch) + main; 5 tests"
```
(+ `Co-Authored-By` trailer.)

---

## Definition of Done (P1c.2)

- [ ] `import mineru_rocm.driver` succeeds with NO engine AND NO torch pulled into `sys.modules` (lazy backend imports).
- [ ] `python -m pytest -q` green: 96 total (87 from P1a+P1b+P1c + 4 orchestration + 5 entry).
- [ ] **Integration smoke passes** (P1c carry-forward): the full `preflight → select_todo → infer → commit_success → write_manifest → validate_manifest` round-trip with a fake infer, and the written manifest passes `runner.validate_manifest` (all 3 conservation laws hold).
- [ ] Resume is genuine: a pre-complete page is skipped (not re-inferred), counted as `skipped`, and `final_state.complete` reflects it.
- [ ] Per-page failure is recorded (`_errors/<stem>.json`), the run status is `failed`, and the manifest STILL passes `validate_manifest` (conservation holds on failure too).
- [ ] `run()` selects the correct backend (`pipeline`→`backends.pipeline.infer_page`, `vlm-vllm`→`backends.vlm.infer_page`) — verified by monkeypatch.
- [ ] `python -m mineru_rocm.driver --help` shows the 11 driver flags; `adapter/run_adapter.py --help` still shows the unchanged 7-flag engine CLI.
- [ ] `dispatcher.py` / `adapter/` / `backends/` / `types.py` byte-identical to main (P1c.2 is purely additive).

**Deferred (NOT P1c.2):** a real end-to-end run against actual MinerU weights + a vLLM server (needs GPU + the pinned OmniDocBench venv) — belongs to the reproducibility/results phase (P3). `endpoint_pool` + a multi-endpoint VLM driver — deferred until MinerU actually runs multi-replica VLM (YAGNI today; single-server per Plan 1/2). The unit tests prove the orchestration + runner integration via a fake infer; the real backends are exercised only through the monkeypatched routing test.

## Follow-on

- **P1d** — `cli.py` (`mineru-rocm doctor|validate|predict|score|canary|manifest verify`): the `predict` subcommand wraps `driver.parse_args`/`driver.run`; `score` wraps `scoring.score_directory`; `manifest verify` wraps `runner.validate_manifest`. + `scripts/check_repo.py` (lock↔README consistency + the `pip install -e .` smoke as the PEP 639 regression guard + structural `tomllib`/AST no-engine scan — the P1a/P1b carry-forward).
- **(future) multi-replica VLM** — if MinerU runs N vLLM servers across the 4× gfx1100, port `endpoint_pool` + a multi-endpoint VLM driver then (MinerUClient-per-endpoint; the single-endpoint driver's `_orchestrate` is reused as the per-endpoint worker core).
