# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
"""Unified ``mineru-rocm`` CLI.

A thin entry surface over the package modules — mirrors Hunyuan's
``hunyuan_ocr/cli.py`` structure (argparse subparsers -> dispatch dict ->
per-subcommand handlers), adapted to MinerU's modules. Self-contained subcommands
that work from a wheel install (no repo checkout):

  doctor             -- minimal env probe (package importable + optional deps advisory)
  validate           -- pre-score validation of a prediction dir
  manifest verify    -- conservation-law check of a run_manifest.json
  canary materialize -- rebuild the canary subset from the full GT
  predict            -- robust run path via mineru_rocm.driver (pipeline | vlm-vllm)
  score              -- OmniDocBench v1.6 scoring via mineru_rocm.scoring

Top-level imports are stdlib + ``mineru_rocm.runner`` (stdlib-only); the heavy
modules (``driver``/``scoring``/``validation``/``canary``) are imported LAZILY
inside their handlers, so ``import mineru_rocm.cli`` is GPU-free and engine-free.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

from mineru_rocm import runner
from mineru_rocm import config as _config  # noqa: F401 — config is stdlib-only; used for BACKEND info


# --- doctor ------------------------------------------------------------------


def _try_version(modname: str):
    """Return ``__version__`` of ``modname`` if importable, else None."""
    try:
        m = importlib.import_module(modname)
    except Exception:  # noqa: BLE001 — advisory probe; any import failure is "miss"
        return None
    return getattr(m, "__version__", None) or "(importable, no __version__)"


def _collect_doctor_checks() -> list[dict]:
    """Build the (label, status, detail) check list.

    ``status`` is one of ``ok``/``miss``/``info``. Only the ``mineru_rocm``
    package itself is required; the engine/GPU deps (mineru/torch/vllm/yaml) are
    advisory (reported but never fail ``--strict``).
    """
    checks: list[dict] = []

    # Required: the package itself.
    try:
        import mineru_rocm  # noqa: F401 — local import for the probe
        from mineru_rocm import __version__ as _mv
        checks.append({"label": "mineru_rocm", "status": "ok", "detail": _mv})
    except Exception as exc:  # noqa: BLE001
        checks.append({"label": "mineru_rocm", "status": "miss", "detail": f"import failed: {exc}"})

    # Advisory optional deps.
    for name in ("mineru", "torch", "vllm", "yaml"):
        v = _try_version(name)
        checks.append({"label": name, "status": "ok" if v else "miss", "detail": v or "not installed"})

    # Info: configured backend.
    checks.append({"label": "config.BACKEND", "status": "info", "detail": _config.BACKEND})
    return checks


def _doctor(args) -> int:
    checks = _collect_doctor_checks()

    if getattr(args, "json", False):
        print(json.dumps(checks, indent=2))
        # --strict exits 1 only if the package itself is missing.
        if getattr(args, "strict", False):
            pkg = next((c for c in checks if c["label"] == "mineru_rocm"), None)
            if pkg is not None and pkg["status"] != "ok":
                return 1
        return 0

    for c in checks:
        if c["status"] == "ok":
            print(f"  [OK]   {c['label']}: {c['detail']}")
        elif c["status"] == "miss":
            print(f"  [MISS] {c['label']}: {c['detail']}")
        else:
            print(f"  [INFO] {c['label']}: {c['detail']}")
    # Advisory mode: never fails on missing optional deps. --strict fails ONLY if
    # the package itself is missing (the one required check).
    if getattr(args, "strict", False):
        pkg = next((c for c in checks if c["label"] == "mineru_rocm"), None)
        if pkg is not None and pkg["status"] != "ok":
            print(f"[strict] required check failed: mineru_rocm ({pkg['detail']})", file=sys.stderr)
            return 1
    else:
        print("doctor is advisory (no --strict); optional deps are reported but never fail.")
    return 0


# --- validate / manifest / canary (package-only) -----------------------------


def _validate(args) -> int:
    from mineru_rocm import validation

    rep = validation.validate_predictions(args.gt_json, args.pred_dir, strict=not args.lenient)
    if rep.ok:
        print(f"[OK] valid={rep.valid}/{rep.expected}")
        return 0
    print(f"[FAIL] valid={rep.valid}/{rep.expected} errors={len(rep.errors())} warnings={len(rep.warnings())}")
    for prob in rep.problems:
        tag = "ERROR" if prob.severity == "error" else "WARN"
        print(f"  [{tag}] {prob.code}: {prob.message}")
    return 1


def _manifest_verify(args) -> int:
    mp = Path(args.pred_dir) / "run_manifest.json"
    if not mp.is_file():
        print(f"[error] no run_manifest.json in {args.pred_dir}", file=sys.stderr)
        return 2
    try:
        raw = mp.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"[error] cannot read {mp}: {exc}", file=sys.stderr)
        return 2
    if not raw.strip():
        print(f"[error] {mp} is empty", file=sys.stderr)
        return 1
    try:
        m = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"[error] {mp} is not valid JSON: {exc.msg} (line {exc.lineno} col {exc.colno})", file=sys.stderr)
        return 1
    if not isinstance(m, dict):
        print(f"[error] {mp} is valid JSON but not an object (got {type(m).__name__})", file=sys.stderr)
        return 1
    errs = runner.validate_manifest(m)
    if errs:
        print(f"[FAIL] manifest violates {len(errs)} invariant(s):", file=sys.stderr)
        for e in errs:
            print(f"  - {e}", file=sys.stderr)
        return 1
    print(f"[OK] manifest is valid: backend={m.get('backend')} run={m.get('run_counts')} final={m.get('final_state')}")
    return 0


def _canary_materialize(args) -> int:
    from mineru_rocm import canary

    try:
        sha = canary.materialize(args.full_gt, args.manifest, args.out)
    except canary.CanaryError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    print(f"[OK] {args.out} sha256={sha}")
    return 0


# --- predict / score (package-resident; work from a wheel install) -----------


def _predict(args) -> int:
    """Delegate to ``mineru_rocm.driver`` (the robust run path). The driver
    flags are first-class predict args (added via ``driver.add_arguments`` at
    subparser-build time), so the cli ``args`` Namespace already carries
    ``--backend`` + every driver flag — forward it straight to ``driver.run``.
    The driver imports backend deps lazily inside ``run()``, so reaching its
    arg check is GPU-free; ``command=`` threads this CLI into the run manifest."""
    from mineru_rocm import driver

    return driver.run(args, command=["mineru-rocm", "predict", args.backend])


def _score(args) -> int:
    """Score a prediction dir via the package scorer. Needs the OmniDocBench
    scorer venv (set via --venv-python or OMNIDOCBENCH_VENV). Pre-score
    validation (CPU-only) runs first and raises ScoringError on a dirty dir."""
    from mineru_rocm import scoring

    try:
        result = scoring.score_directory(
            gt_json=args.gt_json,
            pred_dir=args.pred_dir,
            omnidocbench_repo=args.omnidocbench_repo,
            venv_python=args.venv_python,
            skip_validation=args.skip_validation,
        )
    except scoring.ScoringError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    print(scoring.format_score_table(args.label, result["metrics"]))
    return 0


# --- argparse wiring ---------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mineru-rocm",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    doc = sub.add_parser("doctor", help="minimal env probe (package + optional deps advisory)")
    doc.add_argument("--strict", action="store_true", help="exit 1 only if mineru_rocm itself is missing")
    doc.add_argument("--json", action="store_true", help="emit the checks as a JSON list")

    v = sub.add_parser("validate", help="validate a prediction dir against GT")
    v.add_argument("--gt-json", required=True)
    v.add_argument("--pred-dir", required=True)
    v.add_argument("--lenient", action="store_true", help="warnings are non-fatal")

    man = sub.add_parser("manifest", help="run-manifest utilities")
    msub = man.add_subparsers(dest="mcmd", required=True)
    mv = msub.add_parser("verify", help="verify run_manifest.json conservation laws")
    mv.add_argument("--pred-dir", required=True)

    can = sub.add_parser("canary", help="canary-subset utilities")
    csub = can.add_subparsers(dest="ccmd", required=True)
    cmv = csub.add_parser("materialize", help="rebuild the canary subset from the full GT")
    cmv.add_argument("--full-gt", required=True)
    cmv.add_argument("--manifest", required=True)
    cmv.add_argument("--out", required=True)

    pr = sub.add_parser(
        "predict",
        help="robust run path via mineru_rocm.driver (pipeline | vlm-vllm)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Driver flags (--gt-json, --images-dir, --pred-dir, --model, --platform, "
            "--lang, --max-retries, --retry-backoff, --overwrite, --retry-failed) are "
            "first-class predict args, e.g.:\n"
            "    mineru-rocm predict --backend pipeline --gt-json g.json "
            "--images-dir i --pred-dir p\n"
            "Flags are forwarded directly to mineru_rocm.driver.run (no literal '--' needed)."
        ),
    )
    # The driver flags are added via the shared driver.add_arguments so the cli
    # `args` Namespace carries them directly (lazy-import: add_arguments only
    # registers argparse metadata, it does NOT import torch/backend deps).
    from mineru_rocm import driver as _driver

    _driver.add_arguments(pr)

    sc = sub.add_parser("score", help="OmniDocBench v1.6 scoring (scorer venv required)")
    sc.add_argument("--pred-dir", required=True)
    sc.add_argument("--gt-json", required=True)
    sc.add_argument("--label", default="score")
    sc.add_argument("--omnidocbench-repo", default=None)
    sc.add_argument("--venv-python", default=None)
    sc.add_argument("--skip-validation", action="store_true", help="DANGEROUS: bypass pre-score validation")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    dispatch = {
        "doctor": _doctor,
        "validate": _validate,
        "manifest": lambda a: _manifest_verify(a) if a.mcmd == "verify" else 2,
        "canary": lambda a: _canary_materialize(a) if a.ccmd == "materialize" else 2,
        "predict": _predict,
        "score": _score,
    }
    handler = dispatch[args.cmd]
    rc = handler(args)
    return int(rc or 0)


if __name__ == "__main__":
    sys.exit(main())
