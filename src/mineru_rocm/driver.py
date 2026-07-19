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


def _orchestrate(args, *, infer_page, backend: str, model: str, cfg: dict, platform: str = "linux-rocm", command: list[str] | None = None) -> int:
    """Run ``infer_page`` over the OmniDocBench page set with full runner integrity.

    ``infer_page(img, platform, cfg) -> str`` is injected so the orchestration is
    CPU-testable without a GPU. Returns 0 on a fully-ok run, 1 otherwise (failed
    pages, pending pages, or a pre-run abort). Writes ``run_manifest.json`` on
    every run that completes the loop body (not on a conflict abort or a mid-run crash; resume recovers partial progress).
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
            command=command,
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


def add_arguments(parser):
    """Add the driver's CLI flags to ``parser``.

    Shared by ``parse_args`` (the ``python -m mineru_rocm.driver`` path) and the
    ``mineru-rocm predict`` CLI subparser, so the driver flags are first-class
    predict args (no argparse REMAINDER / literal ``--`` needed at the CLI layer).
    """
    parser.add_argument("--gt-json", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--backend", required=True, choices=["pipeline", "vlm-vllm"])
    parser.add_argument("--model", default=None, help="advisory model name for the manifest (default per backend)")
    parser.add_argument("--platform", default="linux-rocm")
    parser.add_argument("--lang", default="ch")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-backoff", type=float, default=2.0)
    parser.add_argument("--overwrite", action="store_true", help="re-run every page (ignore existing complete pages)")
    parser.add_argument("--retry-failed", action="store_true", help="re-run only previously-failed pages")


def parse_args(argv=None):
    """CLI for `python -m mineru_rocm.driver`. The P1d `mineru-rocm predict` CLI forwards its flags via add_arguments."""
    import argparse

    p = argparse.ArgumentParser(
        prog="mineru_rocm.driver",
        description="Run a MinerU backend over an OmniDocBench page set (robust: atomic writes + run_manifest + resume).",
    )
    add_arguments(p)
    return p.parse_args(argv)


def run(args, command=None) -> int:
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
        model=model, cfg=cfg, platform=args.platform, command=command,
    )


def main(argv=None) -> int:
    args = parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
