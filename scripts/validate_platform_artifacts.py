#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
"""Validate the committed platform-standard result bundles.

Thin wrapper around ``omnidocbench_rocm.bundle_validator.validate_bundle`` over
``results/omnidocbench/v16/linux-rocm``, cross-checked against ``model_card.json``.
Exit 0 = CONFORMANT, 1 = NON-CONFORMANT. Run in CI (platform-contract job) + locally.

The engine import is lazy (inside ``main``) so importing this module never
requires ``omnidocbench-rocm`` to be installed — the core package stays engine-free.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    results_dir = REPO / "results" / "omnidocbench" / "v16" / "linux-rocm"
    model_card = REPO / "model_card.json"
    from omnidocbench_rocm.bundle_validator import validate_bundle

    report = validate_bundle(results_dir, model_card=(model_card if model_card.is_file() else None))
    bundles = sorted(p.name for p in results_dir.glob("*_run_summary.json"))
    if report.ok:
        print(f"validate_platform_artifacts: CONFORMANT ({len(bundles)} bundle(s))")
        return 0
    print("validate_platform_artifacts: NON-CONFORMANT:")
    for f in report.failures:
        print(" -", f)
    return 1


if __name__ == "__main__":
    sys.exit(main())
