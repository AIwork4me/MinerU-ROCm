#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
"""P0 validator: core is GPU/platform-free; omnidocbench-rocm is only in [platform]."""
import sys, tomllib
from pathlib import Path

with open(Path(__file__).resolve().parents[1] / "pyproject.toml", "rb") as f:
    p = tomllib.load(f)
proj = p["project"]
assert proj["dependencies"] == ["PyYAML>=6.0"], f"core deps must be exactly [PyYAML>=6.0] (scoring needs yaml); got {proj['dependencies']!r}"
extras = proj["optional-dependencies"]
assert extras.get("platform") == ["omnidocbench-rocm>=0.2.0"], f"[platform] wrong: {extras.get('platform')!r}"
assert "omnidocbench-rocm" not in extras.get("dev", []), "[dev] must not pull omnidocbench-rocm (use [platform])"
assert proj["license"] == "Apache-2.0", f"license must be Apache-2.0, got {proj['license']!r}"
assert proj["urls"]["Upstream"] == "https://github.com/opendatalab/MinerU", "Upstream URL missing"

# P1a: src/mineru_rocm package exists, is src-layout, and core has no engine import.
root = Path(__file__).resolve().parents[1]
pkg = root / "src" / "mineru_rocm" / "__init__.py"
assert pkg.is_file(), f"package missing: {pkg}"
assert (root / "pyproject.toml").read_text().find('[tool.setuptools.packages.find]') != -1, "src-layout not declared"
# core package must not import the platform engine at module top level
for py in (root / "src" / "mineru_rocm").rglob("*.py"):
    src = py.read_text()
    assert "omnidocbench_rocm" not in src, f"engine import leaked into package: {py}"

print("P0 pyproject OK")
