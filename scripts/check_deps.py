#!/usr/bin/env python3
"""P0 validator: core is GPU/platform-free; omnidocbench-amd is only in [platform]."""
import sys, tomllib
from pathlib import Path

with open(Path(__file__).resolve().parents[1] / "pyproject.toml", "rb") as f:
    p = tomllib.load(f)
proj = p["project"]
assert proj["dependencies"] == [], f"core must have no deps, got {proj['dependencies']!r}"
extras = proj["optional-dependencies"]
assert extras.get("platform") == ["omnidocbench-amd>=0.1.0"], f"[platform] wrong: {extras.get('platform')!r}"
assert "omnidocbench-amd" not in extras.get("dev", []), "[dev] must not pull omnidocbench-amd (use [platform])"
assert proj["license"] == "Apache-2.0", f"license must be Apache-2.0, got {proj['license']!r}"
assert proj["urls"]["Upstream"] == "https://github.com/opendatalab/MinerU", "Upstream URL missing"
print("P0 pyproject OK")
