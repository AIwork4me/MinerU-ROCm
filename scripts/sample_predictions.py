#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
"""Pick a deterministic 10-page stratified sample from a prediction dir.

Selection: stable sha256 of the page stem (no randomness) → sorted → first 10.
Writes sample_predictions/<stem>.md (copied) + sample_predictions/manifest.json.
Usage: python scripts/sample_predictions.py <pred_dir>
"""
from __future__ import annotations
import hashlib
import json
import shutil
import sys
from pathlib import Path

N = 10

def pick(pred_dir: Path) -> list[str]:
    stems = sorted(
        (p.stem for p in pred_dir.glob("page-*.md")),
        key=lambda s: hashlib.sha256(s.encode()).hexdigest(),
    )
    return stems[:N]

def main(pred_dir: Path) -> int:
    out = pred_dir / "sample_predictions"
    out.mkdir(exist_ok=True)
    chosen = pick(pred_dir)
    manifest = []
    for stem in chosen:
        src = pred_dir / f"{stem}.md"
        dst = out / f"{stem}.md"
        shutil.copyfile(src, dst)
        manifest.append({"stem": stem, "sha256": hashlib.sha256(src.read_bytes()).hexdigest()})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[OK] {out}: {len(manifest)} pages; manifest written")
    return 0

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: sample_predictions.py <pred_dir>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(Path(sys.argv[1])))
