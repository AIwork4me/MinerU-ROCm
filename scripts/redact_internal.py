#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
"""Redact internal infrastructure from public artefacts in results/ and docs/.

Replaces the internal HF mirror IP, the host eval-root, and the host venv path with
neutral placeholders. Idempotent. Usage: python scripts/redact_internal.py
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REPLACEMENTS = {
    "http://134.199.133.77": "<hf-mirror>",
    "134.199.133.77": "<hf-mirror>",
    "/root/ocr-eval": "<eval-root>",
    "/opt/venv": "<host-venv>",
    "u-20-8d823edc": "<hostname>",
    "/workspace/": "<workspace>/",
}
# Matches the file types scanned by check_repo.check_no_internal_infra (gate +
# redactor must agree on scope, else a .log traceback frame would leak past the gate).
SUFFIXES = (".json", ".md", ".yaml", ".yml", ".log", ".sh")
# Walk dirs: public artefact trees + the dirs that carry host-local paths in
# scripts/configs (examples/, configs/, adapter/setup/). Aligns the redactor
# with the whole-repo scan scope of check_repo.check_no_internal_infra.
WALK_DIRS = ("results", "docs", "examples", "configs", "adapter")

def main() -> int:
    changed = []
    for sub in WALK_DIRS:
        root = REPO / sub
        if not root.is_dir():
            continue
        for p in root.rglob("*"):
            if not p.is_file() or p.suffix not in SUFFIXES or "superpowers" in p.parts:
                continue
            txt = p.read_text(encoding="utf-8")
            new = txt
            for old, repl in REPLACEMENTS.items():
                new = new.replace(old, repl)
            if new != txt:
                p.write_text(new, encoding="utf-8")
                changed.append(str(p.relative_to(REPO)))
    print(f"[OK] redacted {len(changed)} file(s):")
    for c in changed:
        print(f"  - {c}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
