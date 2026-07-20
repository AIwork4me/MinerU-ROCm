#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
"""Repo-consistency gate. Exits 0 clean, 1 on any finding. Run in CI + locally."""
from __future__ import annotations
import ast, re, subprocess, sys
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
        # Only module-top-level statements (tree.body) — imports nested inside
        # function/class bodies are lazy by design and must NOT be flagged.
        for node in tree.body:
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


def check_readme_lock_values(readme: str, lock) -> list[str]:
    """README results tables match the lock values (the drift gate).

    The lock's full-1651 pipeline + vlm_vllm Overall must appear in the README —
    catches a stale README (e.g. README still says 95.56 after a re-run scored 95.46).
    Skipped (returns []) when the lock's Overall values aren't filled."""
    if lock is None:
        return []
    full = (lock.get("benchmark") or {}).get("full_1651") or {}
    findings = []
    for key in ("pipeline", "vlm_vllm"):
        overall = (full.get(key) or {}).get("overall")
        if overall is None:
            continue
        if f"{overall}" not in readme:
            findings.append(f"README drift: lock full_1651.{key} Overall {overall} not in README (stale results table?)")
    return findings


def check_modelcard_lock_agreement(lock) -> list[str]:
    """model_card.json (VLM) + model_card.pipeline.json Overall match the lock, and
    artefacts point at the authoritative v1.6 tree (not the superseded v16 engine tree)."""
    if lock is None:
        return []
    full = (lock.get("benchmark") or {}).get("full_1651") or {}
    findings = []
    mapping = {"model_card.json": "vlm_vllm", "model_card.pipeline.json": "pipeline"}
    for fname, key in mapping.items():
        exp = (full.get(key) or {}).get("overall")
        if exp is None:
            continue
        path = REPO / fname
        if not path.is_file():
            findings.append(f"{fname} missing")
            continue
        import json
        card = json.loads(path.read_text(encoding="utf-8"))
        if card.get("overall") != exp:
            findings.append(f"{fname}.overall {card.get('overall')} != lock full_1651.{key}.overall {exp}")
        arts = json.dumps(card.get("artifacts", {}))
        if "omnidocbench/v1.6/" not in arts or "omnidocbench/v16/" in arts:
            findings.append(f"{fname} artefacts do not point at the authoritative results/omnidocbench/v1.6/ tree")
    return findings


_STALE_VLM_OVERALL = "95.56"
_CURRENT_VLM_OVERALL = "95.46"
def check_no_stale_overall(repo=REPO) -> list[str]:
    """No user-facing doc under docs/ (excl docs/superpowers/ design records) presents
    the stale VLM Overall (95.56) as the current value. A doc is flagged iff it contains
    95.56 WITHOUT also containing 95.46 — legitimate comparative mentions (e.g. 'vs a
    prior 95.56' alongside the current 95.46) are fine."""
    errs = []
    for md in (repo / "docs").rglob("*.md"):
        if "superpowers" in md.parts:
            continue
        txt = md.read_text(encoding="utf-8")
        if _STALE_VLM_OVERALL in txt and _CURRENT_VLM_OVERALL not in txt:
            errs.append(f"{md.relative_to(repo)} quotes stale Overall {_STALE_VLM_OVERALL} without the current {_CURRENT_VLM_OVERALL}")
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
    lock = _load_lock()
    readme = (REPO / "README.md").read_text(encoding="utf-8") if (REPO / "README.md").is_file() else ""
    findings += find_engine_imports(REPO / "src" / "mineru_rocm")
    findings += check_lock_sections(lock)
    findings += check_spdx()
    findings += check_readme_scripts_exist(readme)
    findings += check_readme_lock_values(readme, lock)
    findings += check_modelcard_lock_agreement(lock)
    findings += check_no_stale_overall()
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
