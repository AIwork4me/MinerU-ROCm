#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
"""Repo-consistency gate. Exits 0 clean, 1 on any finding. Run in CI + locally."""
from __future__ import annotations
import ast
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ENGINE_MODULES = ("omnidocbench_rocm", "torch", "mineru", "mineru_vl_utils", "openai", "vllm")
REQUIRED_LOCK_SECTIONS = ("mineru_rocm", "mineru", "model", "omnidocbench", "environment", "benchmark")


def find_engine_imports(pkg_dir: Path) -> list[str]:
    """AST scan: no ENGINE_MODULES imported at module top-level anywhere under pkg_dir.

    (Catches `import omnidocbench_rocm`, `from omnidocbench_rocm import x`, `import torch as t` —
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
    artefacts point at the authoritative v1.6 tree (not the superseded v16 engine tree),
    and every repo-relative artefact path resolves under the repo (no dangling refs)."""
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
        arts = card.get("artifacts", {}) or {}
        arts_blob = json.dumps(arts)
        if "omnidocbench/v1.6/" not in arts_blob and "omnidocbench/v16/" not in arts_blob:
            findings.append(f"{fname} artefacts do not point at an omnidocbench results tree")
        # Every artefact value that looks like a repo-relative path (not a URL or a
        # <placeholder>) must resolve under the repo — catches dangling refs such as a
        # predict.log path when only predict.log.tail is tracked.
        for art_name, art_val in arts.items():
            if not isinstance(art_val, str):
                continue
            if "://" in art_val or art_val.startswith("<") or "(" in art_val:
                continue  # URL, <placeholder>, or "<x> (note)" — not a literal path
            ap = REPO / art_val
            if not ap.exists():
                findings.append(f"{fname} artefact {art_name!r} -> {art_val!r} does not resolve under the repo (dangling ref)")
    return findings


def _current_vlm_overall(lock):
    """The canonical current VLM Overall, sourced from the lock (single source
    of truth). None when not filled."""
    return (((lock.get("benchmark") or {}).get("full_1651") or {}).get("vlm_vllm") or {}).get("overall")


def _prior_vlm_overall(repo=REPO):
    """The prior standalone VLM Overall, read from the legacy v1.6 metric_result
    (overall_notebook). None when absent. Not hard-coded — derived from the
    committed legacy metric so the gate has no baked-in score."""
    import json
    p = repo / "results" / "omnidocbench" / "v1.6" / "vlm-vllm" / "metric_result.json"
    if not p.is_file():
        return None
    try:
        return round(float(json.loads(p.read_text(encoding="utf-8")).get("overall_notebook")), 2)
    except (ValueError, TypeError):
        return None


def check_current_overall_primary(lock, repo=REPO) -> list[str]:
    """The lock's current VLM Overall is the primary headline number: the VLM
    badge in README.md and README.zh-CN.md must state it. Data-driven — the
    'current' value is whatever the lock records, never hard-coded."""
    current = _current_vlm_overall(lock)
    if current is None:
        return []
    errs: list[str] = []
    cur = f"{current}"
    for name in ("README.md", "README.zh-CN.md"):
        p = repo / name
        if not p.is_file():
            continue
        badge_lines = [line for line in p.read_text(encoding="utf-8").splitlines()
                       if "img.shields.io" in line and "VLM" in line and "(full)" in line]
        if badge_lines and not any(cur in line for line in badge_lines):
            errs.append(f"{name}: VLM badge does not state the current Overall {cur} (lock value)")
    return errs


def check_prior_overall_contextual(lock, repo=REPO) -> list[str]:
    """The prior standalone VLM Overall (read from the legacy v1.6 metric) may
    appear in user-facing docs only alongside the current Overall. Catches a doc
    that quotes the prior as the sole/primary number. Data-driven — no
    hard-coded scores."""
    current = _current_vlm_overall(lock)
    prior = _prior_vlm_overall(repo)
    if current is None or prior is None or current == prior:
        return []
    errs: list[str] = []
    cur, pri = f"{current}", f"{prior}"
    targets: list[Path] = [repo / n for n in ("README.md", "README.zh-CN.md", "CHANGELOG.md")
                           if (repo / n).is_file()]
    for md in (repo / "docs").rglob("*.md"):
        if "superpowers" not in md.parts:
            targets.append(md)
    for p in targets:
        txt = p.read_text(encoding="utf-8", errors="ignore")
        if pri in txt and cur not in txt:
            errs.append(f"{p.relative_to(repo)} quotes prior VLM Overall {pri} without the current {cur}")
    return errs


# The withdrawn unofficial-anchor story (95.75 / ~95.69 / not_verified / "anchor is
# unverified") was superseded by the verified upstream-README anchors (vlm-engine
# 95.30, pipeline 86.47). These tokens MUST NOT recur in any user-facing surface.
# (The lock is NOT scanned — it legitimately carries verified/not_recorded.)
_WITHDRAWN_ANCHOR_TOKENS = ("95.75", "~95.69", "not_verified", "anchor is unverified")
_WITHDRAWN_ANCHOR_FILES = ("README.md", "README.zh-CN.md", "CHANGELOG.md")
def check_no_withdrawn_anchor_claims(repo=REPO) -> list[str]:
    """No user-facing surface re-cites the withdrawn unofficial-anchor story.

    Scans docs/*.md (excl docs/superpowers/ design records) + the top-level
    README.md / README.zh-CN.md / CHANGELOG.md for the withdrawn tokens
    (`95.75`, `~95.69`, `not_verified`, `anchor is unverified`). The
    reproducibility lock is deliberately NOT scanned — it legitimately carries
    `verified` / `not_recorded` as field values."""
    errs = []
    targets: list[Path] = []
    for md in (repo / "docs").rglob("*.md"):
        if "superpowers" in md.parts:
            continue
        targets.append(md)
    for name in _WITHDRAWN_ANCHOR_FILES:
        p = repo / name
        if p.is_file():
            targets.append(p)
    for p in targets:
        txt = p.read_text(encoding="utf-8", errors="ignore")
        for tok in _WITHDRAWN_ANCHOR_TOKENS:
            if tok in txt:
                errs.append(f"{p.relative_to(repo)} re-cites withdrawn-anchor token {tok!r} (use the verified upstream-README anchors: vlm-engine 95.30, pipeline 86.47)")
    return errs


# Key-version consistency + ROCm-overclaim guardrail. The lock is the source of
# truth for ROCm version + GPU arch; README.md and the upstream issue draft must
# state the same values (catches 'ROCm 7.2' in the lock drifting to 'ROCm 7.2+'
# in a doc). Assertion-form overclaims are forbidden; honest negations use
# different phrasing ('not an official-support claim', 'other RDNA3 variants
# untested') and are not matched, so they pass.
_OVERCLAIM_PATTERNS = ("ROCm 7.2+", "ROCm 7.x", "officially support", "all RDNA3", "full AMD ROCm support",
                       "is parity with", "within vLLM non-determinism",
                       # provenance / over-attribution anti-drift (issue #5288 round 3):
                       "results-producing tree",             # annotated-tag object SHA mis-called a commit
                       "only ROCm-specific configuration",   # understates the pinned ROCm dependency stack
                       "bf16 matmul kernel non-determinism", # unproven run-to-run drift root cause
                       "sparse pages",                       # unproven empty-output characterization
                       "Server flags are recorded in the lock")  # the lock records a summary, not the raw flags


def check_version_consistency(lock, repo=REPO) -> list[str]:
    """README, the upstream issue draft, and the lock agree on the key versions,
    and no user-facing surface carries an assertion-form ROCm/version overclaim.

    Consistency: the lock's ``environment.rocm_hip`` and ``environment.gpu_arch``
    must each appear in README.md and ``docs/upstream/mineru-issue-5288.md``.
    Overclaim guard: forbids ``ROCm 7.2+`` / ``ROCm 7.x`` / ``officially support``
    / ``all RDNA3`` / ``full AMD ROCm support`` in README + README.zh-CN + docs/**
    (excl. ``docs/superpowers/``). Only what was actually tested (ROCm 7.2,
    gfx1100/W7900) may be claimed.
    """
    findings: list[str] = []
    # 1. consistency: README + issue draft state the lock's ROCm version + GPU arch
    if lock is not None:
        env = lock.get("environment") or {}
        rocm = env.get("rocm_hip")
        gpu = env.get("gpu_arch") or ((lock.get("rocm_recipe") or {}).get("gpu_arch"))
        for fname in ("README.md", "docs/upstream/mineru-issue-5288.md"):
            p = repo / fname
            if not p.is_file():
                continue
            txt = p.read_text(encoding="utf-8")
            if rocm and f"ROCm {rocm}" not in txt:
                findings.append(f"{fname}: does not state the lock's ROCm version (expected 'ROCm {rocm}')")
            if gpu and gpu not in txt:
                findings.append(f"{fname}: does not state the lock's GPU arch {gpu}")
    # 2. overclaim guard: assertion-form overclaims in README + README.zh-CN + docs/*.md (excl superpowers)
    targets: list[Path] = [repo / n for n in ("README.md", "README.zh-CN.md") if (repo / n).is_file()]
    for md in (repo / "docs").rglob("*.md"):
        if "superpowers" not in md.parts:
            targets.append(md)
    for p in targets:
        if not p.is_file():
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        for pat in _OVERCLAIM_PATTERNS:
            if pat in txt:
                findings.append(f"{p.relative_to(repo)}: overclaim pattern {pat!r} (scope claims to what was tested)")
    return findings


def _git(repo: Path, *args: str):
    """Run a git command in `repo`; return (rc, stdout-strip). Never raises."""
    try:
        cp = subprocess.run(["git", "-C", str(repo), *args],
                            capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return 1, ""
    return cp.returncode, (cp.stdout or "").strip()


def check_release_and_run_provenance(lock, repo=REPO) -> list[str]:
    """The lock's tag / release / run-commit provenance matches git AND the run
    manifests, and no user-facing doc calls the annotated-tag object SHA a commit.

    Three DISTINCT commits are enforced:
      release.commit         = `git rev-parse '<tag>^{commit}'` (install this for stable code)
      release.tag_object_sha = `git rev-parse <tag>` (the ANNOTATED TAG object; NOT a commit)
      benchmark_run_commits.* = each run_manifest.json `repo_commit` (the two runs differ)
    """
    findings: list[str] = []
    if lock is None:
        return findings
    import json
    mr = lock.get("mineru_rocm") or {}
    release = mr.get("release") or {}
    run_commits = mr.get("benchmark_run_commits") or {}
    tag = release.get("tag")
    if tag:
        # 1. release.commit == peeled tag commit
        rc, peeled = _git(repo, "rev-parse", f"{tag}^{{commit}}")
        if rc == 0 and release.get("commit") and peeled != release["commit"]:
            findings.append(f"lock mineru_rocm.release.commit {release['commit']} != git rev-parse '{tag}^{{commit}}' = {peeled}")
        # 2. annotated tag: tag_object_sha == git rev-parse <tag>; cat-file -t must be 'tag'
        rc2, obj = _git(repo, "rev-parse", tag)
        rc3, typ = _git(repo, "cat-file", "-t", tag)
        if rc3 == 0:  # only judge the tag type when git actually ran
            if typ == "tag":
                if release.get("tag_object_sha") and obj != release["tag_object_sha"]:
                    findings.append(f"lock mineru_rocm.release.tag_object_sha {release['tag_object_sha']} != git rev-parse {tag} = {obj}")
            elif release.get("tag_object_sha"):
                findings.append(f"{tag} is not an annotated tag (cat-file -t = {typ!r}) but lock records tag_object_sha {release['tag_object_sha']}")
    # 3-4. each benchmark run commit matches its run_manifest.json repo_commit
    for backend, key in (("pipeline", "pipeline"), ("vlm-vllm", "vlm_vllm")):
        mp = repo / "results" / "omnidocbench" / "v1.6" / backend / "run_manifest.json"
        locked = run_commits.get(key)
        if not mp.is_file() or not locked:
            continue
        try:
            manifest_commit = json.loads(mp.read_text(encoding="utf-8")).get("repo_commit")
        except (OSError, ValueError):
            continue
        if manifest_commit != locked:
            findings.append(f"lock mineru_rocm.benchmark_run_commits.{key} {locked} != {backend} run_manifest.repo_commit {manifest_commit}")
    # 5. the annotated-tag object SHA must not appear in user-facing docs as a commit
    tag_sha = release.get("tag_object_sha")
    if tag_sha:
        toks = {tag_sha, tag_sha[:7]}
        for name in ("README.md", "README.zh-CN.md", "docs/upstream/mineru-issue-5288.md", "docs/reproducibility.md"):
            p = repo / name
            if not p.is_file():
                continue
            txt = p.read_text(encoding="utf-8")
            hit = next((t for t in toks if t in txt), None)
            if hit:
                findings.append(f"{name} cites annotated-tag object SHA {hit} (a tag object, not a commit — use release.commit)")
    return findings


def check_score_commands_have_scorer_args(repo=REPO) -> list[str]:
    """Every `mineru-rocm score` example in the user-facing docs (and the lock's
    rocm_recipe.cli) passes the OmniDocBench scorer repo. The score step has NO
    machine-private default for the repo — it must be supplied via
    `--omnidocbench-repo` or `OMNIDOCBENCH_REPO`."""
    findings: list[str] = []
    targets = [repo / n for n in ("README.md", "docs/upstream/mineru-issue-5288.md",
                                  "docs/reproducibility.md", "docs/benchmark-methodology.md")
               if (repo / n).is_file()]
    for p in targets:
        txt = p.read_text(encoding="utf-8")
        for block in re.findall(r"```[a-zA-Z]*\n(.*?)```", txt, re.DOTALL):
            if not re.search(r"mineru-rocm\s+score\b", block):
                continue
            if "--omnidocbench-repo" not in block and "OMNIDOCBENCH_REPO" not in block:
                findings.append(f"{p.relative_to(repo)}: `mineru-rocm score` example lacks --omnidocbench-repo / OMNIDOCBENCH_REPO")
    lockp = repo / "reproducibility.lock.yaml"
    if lockp.is_file():
        for line in lockp.read_text(encoding="utf-8").splitlines():
            if "mineru-rocm score" in line and "--omnidocbench-repo" not in line:
                findings.append("reproducibility.lock.yaml: rocm_recipe.cli score line lacks --omnidocbench-repo")
    return findings


_LEAK_PATTERNS = ("134.199.133.77", "/root/ocr-eval", "/opt/venv", "u-20-8d823edc", "/workspace/")
# Whole-repo text scan: covers source code, configs, scripts, and public docs
# alike (the original results/+docs.+lock scope missed src/ and examples/).
_LEAK_SUFFIXES = (".sh", ".py", ".yaml", ".yml", ".json", ".md", ".log", ".txt", ".cfg", ".ini", ".toml", ".jinja", ".j2")
# This very gate and scripts/redact_internal.py define the leak patterns (and
# the placeholder mapping) by name; they MUST contain the literal strings to
# function, so they are excluded from the scan.
_LEAK_SELF_EXEMPT = ("scripts/check_repo.py", "scripts/redact_internal.py")


def _git_ls_text_files(repo: Path, suffixes: tuple[str, ...]) -> list[Path]:
    """Return tracked text files (by suffix). Falls back to an rglob walk if git
    is unavailable, skipping nothing here (exclusions applied by the caller)."""
    try:
        cp = subprocess.run(
            ["git", "-C", str(repo), "ls-files"],
            capture_output=True, text=True, check=False,
        )
        if cp.returncode == 0:
            files = []
            for line in cp.stdout.splitlines():
                if not line:
                    continue
                p = repo / line
                if p.is_file() and p.suffix in suffixes:
                    files.append(p)
            return files
    except FileNotFoundError:
        pass
    # Fallback: walk the tree (skips .git via the caller's exclusion check).
    files = []
    for p in repo.rglob("*"):
        if p.is_file() and p.suffix in suffixes:
            files.append(p)
    return files


def check_no_internal_infra(repo=REPO) -> list[str]:
    """No tracked text file in the repo leaks internal infra (HF mirror IP, host
    eval-root, host venv). Scans the WHOLE repo so source code, configs, and
    shell scripts are covered (the original results/+docs.+lock scope missed
    src/examples/adapter/configs).

    Exclusions:
      - ``.git/`` — VCS internals, not authored content.
      - ``docs/superpowers/**`` — design records that legitimately reference the
        patterns (specs, plans, this task's brief).
      - the gate + redactor themselves (``scripts/check_repo.py``,
        ``scripts/redact_internal.py``) — they define the patterns as the
        scan-targets / replacement keys and must contain the literals to work.
    """
    errs = []
    targets = _git_ls_text_files(repo, _LEAK_SUFFIXES)
    for p in targets:
        try:
            rel = p.relative_to(repo).as_posix()
        except ValueError:
            continue
        # Exclude .git/, docs/superpowers/, and the defining scripts.
        if rel.startswith(".git/") or rel in _LEAK_SELF_EXEMPT or rel.startswith("docs/superpowers/"):
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        for pat in _LEAK_PATTERNS:
            if pat in txt:
                errs.append(f"{rel} leaks internal infra pattern {pat!r}")
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
    findings += check_current_overall_primary(lock)
    findings += check_prior_overall_contextual(lock)
    findings += check_no_withdrawn_anchor_claims()
    findings += check_version_consistency(lock)
    findings += check_release_and_run_provenance(lock)
    findings += check_score_commands_have_scorer_args()
    findings += check_no_internal_infra()
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
