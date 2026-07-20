import ast
import subprocess
import sys
from pathlib import Path
import pytest

REPO = Path(__file__).resolve().parents[1]


def test_no_engine_imports_in_core_uses_ast():
    """The no-engine scan is AST-based (catches 'import omnidocbench_amd' anywhere in src/mineru_rocm)."""
    import scripts.check_repo as cr  # noqa
    leaks = cr.find_engine_imports(REPO / "src" / "mineru_rocm")
    assert leaks == [], f"engine imports leaked into core: {leaks}"


def test_required_lock_sections_present():
    import scripts.check_repo as cr
    lock = cr._load_lock()
    missing = cr.check_lock_sections(lock)
    assert missing == [], missing


def test_spdx_headers_on_src_and_scripts():
    import scripts.check_repo as cr
    assert cr.check_spdx(REPO) == []


def test_readme_script_references_exist():
    import scripts.check_repo as cr
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    assert cr.check_readme_scripts_exist(readme) == []


def test_check_readme_lock_values_pass_when_consistent():
    """The README results tables match the lock values (the drift gate)."""
    import scripts.check_repo as cr
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    lock = cr._load_lock()
    findings = cr.check_readme_lock_values(readme, lock)
    assert findings == [], findings


def test_check_readme_lock_values_flags_drift():
    """README numbers that disagree with the lock are findings (stale-table detection)."""
    import scripts.check_repo as cr
    lock = {"benchmark": {"full_1651": {"pipeline": {"overall": 99.99}, "vlm_vllm": {"overall": 88.88}}}}
    readme = "Pipeline Overall 86.48 | VLM 95.46"  # neither 99.99 nor 88.88 present
    findings = cr.check_readme_lock_values(readme, lock)
    assert len(findings) == 2  # both pipeline + vlm_vllm flagged as drift


def test_check_repo_clean_on_repo(capsys):
    """Integration gate: the FAST checks (no install smoke) all pass on the real repo.

    `check_install_smoke` runs `pip install -e .` (slow + mutates env) so it runs
    in CI via `main()`, not here. This test covers engine/lock/SPDX/README + the
    README↔lock value cross-check."""
    import scripts.check_repo as cr
    findings = []
    findings += cr.find_engine_imports(REPO / "src" / "mineru_rocm")
    findings += cr.check_lock_sections(cr._load_lock())
    findings += cr.check_spdx()
    readme = (REPO / "README.md").read_text(encoding="utf-8") if (REPO / "README.md").is_file() else ""
    findings += cr.check_readme_scripts_exist(readme)
    findings += cr.check_readme_lock_values(readme, cr._load_lock())
    findings += cr.check_modelcard_lock_agreement(cr._load_lock())
    findings += cr.check_no_stale_overall()
    findings += cr.check_no_withdrawn_anchor_claims()
    findings += cr.check_no_internal_infra()
    assert findings == [], findings


def test_upstream_commits_pinned_in_lock():
    """mineru + mineru_vl_utils carry real git commit SHAs (not 'not_recorded')."""
    import scripts.check_repo as cr
    lock = cr._load_lock()
    for dep in ("mineru", "mineru_vl_utils"):
        commit = (lock.get(dep) or {}).get("commit", "")
        assert commit != "not_recorded" and len(commit) == 40, f"{dep}.commit not pinned: {commit!r}"


def test_official_reference_verified():
    """The official anchor is sourced from the upstream README (not not_verified)."""
    import scripts.check_repo as cr
    lock = cr._load_lock()
    ref = (lock.get("benchmark") or {}).get("official_reference") or {}
    assert ref.get("source") == "verified", f"official_reference.source not verified: {ref.get('source')!r}"
    assert ref.get("pipeline_overall") == 86.47
    assert ref.get("vlm_overall") == 95.30


def test_modelcard_lock_agreement():
    """model_card.json (VLM) + model_card.pipeline.json Overall match the lock (tri-source)."""
    import json
    import scripts.check_repo as cr
    lock = cr._load_lock()
    full = (lock.get("benchmark") or {}).get("full_1651") or {}
    expected = {
        "model_card.json": (full.get("vlm_vllm") or {}).get("overall"),
        "model_card.pipeline.json": (full.get("pipeline") or {}).get("overall"),
    }
    for fname, exp in expected.items():
        if exp is None:
            continue
        card = json.loads((cr.REPO / fname).read_text(encoding="utf-8"))
        assert card["overall"] == exp, f"{fname}.overall {card['overall']} != lock {exp}"
        # artefacts must point at the authoritative v1.6 tree, not the old v16 engine tree
        arts = json.dumps(card.get("artifacts", {}))
        assert "omnidocbench/v1.6/" in arts and "omnidocbench/v16/" not in arts, f"{fname} still points at v16/"


def test_no_stale_vlm_overall_in_docs():
    """No user-facing doc under docs/ (excl. superpowers/) still quotes the stale 95.56."""
    import scripts.check_repo as cr
    findings = cr.check_no_stale_overall()
    assert findings == [], findings


def test_no_internal_infra_in_public_files():
    """No committed file under results/ or docs/ (excl. docs/superpowers/) leaks the
    internal HF mirror IP or host eval-root path."""
    import scripts.check_repo as cr
    findings = cr.check_no_internal_infra()
    assert findings == [], findings


def test_no_withdrawn_anchor_claims_in_public_files():
    """No user-facing surface re-cites the withdrawn unofficial-anchor story."""
    import scripts.check_repo as cr
    findings = cr.check_no_withdrawn_anchor_claims()
    assert findings == [], findings


def test_check_no_withdrawn_anchor_claims_flags_tokens(tmp_path):
    """The gate flags each withdrawn token in docs/ (excl superpowers) + top-level
    README/CHANGELOG, and leaves docs/superpowers/ + the lock alone."""
    import scripts.check_repo as cr
    # docs/*.md (not under superpowers/) — each token flagged once
    (tmp_path / "docs").mkdir(parents=True)
    (tmp_path / "docs" / "a.md").write_text("the old 95.75 was withdrawn\n", encoding="utf-8")
    (tmp_path / "docs" / "b.md").write_text("upstream ~95.69 guess\n", encoding="utf-8")
    # superpowers/ is exempt even if it contains the token
    (tmp_path / "docs" / "superpowers").mkdir()
    (tmp_path / "docs" / "superpowers" / "spec.md").write_text("95.75 design note\n", encoding="utf-8")
    # top-level surfaces scanned
    (tmp_path / "README.md").write_text("anchor is unverified\n", encoding="utf-8")
    (tmp_path / "CHANGELOG.md").write_text("status: not_verified\n", encoding="utf-8")
    findings = cr.check_no_withdrawn_anchor_claims(tmp_path)
    # 4 distinct flagged files (a.md, b.md, README.md, CHANGELOG.md); superpowers exempt
    flagged_files = {f.split(" re-cites ")[0] for f in findings}
    assert "docs/a.md" in flagged_files and "docs/b.md" in flagged_files
    assert "README.md" in flagged_files and "CHANGELOG.md" in flagged_files
    assert not any("superpowers" in f for f in findings), findings
