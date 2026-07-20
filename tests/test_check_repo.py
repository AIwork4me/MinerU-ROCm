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
    findings += cr.check_version_consistency(cr._load_lock())
    findings += cr.check_release_and_run_provenance(cr._load_lock())
    findings += cr.check_score_commands_have_scorer_args()
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


def test_modelcard_dangling_artefref_flagged(tmp_path, monkeypatch):
    """check_modelcard_lock_agreement flags an artefact path that does not resolve
    under the repo (e.g. predict.log when only predict.log.tail is tracked), while
    leaving URLs + <placeholder> values alone."""
    import json
    import scripts.check_repo as cr
    # Build a minimal repo tree: model_card.json with one good ref, one dangling
    # repo-relative path, one URL, and one <placeholder>.
    (tmp_path / "model_card.json").write_text(json.dumps({
        "overall": 95.46,
        "artifacts": {
            "metric_result": "results/omnidocbench/v1.6/vlm-vllm/metric_result.json",  # will NOT exist here
            "predict_log": "results/omnidocbench/v1.6/vlm-vllm/predict.log",            # dangling
            "sample_predictions": "results/omnidocbench/v1.6/vlm-vllm/sample_predictions/",
            "upstream_url": "https://example.com/x.json",                                # URL — skip
            "pred_dir": "<your-pred-dir>",                                               # placeholder — skip
        },
    }), encoding="utf-8")
    # Create only sample_predictions/ so exactly one ref resolves; the other two
    # repo-relative paths (metric_result.json, predict.log) dangle.
    (tmp_path / "results" / "omnidocbench" / "v1.6" / "vlm-vllm" / "sample_predictions").mkdir(parents=True)
    monkeypatch.setattr(cr, "REPO", tmp_path)
    lock = {"benchmark": {"full_1651": {"vlm_vllm": {"overall": 95.46}}}}
    findings = cr.check_modelcard_lock_agreement(lock)
    joined = "\n".join(findings)
    assert "predict_log" in joined and "metric_result" in joined, findings
    assert "upstream_url" not in joined and "pred_dir" not in joined, findings


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


def test_version_consistency_clean_on_repo():
    """README + the issue draft state the lock's ROCm version + GPU arch, and no
    user-facing surface carries a ROCm/version overclaim."""
    import scripts.check_repo as cr
    findings = cr.check_version_consistency(cr._load_lock())
    assert findings == [], findings


def test_version_consistency_flags_overclaim(tmp_path, monkeypatch):
    """An assertion-form overclaim is flagged; docs/superpowers/ is exempt; the
    consistency check passes when README states the lock's versions."""
    import scripts.check_repo as cr
    (tmp_path / "README.md").write_text("tested on ROCm 7.2 + gfx1100\n", encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text("we officially support all RDNA3 GPUs\n", encoding="utf-8")
    (tmp_path / "docs" / "superpowers").mkdir()
    (tmp_path / "docs" / "superpowers" / "s.md").write_text("ROCm 7.2+ design note\n", encoding="utf-8")
    monkeypatch.setattr(cr, "REPO", tmp_path)
    lock = {"environment": {"rocm_hip": "7.2", "gpu_arch": "gfx1100"}}
    findings = cr.check_version_consistency(lock, tmp_path)
    joined = "\n".join(findings)
    # docs/a.md overclaim flagged (officially support + all RDNA3); superpowers exempt
    assert "docs/a.md" in joined and ("officially support" in joined or "all RDNA3" in joined), findings
    assert not any("superpowers" in f for f in findings), findings


def test_release_and_run_provenance_clean_on_repo():
    """The lock's release tag/commit/tag-object SHA + both run commits match git
    and the manifests on the real repo (Task 1 provenance gate)."""
    import scripts.check_repo as cr
    findings = cr.check_release_and_run_provenance(cr._load_lock())
    assert findings == [], findings


def test_release_and_run_provenance_flags_mismatched_lock():
    """A lock whose release/run commits disagree with git + the manifests is flagged."""
    import scripts.check_repo as cr
    bad = {"mineru_rocm": {
        "release": {"tag": "v0.1.0", "commit": "0" * 40, "tag_object_sha": "1" * 40},
        "benchmark_run_commits": {"pipeline": "2" * 40, "vlm_vllm": "3" * 40},
    }}
    findings = cr.check_release_and_run_provenance(bad, cr.REPO)
    joined = "\n".join(findings)
    assert "release.commit" in joined, findings
    assert "tag_object_sha" in joined, findings
    assert "benchmark_run_commits.pipeline" in joined, findings
    assert "benchmark_run_commits.vlm_vllm" in joined, findings


def test_provenance_flags_tag_sha_cited_as_commit(tmp_path, monkeypatch):
    """The annotated-tag object SHA must not appear in a user-facing doc as a commit."""
    import scripts.check_repo as cr
    tag_sha = "dd591469d009cac246f5090daa7398623d2fd878"
    d = tmp_path / "docs" / "upstream"
    d.mkdir(parents=True)
    (d / "mineru-issue-5288.md").write_text(f"checkout v0.1.0 = {tag_sha}\n", encoding="utf-8")
    monkeypatch.setattr(cr, "REPO", tmp_path)
    lock = {"mineru_rocm": {"release": {"tag": "v0.1.0", "tag_object_sha": tag_sha}}}
    findings = cr.check_release_and_run_provenance(lock, tmp_path)
    assert any("tag object, not a commit" in f for f in findings), findings


def test_score_commands_have_scorer_args_clean_on_repo():
    """Every `mineru-rocm score` example carries the scorer repo on the real repo."""
    import scripts.check_repo as cr
    findings = cr.check_score_commands_have_scorer_args()
    assert findings == [], findings


def test_score_commands_flag_missing_scorer_arg(tmp_path, monkeypatch):
    """A `mineru-rocm score` block without --omnidocbench-repo / OMNIDOCBENCH_REPO is
    flagged; one with either passes."""
    import scripts.check_repo as cr
    (tmp_path / "README.md").write_text(
        "```bash\nmineru-rocm score --gt-json g --pred-dir p --label x\n```\n", encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "ok.md").write_text(
        "```bash\nexport OMNIDOCBENCH_REPO=/x\nmineru-rocm score --gt-json g --pred-dir p --label x\n```\n",
        encoding="utf-8")
    monkeypatch.setattr(cr, "REPO", tmp_path)
    findings = cr.check_score_commands_have_scorer_args(tmp_path)
    joined = "\n".join(findings)
    assert "README.md" in joined, findings          # the bad block is flagged
    assert "docs/ok.md" not in joined, findings      # the OMNIDOCBENCH_REPO block passes
