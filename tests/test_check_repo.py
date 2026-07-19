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


def test_check_repo_clean_on_repo(capsys):
    """Integration gate: the FAST checks (no install smoke) all pass on the real repo.

    `check_install_smoke` runs `pip install -e .` (slow + mutates env) so it runs
    in CI via `main()`, not here. This test covers the engine/lock/SPDX/README checks."""
    import scripts.check_repo as cr
    findings = []
    findings += cr.find_engine_imports(REPO / "src" / "mineru_rocm")
    findings += cr.check_lock_sections(cr._load_lock())
    findings += cr.check_spdx()
    readme = (REPO / "README.md").read_text(encoding="utf-8") if (REPO / "README.md").is_file() else ""
    findings += cr.check_readme_scripts_exist(readme)
    assert findings == [], findings
