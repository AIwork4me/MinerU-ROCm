from pathlib import Path
import tomllib

import yaml

import mineru_rocm


ROOT = Path(__file__).resolve().parents[1]


def test_release_versions_agree():
    project = tomllib.loads(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    citation = yaml.safe_load(
        (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    )
    assert project["version"] == "1.0.0"
    assert mineru_rocm.__version__ == project["version"]
    assert citation["version"] == project["version"]


def test_release_documents_exist():
    for name in (
        "architecture.md",
        "hardware-matrix.md",
        "release-artifact.md",
        "release-checklist.md",
    ):
        assert (ROOT / "docs" / name).is_file()
