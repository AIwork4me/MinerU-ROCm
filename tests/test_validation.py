from pathlib import Path
from mineru_rocm.validation import (
    Report, Problem, validate_predictions, ERROR_PREFIX, _OWN_ARTIFACTS,
)


def _gt(tmp_path, stems):
    import json
    gt = tmp_path / "gt.json"
    gt.write_text(
        json.dumps([{"page_info": {"image_path": f"{s}.png"}} for s in stems]),
        encoding="utf-8",
    )
    return gt


def test_constants_exact():
    assert ERROR_PREFIX == "ERROR:"
    assert "_errors" in _OWN_ARTIFACTS and "run_manifest.json" in _OWN_ARTIFACTS


def test_clean(tmp_path):
    gt = _gt(tmp_path, ["a", "b"])
    pred = tmp_path / "pred"; pred.mkdir()
    (pred / "a.md").write_text("# a"); (pred / "b.md").write_text("# b")
    r = validate_predictions(gt, pred)
    assert r.expected == 2 and r.valid == 2
    assert r.ok and r.ok_strict


def test_missing(tmp_path):
    gt = _gt(tmp_path, ["a", "b"])
    pred = tmp_path / "pred"; pred.mkdir()
    (pred / "a.md").write_text("# a")  # b missing
    r = validate_predictions(gt, pred)
    assert not r.ok
    assert any(p.code == "missing" for p in r.errors())


def test_empty_and_error_marker(tmp_path):
    gt = _gt(tmp_path, ["a", "b"])
    pred = tmp_path / "pred"; pred.mkdir()
    (pred / "a.md").write_text("")                 # empty -> error
    (pred / "b.md").write_text("ERROR: boom")      # error marker
    r = validate_predictions(gt, pred)
    codes = {p.code for p in r.errors()}
    assert "empty" in codes and "error_marker" in codes


def test_partial_and_unexpected(tmp_path):
    gt = _gt(tmp_path, ["a"])
    pred = tmp_path / "pred"; pred.mkdir()
    (pred / "a.md").write_text("# a")
    (pred / "leftover.partial").write_text("x")    # partial -> error
    (pred / "strange.txt").write_text("z")          # unexpected file -> warning
    r = validate_predictions(gt, pred)
    assert any(p.code == "partial" for p in r.errors())
    assert any(p.code == "unexpected_file" and p.severity == "warning" for p in r.problems)


def test_own_artifacts_tolerated(tmp_path):
    gt = _gt(tmp_path, ["a"])
    pred = tmp_path / "pred"; pred.mkdir()
    (pred / "a.md").write_text("# a")
    (pred / "run_manifest.json").write_text("{}")  # owned -> no warning
    (pred / "_errors").mkdir()                      # owned dir -> no warning
    r = validate_predictions(gt, pred)
    assert r.ok_strict
