import json
import sys
import pytest
from mineru_rocm import cli, runner


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as ei:
        cli.main(["--help"])
    assert ei.value.code == 0
    out = capsys.readouterr().out
    for sub in ("predict", "score", "validate", "canary", "manifest", "doctor"):
        assert sub in out


def test_doctor_advisory_exits_zero(capsys):
    # advisory mode (no --strict): never fails on missing optional deps
    assert cli.main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "mineru_rocm" in out  # always reports the package itself


def test_doctor_json_shape(capsys):
    assert cli.main(["doctor", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert isinstance(data, list) and any(d["label"] == "mineru_rocm" for d in data)


def test_manifest_verify_ok(tmp_path, capsys):
    # write a valid manifest via the runner, then verify it
    runner.write_run_manifest(
        tmp_path, backend="pipeline", model="m",
        run_counts={"attempted": 1, "succeeded": 1, "failed": 0, "skipped": 0},
        final_state={"expected": 1, "complete": 1, "failed": 0, "pending": 0},
        command=["mineru-rocm", "predict"],
    )
    assert cli.main(["manifest", "verify", "--pred-dir", str(tmp_path)]) == 0
    assert "[OK]" in capsys.readouterr().out


def test_manifest_verify_invalid_reports_violations(tmp_path, capsys):
    (tmp_path / "run_manifest.json").write_text(json.dumps({"schema_version": 99}), encoding="utf-8")
    assert cli.main(["manifest", "verify", "--pred-dir", str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "schema_version" in err  # friendly violation report, no traceback


def test_score_invalid_preddir_is_friendly(tmp_path, capsys):
    # a pred-dir with no .md files fails pre-score VALIDATION (the CPU-only path before the
    # scorer subprocess) → ScoringError → friendly message, exit 1, no traceback.
    gt = tmp_path / "gt.json"
    gt.write_text(json.dumps([{"page_info": {"image_path": "a.png"}}]), encoding="utf-8")
    rc = cli.main(["score", "--gt-json", str(gt), "--pred-dir", str(tmp_path / "nope")])
    assert rc == 1
    assert "Traceback" not in capsys.readouterr().err


def test_canary_materialize_missing_inputs_friendly(tmp_path, capsys):
    rc = cli.main(["canary", "materialize", "--full-gt", str(tmp_path / "nope.json"),
                   "--manifest", str(tmp_path / "nope-m.json"), "--out", str(tmp_path / "o.json")])
    assert rc == 1
    assert "Traceback" not in capsys.readouterr().err


def test_validate_clean(tmp_path, capsys):
    gt = tmp_path / "gt.json"
    gt.write_text(json.dumps([{"page_info": {"image_path": "a.png"}}]), encoding="utf-8")
    pred = tmp_path / "pred"; pred.mkdir()
    (pred / "a.md").write_text("# a")
    assert cli.main(["validate", "--gt-json", str(gt), "--pred-dir", str(pred)]) == 0
    assert "[OK]" in capsys.readouterr().out


def test_predict_reaches_driver_arg_check():
    # no extra args → driver.parse_args raises SystemExit (missing required --gt-json/--pred-dir) before any GPU work
    with pytest.raises(SystemExit):
        cli.main(["predict", "--backend", "pipeline"])


def test_predict_with_separator_reaches_driver(tmp_path, monkeypatch):
    """predict forwards driver flags after a literal '--' (argparse REMAINDER limitation).

    The natural form (no '--') is rejected by argparse; the '--' form reaches driver.run.
    (Direct forwarding without '--' is tracked as a P3 improvement.)"""
    gt = tmp_path / "gt.json"
    gt.write_text(json.dumps([{"page_info": {"image_path": "a.png"}}]), encoding="utf-8")
    img = tmp_path / "images"; img.mkdir(); (img / "a.png").write_bytes(b"x")
    pred = tmp_path / "pred"
    seen = {}
    from mineru_rocm import driver
    def _fake_run(dargs, command=None):
        seen["called"] = dargs.backend
        return 0
    monkeypatch.setattr(driver, "run", _fake_run)
    rc = cli.main(["predict", "--backend", "pipeline", "--",
                   "--gt-json", str(gt), "--images-dir", str(img), "--pred-dir", str(pred)])
    assert rc == 0
    assert seen.get("called") == "pipeline"  # driver.run WAS reached with the right backend
