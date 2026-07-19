import json
import subprocess
from pathlib import Path
import pytest
from mineru_rocm import scoring
from mineru_rocm.scoring import (
    ScoringError, overall_score, write_eval_config, parse_run_summary,
    score_directory, format_score_table,
)


def test_overall_score_formula():
    # ((1-text)*100 + cdm*100 + teds*100)/3
    assert overall_score({"text_edit_dist": 0.05, "formula_cdm": 0.95, "table_teds": 0.90}) == \
        pytest.approx(((1 - 0.05) * 100 + 0.95 * 100 + 0.90 * 100) / 3)


def test_overall_score_none_when_metric_missing():
    assert overall_score({"text_edit_dist": 0.05, "formula_cdm": None, "table_teds": 0.90}) is None


def test_write_eval_config_substitutes_paths_and_keeps_metrics(tmp_path):
    out = tmp_path / "cfg" / "_eval_config.yaml"
    write_eval_config(gt_json="/gt/full.json", pred_dir="/pred/vlm", out_yaml=out)
    assert out.is_file()
    import yaml
    cfg = yaml.safe_load(out.read_text())
    assert cfg["end2end_eval"]["dataset"]["ground_truth"]["data_path"] == "/gt/full.json"
    assert cfg["end2end_eval"]["dataset"]["prediction"]["data_path"] == "/pred/vlm"
    # metric structure intact (model-agnostic)
    assert cfg["end2end_eval"]["metrics"]["display_formula"]["metric"] == ["Edit_dist", "CDM"]
    assert cfg["end2end_eval"]["metrics"]["table"]["metric"] == ["TEDS", "Edit_dist"]


def _write_run_summary(result_dir, save_name, *, text=0.0566, cdm=0.9755, teds=0.8204, order=0.1240):
    result_dir = Path(result_dir); result_dir.mkdir(parents=True, exist_ok=True)
    ms = {
        "text_block_Edit_dist": {"raw": text},
        "display_formula_CDM": {"raw": cdm},
        "table_TEDS": {"raw": teds},
        "reading_order_Edit_dist": {"raw": order},
    }
    blob = {"notebook_metric_summary": {"metrics": ms}}
    (result_dir / f"{save_name}_run_summary.json").write_text(json.dumps(blob), encoding="utf-8")


def test_parse_run_summary(tmp_path):
    _write_run_summary(tmp_path / "result", "mypred_quick_match")
    m = parse_run_summary(tmp_path / "result", "mypred_quick_match")
    assert m["text_edit_dist"] == 0.0566 and m["formula_cdm"] == 0.9755
    assert m["table_teds"] == 0.8204 and m["reading_order_edit"] == 0.1240
    assert m["overall"] == pytest.approx(((1 - 0.0566) * 100 + 0.9755 * 100 + 0.8204 * 100) / 3)


def test_score_directory_success(monkeypatch, tmp_path):
    # hermetic: don't call the real scorer; fake a clean result
    pred = tmp_path / "mypred"; pred.mkdir()
    (pred / "a.md").write_text("# a")
    gt = tmp_path / "gt.json"; gt.write_text(json.dumps([{"page_info": {"image_path": "a.png"}}]), encoding="utf-8")

    def fake_run(*, omnidocbench_repo, config_yaml, venv_python=None):
        save = f"{pred.name}_quick_match"
        _write_run_summary(Path(omnidocbench_repo) / "result", save)
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(scoring, "run_scorer", fake_run)
    out = score_directory(gt_json=str(gt), pred_dir=str(pred), omnidocbench_repo=str(tmp_path / "repo"))
    assert out["validation_report"].ok
    assert out["metrics"]["overall"] is not None


def test_score_directory_validation_failure(tmp_path):
    pred = tmp_path / "mypred"; pred.mkdir()  # empty -> missing prediction
    gt = tmp_path / "gt.json"; gt.write_text(json.dumps([{"page_info": {"image_path": "a.png"}}]), encoding="utf-8")
    with pytest.raises(ScoringError):
        score_directory(gt_json=str(gt), pred_dir=str(pred), omnidocbench_repo=str(tmp_path))


def test_score_directory_scorer_nonzero(monkeypatch, tmp_path):
    pred = tmp_path / "mypred"; pred.mkdir(); (pred / "a.md").write_text("# a")
    gt = tmp_path / "gt.json"; gt.write_text(json.dumps([{"page_info": {"image_path": "a.png"}}]), encoding="utf-8")
    monkeypatch.setattr(scoring, "run_scorer",
                        lambda **kw: subprocess.CompletedProcess(args=[], returncode=2, stdout="", stderr="boom"))
    with pytest.raises(ScoringError):
        score_directory(gt_json=str(gt), pred_dir=str(pred), omnidocbench_repo=str(tmp_path))


def test_format_score_table_renders_overall():
    m = {"overall": 95.56, "text_edit_dist": 0.0566, "formula_cdm": 0.9755,
         "table_teds": 0.8204, "reading_order_edit": 0.1240}
    s = format_score_table("vlm-vllm", m)
    assert "95.56" in s and "OmniDocBench v1.6" in s
