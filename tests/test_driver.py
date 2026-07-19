import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from mineru_rocm import driver, runner


def _make_gt_and_images(tmp_path, stems):
    """Write a minimal OmniDocBench GT json + an images dir with one PNG per stem."""
    gt = tmp_path / "gt.json"
    gt.write_text(
        json.dumps([{"page_info": {"image_path": f"{s}.png"}} for s in stems]),
        encoding="utf-8",
    )
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for s in stems:
        (img_dir / f"{s}.png").write_bytes(b"\x89PNG fake")  # contents irrelevant (fake infer)
    return gt, img_dir


def _args(tmp_path, gt, img_dir, **over):
    base = dict(
        gt_json=str(gt), images_dir=str(img_dir), pred_dir=str(tmp_path / "pred"),
        backend="pipeline", model="m", platform="linux-rocm", lang="ch",
        max_retries=2, retry_backoff=0.0, overwrite=False, retry_failed=False,
    )
    base.update(over)
    return SimpleNamespace(**base)


def test_orchestration_smoke_full_round_trip(tmp_path):
    """Integration smoke (P1c carry-forward): preflight→select_todo→infer→commit_success→write_manifest→validate_manifest."""
    gt, img_dir = _make_gt_and_images(tmp_path, ["a", "b", "c"])

    def fake_infer(img, platform, cfg):
        return f"# {Path(img).stem}\n\n(fake)\n"

    code = driver._orchestrate(
        _args(tmp_path, gt, img_dir), infer_page=fake_infer,
        backend="pipeline", model="m", cfg={},
    )
    assert code == 0
    pred = tmp_path / "pred"
    # every page written atomically
    assert {p.stem for p in pred.glob("*.md")} == {"a", "b", "c"}
    # manifest present + conservation laws hold
    m = json.loads((pred / "run_manifest.json").read_text("utf-8"))
    assert m["status"] == "ok"
    assert m["run_counts"] == {"attempted": 3, "succeeded": 3, "failed": 0, "skipped": 0, "interrupted": 0}
    assert m["final_state"] == {"expected": 3, "complete": 3, "failed": 0, "pending": 0}
    assert m["backend"] == "pipeline" and m["model"] == "m"
    assert runner.validate_manifest(m) == []  # the load-bearing conservation check


def test_orchestration_resume_skips_complete(tmp_path):
    """A genuinely-complete page is skipped on re-run (select_todo), counted as skipped."""
    gt, img_dir = _make_gt_and_images(tmp_path, ["a", "b"])
    pred = tmp_path / "pred"
    # pre-complete page "a" the way the runner does (atomic + valid content)
    runner.commit_success(pred, "a", "# a already done\n")

    seen = []
    def fake_infer(img, platform, cfg):
        seen.append(Path(img).stem)
        return f"# {Path(img).stem}\n"

    code = driver._orchestrate(
        _args(tmp_path, gt, img_dir), infer_page=fake_infer,
        backend="pipeline", model="m", cfg={},
    )
    assert code == 0
    assert seen == ["b"]  # "a" was skipped, never inferred
    m = json.loads((pred / "run_manifest.json").read_text("utf-8"))
    assert m["run_counts"]["attempted"] == 1 and m["run_counts"]["skipped"] == 1
    assert m["final_state"]["complete"] == 2  # a (pre-done) + b (this run)
    assert runner.validate_manifest(m) == []


def test_orchestration_failure_recorded_manifest_failed(tmp_path):
    """A page whose infer always raises is recorded + makes the run status 'failed' (conservation still holds)."""
    gt, img_dir = _make_gt_and_images(tmp_path, ["ok1", "bad"])

    def fake_infer(img, platform, cfg):
        if Path(img).stem == "bad":
            raise RuntimeError("boom")
        return f"# {Path(img).stem}\n"

    code = driver._orchestrate(
        _args(tmp_path, gt, img_dir, max_retries=2, retry_backoff=0.0),
        infer_page=fake_infer, backend="pipeline", model="m", cfg={},
    )
    assert code == 1  # non-ok status
    pred = tmp_path / "pred"
    assert (pred / "ok1.md").exists()
    assert not (pred / "bad.md").exists()  # never committed
    assert (pred / "_errors" / "bad.json").exists()  # error record written
    m = json.loads((pred / "run_manifest.json").read_text("utf-8"))
    assert m["status"] == "failed"
    assert m["run_counts"]["failed"] == 1 and m["run_counts"]["succeeded"] == 1
    assert m["final_state"]["failed"] == 1 and m["final_state"]["complete"] == 1
    assert runner.validate_manifest(m) == []  # conservation holds even on failure


def test_orchestration_conflict_aborts(tmp_path):
    """Two images mapping to the same stem abort before any write (returns 1, no manifest)."""
    gt = tmp_path / "gt.json"
    gt.write_text(
        json.dumps([
            {"page_info": {"image_path": "dir/page.png"}},
            {"page_info": {"image_path": "other/page.png"}},  # same stem "page"
        ]),
        encoding="utf-8",
    )
    img_dir = tmp_path / "images"
    (img_dir / "dir").mkdir(parents=True)
    (img_dir / "other").mkdir(parents=True)
    (img_dir / "dir" / "page.png").write_bytes(b"x")
    (img_dir / "other" / "page.png").write_bytes(b"x")

    code = driver._orchestrate(
        _args(tmp_path, gt, img_dir), infer_page=lambda *a, **k: "# x\n",
        backend="pipeline", model="m", cfg={},
    )
    assert code == 1
    assert not (tmp_path / "pred" / "run_manifest.json").exists()  # nothing written


def test_parse_args_required_and_defaults():
    a = driver.parse_args(["--gt-json", "g.json", "--images-dir", "i", "--pred-dir", "p", "--backend", "pipeline"])
    assert a.gt_json == "g.json" and a.backend == "pipeline" and a.platform == "linux-rocm"
    assert a.max_retries == 2 and a.retry_backoff == 2.0 and a.lang == "ch"
    assert a.overwrite is False and a.retry_failed is False


def test_parse_args_rejects_unknown_backend():
    import pytest as _pytest
    with _pytest.raises(SystemExit):
        driver.parse_args(["--gt-json", "g", "--images-dir", "i", "--pred-dir", "p", "--backend", "bogus"])


def test_run_routes_to_pipeline_backend(tmp_path, monkeypatch):
    """run(backend=pipeline) calls backends.pipeline.infer_page via _orchestrate (no GPU)."""
    gt, img_dir = _make_gt_and_images(tmp_path, ["a"])
    called = {}
    from mineru_rocm.backends import pipeline as be
    monkeypatch.setattr(be, "infer_page", lambda img, platform, cfg: called.setdefault("hit", str(img)) or f"# {Path(img).stem}\n")
    a = _args(tmp_path, gt, img_dir, backend="pipeline")
    assert driver.run(a) == 0
    assert "hit" in called  # the real backend selector was invoked


def test_run_routes_to_vlm_backend(tmp_path, monkeypatch):
    """run(backend=vlm-vllm) calls backends.vlm.infer_page via _orchestrate (no GPU)."""
    gt, img_dir = _make_gt_and_images(tmp_path, ["a"])
    from mineru_rocm.backends import vlm as be
    monkeypatch.setattr(be, "infer_page", lambda img, platform, cfg: f"# {Path(img).stem}\n")
    a = _args(tmp_path, gt, img_dir, backend="vlm-vllm")
    assert driver.run(a) == 0


def test_module_is_runnable_help():
    """`python -m mineru_rocm.driver --help` exits 0 and shows the flags."""
    import subprocess
    res = subprocess.run(
        ["/opt/venv/bin/python", "-m", "mineru_rocm.driver", "--help"],
        capture_output=True, text=True,
    )
    assert res.returncode == 0
    assert "--backend" in res.stdout and "--pred-dir" in res.stdout
