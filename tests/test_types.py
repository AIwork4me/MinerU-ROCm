import json
from pathlib import Path
import pytest
from mineru_rocm.types import RunSummary, PageStatus


def test_page_status_positional_and_kwargs():
    # ok-path call shape (from the dispatcher)
    a = PageStatus("p1.jpg", "ok", seconds=1.25, attempts=1)
    assert a.image == "p1.jpg" and a.status == "ok" and a.seconds == 1.25 and a.attempts == 1
    # failed-path call shape (attempts omitted -> default 0)
    b = PageStatus("p2.jpg", "failed: boom", error="boom", seconds=0.5)
    assert b.status.startswith("failed") and b.error == "boom" and b.attempts == 0


def test_run_summary_positional_engine_kwarg():
    stats = [PageStatus("p.jpg", "ok")]
    rs = RunSummary(1, 1, 0, 0, None, stats, engine="smoke")
    assert (rs.count, rs.ok, rs.fail, rs.fallback, rs.limit_pages, rs.engine) == (1, 1, 0, 0, None, "smoke")


def test_to_run_stats_emits_eight_keys():
    rs = RunSummary(2, 1, 1, 0, None, [PageStatus("a", "ok"), PageStatus("b", "failed: x")], engine="pipeline")
    d = rs.to_run_stats()
    assert set(d) == {"schema_version", "count", "ok", "fail", "fallback", "limit_pages", "engine", "stats"}
    assert d["schema_version"] == 1
    assert (d["count"], d["ok"], d["fail"]) == (2, 1, 1)
    assert d["engine"] == "pipeline"
    assert d["stats"][0]["image"] == "a" and d["stats"][1]["status"] == "failed: x"


def test_write_round_trips(tmp_path):
    rs = RunSummary(1, 1, 0, 0, None, [PageStatus("p.jpg", "ok", seconds=0.1)], engine="vlm-vllm")
    out = rs.write(tmp_path / "_run_stats.json")
    assert out.exists()
    d = json.loads((tmp_path / "_run_stats.json").read_text())
    assert d["count"] == 1 and d["ok"] == 1 and d["engine"] == "vlm-vllm"


def test_run_summary_stats_is_required_positional():
    # Engine parity: omnidocbench_rocm.types.RunSummary requires stats positionally
    # (verified — engine raises TypeError). Our port must match.
    with pytest.raises(TypeError):
        RunSummary(1, 1, 0, 0, None)
