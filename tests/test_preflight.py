import json
import pytest
from mineru_rocm.preflight import (
    PreflightError, load_gt, pages_with_images, shard,
    check_prediction_inputs, assert_ok,
)


def _write_gt(tmp_path, pages):
    gt = tmp_path / "gt.json"
    gt.write_text(json.dumps(pages), encoding="utf-8")
    return gt


def test_load_gt_valid(tmp_path):
    gt = _write_gt(tmp_path, [{"page_info": {"image_path": "a.png"}}])
    assert load_gt(gt) == [{"page_info": {"image_path": "a.png"}}]


def test_load_gt_errors(tmp_path):
    with pytest.raises(PreflightError):  # missing file
        load_gt(tmp_path / "nope.json")
    bad = tmp_path / "bad.json"; bad.write_text("{}", encoding="utf-8")
    with pytest.raises(PreflightError):  # not a list
        load_gt(bad)
    empty = tmp_path / "empty.json"; empty.write_text("[]", encoding="utf-8")
    with pytest.raises(PreflightError):  # empty list
        load_gt(empty)
    nopage = tmp_path / "np.json"; nopage.write_text("[{}]", encoding="utf-8")
    with pytest.raises(PreflightError):  # page missing page_info
        load_gt(nopage)


def test_pages_with_images_missing_image(tmp_path):
    gt = _write_gt(tmp_path, [{"page_info": {"image_path": "a.png"}}])
    (tmp_path / "images").mkdir()
    with pytest.raises(PreflightError):
        pages_with_images(gt, tmp_path / "images")


def test_pages_with_images_ok(tmp_path):
    gt = _write_gt(tmp_path, [{"page_info": {"image_path": "a.png"}}])
    imgd = tmp_path / "images"; imgd.mkdir()
    (imgd / "a.png").write_text("x")
    assert pages_with_images(gt, imgd) == [("a", str(imgd / "a.png"))]


def test_shard_returns_exactly_n_buckets_some_empty():
    buckets = shard(["a", "b", "c"], n=5)
    assert len(buckets) == 5            # exactly n, even though len < n
    assert sum(len(b) for b in buckets) == 3


def test_shard_negative_raises():
    with pytest.raises(ValueError):
        shard(["a"], n=0)


def test_check_prediction_inputs_clean(tmp_path):
    probs = check_prediction_inputs(
        gt_json="x", images_dir="x", ports="8000,8001", gpu_ids="0,1",
        concurrency=2, max_retries=3, retry_backoff=1.5, max_pixels=1000,
        model="m", pred_dir=str(tmp_path / "pred"),
    )
    assert probs == []


def test_check_prediction_inputs_bad_ranges(tmp_path):
    probs = check_prediction_inputs(
        gt_json="x", images_dir="x", ports="", gpu_ids=None,
        concurrency=0, max_retries=0, retry_backoff=-1, max_pixels=-5,
        model="", pred_dir=str(tmp_path / "pred"),
    )
    fields = {f for f, _ in probs}
    assert {"ports", "concurrency", "max-retries", "max-pixels", "retry-backoff", "model"} <= fields


def test_assert_ok_raises_on_problems():
    assert_ok([])  # no-op
    with pytest.raises(PreflightError):
        assert_ok([("ports", "is empty")])
