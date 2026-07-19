import hashlib
import json
import pytest
from mineru_rocm.canary import CanaryError, materialize


def _full_gt(tmp_path):
    pages = [
        {"page_info": {"image_path": "alpha.png"}, "data": 1},
        {"page_info": {"image_path": "beta.png"}, "data": 2},
        {"page_info": {"image_path": "gamma.png"}, "data": 3},
    ]
    gt = tmp_path / "full.json"
    gt.write_text(json.dumps(pages), encoding="utf-8")
    return gt, pages


def _manifest(pages_order, full_pages, expected_count=None, sha=None):
    # Compute the expected SHA exactly as materialize() will: subset in MANIFEST
    # order (not full-GT order). Skip pages absent from full_pages so the helper
    # doesn't KeyError on the missing-page test (materialize raises before the
    # SHA check there anyway).
    by_img = {p["page_info"]["image_path"]: p for p in full_pages}
    subset = [by_img[ip] for ip in pages_order if ip in by_img]
    blob = json.dumps(subset, ensure_ascii=False).encode("utf-8")
    return {
        "expected_count": expected_count if expected_count is not None else len(pages_order),
        "pages": [{"image_path": ip, "stem": ip.rsplit(".", 1)[0]} for ip in pages_order],
        "source_json_sha256": sha if sha is not None else hashlib.sha256(blob).hexdigest(),
    }


def test_materialize_round_trips_sha(tmp_path):
    gt, pages = _full_gt(tmp_path)
    manifest = _manifest(["gamma.png", "alpha.png"], pages)  # reordered subset
    mf = tmp_path / "manifest.json"; mf.write_text(json.dumps(manifest), encoding="utf-8")
    out = tmp_path / "out" / "canary.json"
    digest = materialize(gt, mf, out)
    assert out.exists()
    assert hashlib.sha256(out.read_bytes()).hexdigest() == digest
    # subset is in manifest order, compact serialization
    sub = json.loads(out.read_text())
    assert [p["page_info"]["image_path"] for p in sub] == ["gamma.png", "alpha.png"]


def test_materialize_sha_mismatch(tmp_path):
    gt, pages = _full_gt(tmp_path)
    manifest = _manifest(["alpha.png"], pages, sha="0" * 64)  # wrong sha
    mf = tmp_path / "manifest.json"; mf.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(CanaryError):
        materialize(gt, mf, tmp_path / "out.json")


def test_materialize_missing_page(tmp_path):
    gt, pages = _full_gt(tmp_path)
    manifest = _manifest(["alpha.png", "nope.png"], pages)
    mf = tmp_path / "manifest.json"; mf.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(CanaryError):
        materialize(gt, mf, tmp_path / "out.json")


def test_materialize_duplicate_paths(tmp_path):
    gt, pages = _full_gt(tmp_path)
    manifest = _manifest(["alpha.png", "alpha.png"], pages, expected_count=2)
    mf = tmp_path / "manifest.json"; mf.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(CanaryError):
        materialize(gt, mf, tmp_path / "out.json")
