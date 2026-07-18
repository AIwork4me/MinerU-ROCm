# tests/test_dispatcher.py
from pathlib import Path
import json, run_adapter, pipeline_adapter, vlm_adapter

def _write_img(d, name="p1.jpg"):
    p = Path(d) / name; p.write_bytes(b"\xff\xd8\xff"); return p

def test_smoke_backend_writes_md_and_stats(tmp_path):
    _write_img(tmp_path)
    out = tmp_path / "out"
    run_adapter.run_adapter(tmp_path, out, platform="linux-rocm", config={"backend": "smoke"})
    assert (out / "p1.md").read_text(encoding="utf-8").startswith("# p1")
    rs = json.loads((out / "_run_stats.json").read_text())
    assert rs["count"] == 1 and rs["ok"] == 1 and rs["fail"] == 0 and rs["engine"] == "smoke"

def test_pipeline_backend_routes_to_pipeline_adapter(tmp_path, monkeypatch):
    _write_img(tmp_path)
    called = {}
    def fake(img, platform, cfg):
        called["img"] = img.name; return "# pipeline md\n"
    monkeypatch.setattr(pipeline_adapter, "infer_page", fake)
    out = tmp_path / "out"
    run_adapter.run_adapter(tmp_path, out, platform="linux-rocm", config={"backend": "pipeline"})
    assert (out / "p1.md").read_text(encoding="utf-8") == "# pipeline md\n"
    assert called["img"] == "p1.jpg"
    assert json.loads((out / "_run_stats.json").read_text())["engine"] == "pipeline"

def test_vlm_backend_routes_to_vlm_adapter(tmp_path, monkeypatch):
    _write_img(tmp_path)
    monkeypatch.setattr(vlm_adapter, "infer_page", lambda i, p, c: "# vlm md\n")
    out = tmp_path / "out"
    run_adapter.run_adapter(tmp_path, out, platform="linux-rocm", config={"backend": "vlm-transformers"})
    assert (out / "p1.md").read_text(encoding="utf-8") == "# vlm md\n"

def test_unknown_backend_raises_value_error(tmp_path):
    _write_img(tmp_path)
    try:
        run_adapter.run_adapter(tmp_path, tmp_path / "o", platform="linux-rocm", config={"backend": "wat"})
        assert False, "expected ValueError"
    except ValueError:
        pass

def test_per_page_failure_is_recorded_not_raised(tmp_path, monkeypatch):
    _write_img(tmp_path, "ok.jpg"); _write_img(tmp_path, "bad.jpg")
    def fake(img, platform, cfg):
        if img.name == "bad.jpg": raise RuntimeError("boom")
        return "# ok\n"
    monkeypatch.setattr(pipeline_adapter, "infer_page", fake)
    out = tmp_path / "out"
    run_adapter.run_adapter(tmp_path, out, platform="linux-rocm", config={"backend": "pipeline"})
    rs = json.loads((out / "_run_stats.json").read_text())
    assert rs["ok"] == 1 and rs["fail"] == 1
    assert (out / "ok.md").exists() and not (out / "bad.md").exists()
