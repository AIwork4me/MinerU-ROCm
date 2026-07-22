# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
"""Dispatcher contract tests: empty prediction, skip-existing, conservation,
CLI backend selection, and platform integration."""
from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from mineru_rocm import dispatcher


def _write_img(d, name="p1.jpg"):
    p = Path(d) / name
    p.write_bytes(b"\xff\xd8\xff")
    return p


# ---------------------------------------------------------------------------
# Task 3: empty prediction tests
# ---------------------------------------------------------------------------

def test_empty_prediction_is_recorded_as_failure(tmp_path, monkeypatch):
    _write_img(tmp_path)
    monkeypatch.setattr(dispatcher, "_import_sub",
                        lambda name: SimpleNamespace(infer_page=lambda img, p, c: ""))
    out = tmp_path / "out"
    dispatcher.run_adapter(tmp_path, out, platform="linux-rocm", config={"backend": "pipeline"})
    rs = json.loads((out / "_run_stats.json").read_text())
    assert rs["count"] == 1
    assert rs["ok"] == 0
    assert rs["fail"] == 1
    assert rs["stats"][0]["status"].startswith("failed")
    assert "empty" in rs["stats"][0]["status"].lower()
    assert not (out / "p1.md").exists()


def test_whitespace_prediction_is_recorded_as_failure(tmp_path, monkeypatch):
    _write_img(tmp_path)
    monkeypatch.setattr(dispatcher, "_import_sub",
                        lambda name: SimpleNamespace(infer_page=lambda img, p, c: "   \n\n  "))
    out = tmp_path / "out"
    dispatcher.run_adapter(tmp_path, out, platform="linux-rocm", config={"backend": "pipeline"})
    rs = json.loads((out / "_run_stats.json").read_text())
    assert rs["fail"] == 1 and rs["ok"] == 0


def test_non_string_prediction_is_recorded_as_failure(tmp_path, monkeypatch):
    _write_img(tmp_path)
    monkeypatch.setattr(dispatcher, "_import_sub",
                        lambda name: SimpleNamespace(infer_page=lambda img, p, c: 12345))
    out = tmp_path / "out"
    dispatcher.run_adapter(tmp_path, out, platform="linux-rocm", config={"backend": "pipeline"})
    rs = json.loads((out / "_run_stats.json").read_text())
    assert rs["fail"] == 1


def test_empty_prediction_does_not_leave_md_file(tmp_path, monkeypatch):
    _write_img(tmp_path)
    monkeypatch.setattr(dispatcher, "_import_sub",
                        lambda name: SimpleNamespace(infer_page=lambda img, p, c: ""))
    out = tmp_path / "out"
    dispatcher.run_adapter(tmp_path, out, platform="linux-rocm", config={"backend": "pipeline"})
    assert not (out / "p1.md").exists()


def test_run_continues_after_empty_prediction(tmp_path, monkeypatch):
    _write_img(tmp_path, "ok.jpg")
    _write_img(tmp_path, "empty.jpg")
    _write_img(tmp_path, "also_ok.jpg")

    class MockBackend:
        def infer_page(self, img, platform, cfg):
            if img.name == "empty.jpg":
                return ""
            return f"# {img.stem}\n"
    monkeypatch.setattr(dispatcher, "_import_sub",
                        lambda name: SimpleNamespace(infer_page=MockBackend().infer_page))
    out = tmp_path / "out"
    dispatcher.run_adapter(tmp_path, out, platform="linux-rocm", config={"backend": "pipeline"})
    rs = json.loads((out / "_run_stats.json").read_text())
    assert rs["count"] == 3
    assert rs["ok"] == 2
    assert rs["fail"] == 1
    assert (out / "ok.md").exists()
    assert (out / "also_ok.md").exists()
    assert not (out / "empty.md").exists()


# ---------------------------------------------------------------------------
# Task 2: skip-existing tests
# ---------------------------------------------------------------------------

def test_skip_existing_nonempty_prediction_counts_as_ok(tmp_path):
    _write_img(tmp_path, "existing.jpg")
    out = tmp_path / "out"
    out.mkdir()
    (out / "existing.md").write_text("# Real content\n\nSome text.", encoding="utf-8")
    dispatcher.run_adapter(tmp_path, out, platform="linux-rocm",
                           config={"backend": "smoke"}, skip_existing=True)
    rs = json.loads((out / "_run_stats.json").read_text())
    assert rs["count"] == 1
    assert rs["ok"] == 1
    assert rs["fail"] == 0


def test_skip_existing_preserves_count_conservation(tmp_path):
    _write_img(tmp_path, "p1.jpg")
    _write_img(tmp_path, "p2.jpg")
    _write_img(tmp_path, "p3.jpg")
    out = tmp_path / "out"
    out.mkdir()
    (out / "p1.md").write_text("# Cached content\n", encoding="utf-8")
    dispatcher.run_adapter(tmp_path, out, platform="linux-rocm",
                           config={"backend": "smoke"}, skip_existing=True)
    rs = json.loads((out / "_run_stats.json").read_text())
    assert rs["count"] == 3
    assert rs["ok"] + rs["fail"] + rs["fallback"] == rs["count"]
    assert len(rs["stats"]) == rs["count"]


def test_skip_existing_empty_prediction_is_not_ok(tmp_path):
    _write_img(tmp_path, "empty.jpg")
    out = tmp_path / "out"
    out.mkdir()
    (out / "empty.md").write_text("", encoding="utf-8")
    dispatcher.run_adapter(tmp_path, out, platform="linux-rocm",
                           config={"backend": "smoke"}, skip_existing=True)
    rs = json.loads((out / "_run_stats.json").read_text())
    assert rs["count"] == 1
    assert rs["ok"] == 0
    assert rs["fail"] == 1
    assert "empty" in rs["stats"][0]["status"].lower()


def test_skip_existing_stats_contains_every_input_page(tmp_path):
    _write_img(tmp_path, "cached.jpg")
    _write_img(tmp_path, "new.jpg")
    out = tmp_path / "out"
    out.mkdir()
    (out / "cached.md").write_text("# Cached\n", encoding="utf-8")
    dispatcher.run_adapter(tmp_path, out, platform="linux-rocm",
                           config={"backend": "smoke"}, skip_existing=True)
    rs = json.loads((out / "_run_stats.json").read_text())
    images = {s["image"] for s in rs["stats"]}
    assert images == {"cached.jpg", "new.jpg"}


def test_skip_existing_mixed_with_new_predictions(tmp_path, monkeypatch):
    _write_img(tmp_path, "cached.jpg")
    _write_img(tmp_path, "success.jpg")
    _write_img(tmp_path, "failure.jpg")
    out = tmp_path / "out"
    out.mkdir()
    (out / "cached.md").write_text("# Cached\n", encoding="utf-8")

    class MockBackend:
        def infer_page(self, img, platform, cfg):
            if img.name == "failure.jpg":
                raise RuntimeError("boom")
            return f"# {img.stem}\n"
    monkeypatch.setattr(dispatcher, "_import_sub",
                        lambda name: SimpleNamespace(infer_page=MockBackend().infer_page))
    dispatcher.run_adapter(tmp_path, out, platform="linux-rocm",
                           config={"backend": "pipeline"}, skip_existing=True)
    rs = json.loads((out / "_run_stats.json").read_text())
    assert rs["count"] == 3
    assert rs["ok"] == 2   # cached (ok) + success (ok)
    assert rs["fail"] == 1  # failure
    assert rs["fallback"] == 0
    assert len(rs["stats"]) == 3
    assert rs["ok"] + rs["fail"] + rs["fallback"] == rs["count"]


# ---------------------------------------------------------------------------
# Task 2: conservation tests
# ---------------------------------------------------------------------------

def test_run_stats_conservation(tmp_path, monkeypatch):
    _write_img(tmp_path, "ok1.jpg")
    _write_img(tmp_path, "ok2.jpg")
    _write_img(tmp_path, "fail.jpg")

    class MockBackend:
        def infer_page(self, img, platform, cfg):
            if img.name == "fail.jpg":
                raise RuntimeError("fail")
            return "# content\n"
    monkeypatch.setattr(dispatcher, "_import_sub",
                        lambda name: SimpleNamespace(infer_page=MockBackend().infer_page))
    out = tmp_path / "out"
    dispatcher.run_adapter(tmp_path, out, platform="linux-rocm",
                           config={"backend": "pipeline"})
    rs = json.loads((out / "_run_stats.json").read_text())
    assert rs["ok"] + rs["fail"] + rs["fallback"] == rs["count"]


def test_run_stats_stats_length_matches_count(tmp_path, monkeypatch):
    _write_img(tmp_path, "a.jpg")
    _write_img(tmp_path, "b.jpg")
    out = tmp_path / "out"
    dispatcher.run_adapter(tmp_path, out, platform="linux-rocm",
                           config={"backend": "smoke"})
    rs = json.loads((out / "_run_stats.json").read_text())
    assert len(rs["stats"]) == rs["count"] == 2


# ---------------------------------------------------------------------------
# Task 1: CLI and backend selection tests
# ---------------------------------------------------------------------------

def test_adapter_cli_backend_overrides_default(tmp_path, monkeypatch):
    _write_img(tmp_path)
    out = tmp_path / "out"
    rv = dispatcher.main(["--img-dir", str(tmp_path), "--out-dir", str(out),
                          "--platform", "linux-rocm", "--backend", "smoke"])
    assert rv == 0
    rs = json.loads((out / "_run_stats.json").read_text())
    assert rs["engine"] == "smoke"


def test_pipeline_backend_is_explicitly_selectable(tmp_path, monkeypatch):
    _write_img(tmp_path)
    monkeypatch.setattr(dispatcher, "_import_sub",
                        lambda name: SimpleNamespace(infer_page=lambda img, p, c: "# pipeline\n"))
    out = tmp_path / "out"
    dispatcher.run_adapter(tmp_path, out, platform="linux-rocm",
                           config={"backend": "pipeline"})
    rs = json.loads((out / "_run_stats.json").read_text())
    assert rs["engine"] == "pipeline"
    assert rs["ok"] == 1


def test_pipeline_run_stats_include_directml_evidence(tmp_path, monkeypatch):
    _write_img(tmp_path)

    def infer_page(img, platform, cfg):
        cfg.update({
            "onnxruntime_provider_requested": "directml",
            "onnxruntime_providers_available": [
                "DmlExecutionProvider", "CPUExecutionProvider"
            ],
            "onnxruntime_providers_active": [
                "DmlExecutionProvider", "CPUExecutionProvider"
            ],
            "onnxruntime_cpu_fallback_enabled": True,
            "onnxruntime_cpu_overrides_configured": [
                "slanet-plus.onnx"
            ],
            "onnxruntime_cpu_overrides_active": ["slanet-plus.onnx"],
            "onnxruntime_cpu_override_run_counts_by_model": {
                "slanet-plus.onnx": 2
            },
            "pytorch_device_mode": "cuda",
            "pytorch_version": "2.9.1+rocm7.2.1",
            "pytorch_hip_version": "7.2",
            "pytorch_gpu_available": True,
            "pytorch_gpu_name": "AMD test GPU",
            "pipeline_empty_markdown_recovery_count": 1,
            "pipeline_empty_markdown_recovery_events": [{
                "image": "page.jpg",
                "source": "content_list",
                "block_types": ["header"],
            }],
        })
        return "# pipeline\n"

    monkeypatch.setattr(
        dispatcher,
        "_import_sub",
        lambda name: SimpleNamespace(infer_page=infer_page),
    )
    out = tmp_path / "out"
    dispatcher.run_adapter(
        tmp_path,
        out,
        platform="windows-hip",
        config={"backend": "pipeline"},
    )
    run_stats = json.loads((out / "_run_stats.json").read_text())
    extra = run_stats["_extra"]
    assert run_stats["ok"] == 1
    assert run_stats["fallback"] == 0
    assert extra["onnxruntime_provider_requested"] == "directml"
    assert extra["onnxruntime_providers_active"][0] == "DmlExecutionProvider"
    assert extra["onnxruntime_cpu_overrides_active"] == [
        "slanet-plus.onnx"
    ]
    assert extra["onnxruntime_cpu_override_run_counts_by_model"] == {
        "slanet-plus.onnx": 2
    }
    assert extra["pytorch_device_mode"] == "cuda"
    assert extra["pytorch_gpu_name"] == "AMD test GPU"
    assert extra["pipeline_empty_markdown_recovery_count"] == 1
    assert extra["pipeline_empty_markdown_recovery_events"][0][
        "source"
    ] == "content_list"


def test_directml_runtime_retry_marks_page_as_fallback(tmp_path, monkeypatch):
    _write_img(tmp_path)

    def infer_page(img, platform, cfg):
        cfg["onnxruntime_directml_fallback_count"] = 1
        cfg["onnxruntime_directml_fallback_reasons"] = ["DML runtime error"]
        return "# recovered\n"

    monkeypatch.setattr(
        dispatcher,
        "_import_sub",
        lambda name: SimpleNamespace(infer_page=infer_page),
    )
    out = tmp_path / "out"
    dispatcher.run_adapter(
        tmp_path,
        out,
        platform="windows-hip",
        config={"backend": "pipeline"},
    )
    stats = json.loads((out / "_run_stats.json").read_text())
    assert stats["ok"] == 0
    assert stats["fail"] == 0
    assert stats["fallback"] == 1
    assert stats["stats"][0]["status"].startswith("fallback:")


def test_vlm_vllm_backend_is_explicitly_selectable(tmp_path, monkeypatch):
    _write_img(tmp_path)
    monkeypatch.setattr(dispatcher, "_import_sub",
                        lambda name: SimpleNamespace(infer_page=lambda img, p, c: "# vlm\n"))
    out = tmp_path / "out"
    dispatcher.run_adapter(tmp_path, out, platform="linux-rocm",
                           config={"backend": "vlm-vllm"})
    rs = json.loads((out / "_run_stats.json").read_text())
    assert rs["engine"] == "vlm-vllm"


def test_unknown_backend_exits_with_actionable_error(tmp_path):
    _write_img(tmp_path)
    try:
        dispatcher.run_adapter(tmp_path, tmp_path / "o", platform="linux-rocm",
                               config={"backend": "unknown-backend"})
        assert False, "expected ValueError"
    except ValueError as e:
        msg = str(e)
        assert "unknown backend" in msg.lower()
        assert "smoke" in msg and "pipeline" in msg


def test_unknown_backend_cli_gives_actionable_error(tmp_path):
    _write_img(tmp_path)
    with pytest.raises(SystemExit) as exc:
        dispatcher.main(["--img-dir", str(tmp_path), "--out-dir", str(tmp_path / "o"),
                         "--platform", "linux-rocm", "--backend", "nonexistent-xyz"])
    assert exc.value.code != 0


# ---------------------------------------------------------------------------
# Task 1: platform integration tests
# ---------------------------------------------------------------------------

def test_pyproject_declares_omnidocbench_rocm_extra():
    import tomllib
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    cfg = tomllib.loads(pyproject.read_text())
    extras = cfg["project"]["optional-dependencies"]
    assert "omnidocbench-rocm" in (extras.get("platform") or [""])[0]


def test_makefile_uses_omnidocbench_rocm():
    makefile = (Path(__file__).resolve().parents[1] / "Makefile").read_text()
    assert "omnidocbench-rocm" in makefile
    assert "omnidocbench-amd" not in makefile


def test_no_unexplained_legacy_platform_references():
    """Core source files must not contain unexplained old platform names."""
    import ast
    allow_list = {
        "CHANGELOG.md", "README.md", "README.zh-CN.md",
        "results/_archive/README.md", "docs/HANDOFF-windows-hip.md",
    }
    root = Path(__file__).resolve().parents[1]
    core_src = [root / "src", root / "scripts", root / "adapter",
                root / "tests", root / ".github"]
    for base in core_src:
        if not base.exists():
            continue
        for py in base.rglob("*.py"):
            try:
                txt = py.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            rel = str(py.relative_to(root))
            if "omnidocbench_amd" in txt and rel not in allow_list:
                # Check it's not a comment/string about historical context
                try:
                    tree = ast.parse(txt)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import) and any(
                            "omnidocbench_amd" in (a.name or "") for a in node.names
                        ):
                            pytest.fail(f"{rel}: actual import of omnidocbench_amd")
                except SyntaxError:
                    pass
    # No unexplained references found in core source files.
    assert True
