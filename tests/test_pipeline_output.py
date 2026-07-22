"""R4 output-convention tests for the pipeline adapter.

CPU-only — does NOT import mineru (keeps CI cheap and detached from GPU
provisioning). Verifies the pure `normalize_markdown` post-processing helper
that `infer_page` runs on mineru's raw Markdown output.
"""
from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import mineru_rocm.backends.pipeline as pipeline_backend
from mineru_rocm.backends.pipeline import (
    MineruPipelineRunner,
    _content_list_text_fallback,
    _parse_stem,
    normalize_markdown,
)


def test_windows_uses_short_deterministic_internal_stem():
    image = Path("a" * 240 + ".png")
    first = _parse_stem(image, "windows-hip")

    assert first == _parse_stem(image, "windows-hip")
    assert len(first) == 16
    assert first != image.stem


def test_linux_keeps_original_internal_stem():
    image = Path("page.png")
    assert _parse_stem(image, "linux-rocm") == "page"


def test_windows_pipeline_defaults_to_rocm_before_mineru_import(monkeypatch):
    monkeypatch.delenv("MINERU_DEVICE_MODE", raising=False)
    monkeypatch.setattr(
        pipeline_backend,
        "configure_onnxruntime_directml",
        lambda: {"onnxruntime_provider_requested": "directml"},
    )
    fake_torch = types.SimpleNamespace(
        __version__="2.9.1+rocm7.2.1",
        version=types.SimpleNamespace(hip="7.2"),
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            get_device_name=lambda index: "AMD test GPU",
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class FakeSingleton:
        def get_model(self, **kwargs):
            assert os.environ["MINERU_DEVICE_MODE"] == "cuda"

    fake_module = types.SimpleNamespace(ModelSingleton=FakeSingleton)
    monkeypatch.setitem(
        sys.modules, "mineru.backend.pipeline.pipeline_analyze", fake_module
    )

    cfg = {}
    MineruPipelineRunner(platform="windows-hip", cfg=cfg).load()
    assert cfg["pytorch_hip_version"] == "7.2"
    assert cfg["pytorch_gpu_name"] == "AMD test GPU"


def test_pipeline_respects_explicit_device_override(monkeypatch):
    monkeypatch.setenv("MINERU_DEVICE_MODE", "custom-device")
    monkeypatch.setattr(
        pipeline_backend,
        "configure_onnxruntime_directml",
        lambda: {"onnxruntime_provider_requested": "directml"},
    )

    class FakeSingleton:
        def get_model(self, **kwargs):
            assert os.environ["MINERU_DEVICE_MODE"] == "custom-device"

    fake_module = types.SimpleNamespace(ModelSingleton=FakeSingleton)
    monkeypatch.setitem(
        sys.modules, "mineru.backend.pipeline.pipeline_analyze", fake_module
    )

    MineruPipelineRunner(platform="windows-hip", cfg={}).load()


def test_content_list_fallback_uses_only_nonempty_text_blocks():
    text, block_types = _content_list_text_fallback([
        {"type": "header", "text": "  NO. Date  "},
        {"type": "image", "img_path": "images/0.jpg"},
        {"type": "footer", "text": ""},
        "invalid",
    ])

    assert text == "NO. Date"
    assert block_types == ["header"]


def test_extract_recovers_content_list_text_only_for_empty_markdown(
    tmp_path, monkeypatch
):
    cfg = {}
    runner = MineruPipelineRunner(platform="windows-hip", cfg=cfg)
    runner._tmp_out = tmp_path / "mineru-tmp"

    def fake_do_parse(**kwargs):
        stem = kwargs["pdf_file_names"][0]
        out = Path(kwargs["output_dir"]) / stem / "auto"
        out.mkdir(parents=True)
        (out / f"{stem}.md").write_text("", encoding="utf-8")
        (out / f"{stem}_content_list.json").write_text(
            json.dumps([{"type": "header", "text": "NO. Date"}]),
            encoding="utf-8",
        )
        assert kwargs["f_dump_content_list"] is True

    fake_common = types.SimpleNamespace(
        do_parse=fake_do_parse,
        read_fn=lambda path: b"image-bytes",
    )
    monkeypatch.setitem(sys.modules, "mineru.cli.common", fake_common)

    result = runner.extract(Path("blank-page.jpg"))

    assert result == "NO. Date\n"
    assert cfg["pipeline_empty_markdown_recovery_count"] == 1
    assert cfg["pipeline_empty_markdown_recovery_events"] == [{
        "image": "blank-page.jpg",
        "source": "content_list",
        "block_types": ["header"],
    }]


def test_display_formula_wrapped_in_double_dollar():
    # LaTeX display formulas survive normalization (R4: keep LaTeX as-is).
    assert "$$E=mc^2$$" in normalize_markdown("$$E=mc^2$$")


def test_inline_formula_survives():
    assert "$x^2 + y^2 = r^2$" in normalize_markdown("The identity $x^2 + y^2 = r^2$ holds.")


def test_html_table_passes_through():
    # HTML tables are preserved verbatim (R4: keep HTML tables).
    md = normalize_markdown("<table><tr><td>a</td></tr></table>")
    assert "<table>" in md
    assert "</table>" in md
    assert "<td>a</td>" in md


def test_pipe_table_passes_through():
    md = normalize_markdown("| a | b |\n|---|---|\n| 1 | 2 |\n")
    assert "| a | b |" in md


def test_no_div_wrappers_around_images():
    # R4 pitfall: md_tex_filter strips ![](path) but leaves orphan <div> wrappers.
    # normalize_markdown must scrub any <div>...</div> (including class/id attrs).
    src = '<div class="figure"><img src="x.png"/></div>\n'
    out = normalize_markdown(src)
    assert "<div" not in out
    assert "</div>" not in out
    # The figure body itself is preserved — we only strip the wrapper tags.
    assert '<img src="x.png"' in out


def test_strips_div_with_attributes():
    # Mineru sometimes emits <div class="page_clip..." style="..."> wrappers.
    src = '<div class="page_clip" id="p1" style="border:1px">body</div>'
    assert normalize_markdown(src) == "body"


def test_empty_string_idempotent():
    assert normalize_markdown("") == ""
