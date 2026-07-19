"""R4 output-convention tests for the pipeline adapter.

CPU-only — does NOT import mineru (keeps CI cheap and detached from GPU
provisioning). Verifies the pure `normalize_markdown` post-processing helper
that `infer_page` runs on mineru's raw Markdown output.
"""
from __future__ import annotations

from mineru_rocm.backends.pipeline import normalize_markdown


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
