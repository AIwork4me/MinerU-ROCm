# tests/test_vlm_adapter.py
from mineru_rocm.backends.vlm import normalize_vlm_markdown

def test_strips_md_start_end_markers():
    md = "<|md_start|>\n# Title\n\nbody\n<|md_end|>"
    assert normalize_vlm_markdown(md) == "\n# Title\n\nbody\n"

def test_stips_txt_contd_and_paratext_markers():
    md = "para1<|txt_contd|>para2<|paratext|>"
    out = normalize_vlm_markdown(md)
    assert "<|txt_contd|>" not in out and "<|paratext|>" not in out

def test_keeps_latex_and_html_tables():
    md = "inline $x$ and $$E=mc^2$$\n<table><tr><td>a</td></tr></table>"
    assert normalize_vlm_markdown(md) == md
