from pathlib import Path
from mineru_rocm.omnidocbench import derive_prediction_filename, iter_page_images


def test_derive_prediction_filename_strips_dir_and_ext():
    assert derive_prediction_filename("a/b/c/page-001.png") == "page-001.md"
    assert derive_prediction_filename(Path("/x/yo.JPG")) == "yo.md"
    assert derive_prediction_filename("noext") == "noext.md"


def test_iter_page_images_yields_stem_and_abs_path(tmp_path):
    gt = tmp_path / "gt.json"
    gt.write_text(
        '[{"page_info": {"image_path": "page-001.png"}}, '
        '{"page_info": {"image_path": "sub/page-002.jpg"}}]',
        encoding="utf-8",
    )
    images = tmp_path / "images"
    (images / "sub").mkdir(parents=True)
    (images / "page-001.png").write_text("x")
    (images / "sub" / "page-002.jpg").write_text("y")
    out = list(iter_page_images(gt, images))
    assert [stem for stem, _ in out] == ["page-001", "page-002"]
    assert out[0][1] == images / "page-001.png"
    assert out[1][1] == images / "sub" / "page-002.jpg"
