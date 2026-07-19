# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
"""MinerU 3.4 pipeline adapter (backend=pipeline).

Wraps upstream mineru[all] in-process on ROCm cuda. Loads the pipeline ONCE
(first page) and reuses it for every page via `ModelSingleton`. Per-page
images are run through `do_parse` (one stem per call) and the resulting
`<stem>.md` is returned. See docs/spike-mineru-api.md for the full API
contract (§3 in-process, §4 Option A — the chosen sketch).

Contract R4 is enforced by `normalize_markdown`, run on every page's output
before it is handed back to the dispatcher.
"""
from __future__ import annotations
import os
import re
from pathlib import Path

# --- Env BEFORE mineru import -----------------------------------------------
# Both import surfaces used below (`pipeline_analyze.ModelSingleton` in load()
# and `cli.common.do_parse` in extract()) converge on the same module-level
# singleton inside mineru; device-mode is read at first model init, so the env
# vars must be set before any mineru import. Module-level setdefault fires at
# adapter import time, which precedes the lazy mineru imports inside the
# methods — exactly the order the spike requires.
os.environ.setdefault("MINERU_DEVICE_MODE", "cuda")  # HIP_VISIBLE_DEVICES scoped by the launcher
os.environ.setdefault("HF_ENDPOINT", "http://134.199.133.77")

_runner = None  # lazy singleton, created on first infer_page call


_DIV_RE = re.compile(r"</?div[^>]*>")


def normalize_markdown(md: str) -> str:
    """Enforce contract R4: keep LaTeX/HTML/pipe tables, drop <div> wrappers.

    mineru already emits LaTeX (`$$...$$` / `$...$`) and HTML `<table>`s, so
    this is a safety pass — the only transform is stripping `<div>` figure
    wrappers that survive md_tex_filter (a known R4 pitfall). Extend here if
    the spike surfaces other normalization needs.
    """
    return _DIV_RE.sub("", md)


def infer_page(img: Path, platform: str, cfg: dict) -> str:
    """Return Markdown for one page image. Raises on per-page failure (caller catches).

    Uses a module-level `_runner` singleton so the first call pays the one-time
    model load on cuda; every subsequent call reuses `ModelSingleton`.
    """
    global _runner
    if _runner is None:
        _runner = MineruPipelineRunner(platform=platform, cfg=cfg)
        _runner.load()
    return normalize_markdown(_runner.extract(img))


class MineruPipelineRunner:
    def __init__(self, platform: str, cfg: dict):
        self.platform = platform
        self.cfg = cfg
        self._lang = cfg.get("lang", "ch")
        self._tmp_out = Path(cfg.get("mineru_out_root", "/tmp/mineru-adapter-out"))

    def load(self):
        """Warm the pipeline sub-models on cuda.

        Uses the in-process API from docs/spike-mineru-api.md §3. `ModelSingleton`
        is module-level inside mineru, so the model init paid here is reused by
        every later `do_parse` in extract() within this process.
        """
        # Lazy import so the env vars set at module top are in place first.
        from mineru.backend.pipeline.pipeline_analyze import ModelSingleton
        # Force model init now — lays layout/OCR/UniMERNet onto cuda:0
        # (ONNX table models stay on CPU per the spike; not a problem).
        ModelSingleton().get_model(
            lang=self._lang, formula_enable=True, table_enable=True)

    def extract(self, img: Path) -> str:
        """Run the warmed pipeline on one image → Markdown string.

        Writes to a per-run tmp dir (`run-<pid>`) so parallel adapter processes
        don't collide, then reads back the `<stem>/auto/<stem>.md` produced by
        `do_parse` (the deliverable — `f_dump_md=True`).
        """
        # Lazy import — see load(). `infer_page`/`load`/`extract` stay
        # importable functions (no top-level do_parse call) so spawn workers
        # in mineru.utils.pdf_image_tools don't re-enter this code (the spike
        # documents BrokenProcessPool if violated).
        from mineru.cli.common import do_parse, read_fn
        stem = img.stem
        out_dir = self._tmp_out / f"run-{os.getpid()}"
        out_dir.mkdir(parents=True, exist_ok=True)
        do_parse(
            output_dir=str(out_dir),
            pdf_file_names=[stem],
            pdf_bytes_list=[read_fn(img)],
            p_lang_list=[self._lang],
            backend="pipeline",
            parse_method="auto",
            formula_enable=True,
            table_enable=True,
            f_draw_layout_bbox=False,
            f_draw_span_bbox=False,
            f_dump_md=True,                    # the deliverable
            f_dump_middle_json=False,
            f_dump_model_output=False,
            f_dump_orig_pdf=False,
            f_dump_content_list=False,
        )
        md_path = out_dir / stem / "auto" / f"{stem}.md"
        return md_path.read_text(encoding="utf-8")
