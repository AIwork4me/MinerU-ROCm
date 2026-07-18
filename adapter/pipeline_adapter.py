"""MinerU 3.4 pipeline adapter (backend=pipeline).

Wraps upstream mineru[all] in-process on ROCm cuda. Loads the pipeline ONCE
(first page) and reuses it for every page. The actual mineru call lives in
MineruPipelineRunner._extract_markdown (filled by Task 5, after the Task 4
spike documents the exact API in docs/spike-mineru-api.md).
"""
from __future__ import annotations
from pathlib import Path

_runner = None  # lazy singleton, created on first infer_page call

def infer_page(img: Path, platform: str, cfg: dict) -> str:
    """Return Markdown for one page image. Raises on per-page failure (caller catches)."""
    global _runner
    if _runner is None:
        _runner = MineruPipelineRunner(platform=platform, cfg=cfg)
        _runner.load()
    return _runner.extract(img)

class MineruPipelineRunner:
    def __init__(self, platform: str, cfg: dict):
        self.platform = platform
        self.cfg = cfg

    def load(self):
        """Warm the pipeline sub-models on cuda. Filled in Task 5."""
        raise NotImplementedError("Task 5 fills load() using the Task 4 spike findings.")

    def extract(self, img: Path) -> str:
        """Run the pipeline on one image → Markdown. Filled in Task 5."""
        raise NotImplementedError("Task 5 fills extract() using the Task 4 spike findings.")
