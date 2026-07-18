"""VLM adapter (MinerU2.5-Pro-2605-1.2B). Drives mineru-vl-utils two-step inference
(layout -> per-block extract) against a vLLM-on-ROCm (or transformers) server.

The dispatcher calls infer_page(img, platform, cfg) per page; we lazily create one
MinerUClient (http-client) pointed at the server and reuse it. Per-page failures
propagate to the dispatcher's try/except (R2).
"""
from __future__ import annotations
import re
from pathlib import Path

# Safety markers mineru-vl-utils may emit around/inside the markdown (spike §2):
# <|md_start|>/<|md_end|> wrap the doc; <|txt_contd|>/<|paratext|> are cross-page tokens.
_MARKER_RE = re.compile(r"<\|(?:md_start|md_end|txt_contd|paratext)\|>")


def normalize_vlm_markdown(md: str) -> str:
    """Strip mineru control markers; keep LaTeX ($...$/$$...$$) and HTML tables intact (R4)."""
    return _MARKER_RE.sub("", md)


_runner = None  # lazy singleton


def infer_page(img: Path, platform: str, cfg: dict) -> str:
    """Return Markdown for one page image via the VLM two-step path."""
    global _runner
    if _runner is None:
        _runner = MineruVLRunner(platform=platform, cfg=cfg)
        _runner.load()
    return normalize_vlm_markdown(_runner.extract(img))


class MineruVLRunner:
    """Wraps mineru_vl_utils.MinerUClient(http-client) against a persistent VLM server."""

    def __init__(self, platform: str, cfg: dict):
        self.platform = platform
        self.cfg = cfg
        self.server_url = cfg.get("server_url") or "http://127.0.0.1:8265/v1"
        self.model_name = cfg.get("api_model_name") or "mineru-pro"

    def load(self):
        from mineru_vl_utils import MinerUClient  # confirmed in Task 2 Step 1
        self._client = MinerUClient(
            backend="http-client",
            server_url=self.server_url,
            model_name=self.model_name,
        )

    def extract(self, img: Path) -> str:
        from PIL import Image
        # json2md lives in the post_process submodule, NOT at mineru_vl_utils top-level
        # (verified Task 2 Step 1: top-level __all__ = MinerUClient, MinerUSamplingParams,
        #  MinerULogitsProcessor, __version__; the spike harness test_twostep.py uses
        #  `from mineru_vl_utils.post_process.json2markdown import json2md`).
        from mineru_vl_utils.post_process.json2markdown import json2md
        pil = Image.open(img).convert("RGB")
        result = self._client.two_step_extract(pil)
        return json2md(result)
