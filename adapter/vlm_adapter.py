"""VLM adapter (MinerU2.5-Pro-2605-1.2B). Filled in Plan 2 (Phase 2+3).

This plan ships only the stub so the dispatcher routes vlm-* backends cleanly.
"""
from __future__ import annotations
from pathlib import Path

_NOT_IMPLEMENTED = (
    "The VLM adapter (MinerU2.5-Pro-2605-1.2B) is implemented in Plan 2 (Phase 2+3). "
    "It drives opendatalab/mineru-vl-utils two-step inference (layout→extract) with the "
    "MinerULogitsProcessor against a vLLM-on-ROCm or transformers server. "
    "See docs/superpowers/specs/2026-07-17-mineru-rocm-design.md §8."
)

def infer_page(img: Path, platform: str, cfg: dict) -> str:
    raise NotImplementedError(_NOT_IMPLEMENTED)
