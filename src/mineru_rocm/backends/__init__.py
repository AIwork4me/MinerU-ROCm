"""Inference backends for mineru_rocm.

- pipeline: in-process MinerU 3.4 pipeline on ROCm cuda.
- vlm: MinerU2.5-Pro VLM via vLLM-on-ROCm (http-client) or transformers.

Each backend exposes infer_page(img: Path, platform: str, cfg: dict) -> str.
Heavy deps (mineru, mineru_vl_utils, transformers, PIL) are imported lazily
inside methods so the package imports with no GPU deps installed.
"""
