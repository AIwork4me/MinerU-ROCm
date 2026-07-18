"""MinerU-ROCm adapter dispatcher — implements the omnidocbench-amd contract.

Routes --backend to the right sub-adapter (pipeline | vlm-*). Keeps the
contract signature and the out_dir/<image_stem>.md + _run_stats.json output
convention. Per-page failures are caught and recorded (R2) — never raised.
"""
from __future__ import annotations
import argparse, importlib, sys, time
from pathlib import Path
from omnidocbench_amd.types import RunSummary, PageStatus

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
PLATFORMS = ("linux-rocm", "windows-hip")
SUB_ADAPTERS = {"pipeline": "pipeline_adapter",
                "vlm-vllm": "vlm_adapter", "vlm-transformers": "vlm_adapter"}


def _load_adapter_config():
    try:
        from . import adapter_config  # package-relative
    except ImportError:
        _here = Path(__file__).resolve().parent
        if str(_here) not in sys.path:
            sys.path.insert(0, str(_here))
        import adapter_config  # type: ignore[import-not-found]
    return adapter_config


def _import_sub(name: str):
    """Import a sibling adapter module whether run as a package or a bare script."""
    pkg = __package__
    if pkg:
        try:
            return importlib.import_module(f".{name}", pkg)
        except ImportError:
            pass
    _here = Path(__file__).resolve().parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))
    return importlib.import_module(name)


def run_adapter(img_dir: Path, out_dir: Path, *, platform: str, config: dict) -> dict:
    assert platform in PLATFORMS, f"unknown platform: {platform}"
    adapter_config = _load_adapter_config()
    cfg = {**adapter_config.as_dict(), **config}
    out_dir.mkdir(parents=True, exist_ok=True)
    imgs = sorted(p for p in Path(img_dir).iterdir() if p.suffix.lower() in IMG_EXT)
    stats: list[PageStatus] = []
    backend = cfg.get("backend", "smoke")
    try:
        sub = None if backend == "smoke" else _import_sub(SUB_ADAPTERS[backend])  # KeyError → ValueError below
    except KeyError:
        raise ValueError(f"unknown backend: {backend!r} (expected smoke|pipeline|vlm-vllm|vlm-transformers)")
    for i in imgs:
        t0 = time.time()
        try:
            if sub is None:
                md = f"# {i.stem}\n\n(smoke output — backend=smoke)\n"
            else:
                md = sub.infer_page(i, platform, cfg)
            (out_dir / f"{i.stem}.md").write_text(md, encoding="utf-8")
            stats.append(PageStatus(i.name, "ok", seconds=time.time() - t0, attempts=1))
        except Exception as e:  # per-page failure → record, continue, never raise
            stats.append(PageStatus(i.name, f"failed: {e}", error=str(e), seconds=time.time() - t0))
    rs = RunSummary(len(imgs), sum(1 for s in stats if s.status == "ok"),
                    sum(1 for s in stats if s.status.startswith("failed")),
                    sum(1 for s in stats if s.status.startswith("fallback")),
                    cfg.get("limit_pages"), stats, engine=backend)
    rs.write(out_dir / "_run_stats.json")
    return rs.to_run_stats()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--img-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--platform", required=True, choices=PLATFORMS)
    p.add_argument("--backend", default="smoke")
    p.add_argument("--server-url", default="")
    p.add_argument("--api-model-name", default="")
    a = p.parse_args()
    if a.backend != "smoke" and a.backend not in SUB_ADAPTERS:
        raise SystemExit(f"unknown backend: {a.backend!r} (expected smoke|pipeline|vlm-vllm|vlm-transformers)")
    run_adapter(Path(a.img_dir), Path(a.out_dir), platform=a.platform,
                config={"backend": a.backend, "server_url": a.server_url, "api_model_name": a.api_model_name})
