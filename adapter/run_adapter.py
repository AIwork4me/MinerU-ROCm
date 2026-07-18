"""MinerU-ROCm adapter — implements the omnidocbench-amd contract.

Replace the `smoke` branch with your model's inference. Keep the signature and the
out_dir/<image_stem>.md output convention. Per-page failures must be caught and
recorded (a missing page scores zero) — never raise.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
from omnidocbench_amd.types import RunSummary, PageStatus

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
PLATFORMS = ("linux-rocm", "windows-hip")


def _load_adapter_config():
    """Import ``adapter_config`` whether run as a package module or a script.

    When invoked via the engine (``stage_infer``) this file is run as a
    subprocess (``python adapter/run_adapter.py``) with no parent package, so
    the plain ``from . import adapter_config`` fails. Fall back to importing
    the sibling module off ``sys.path``.
    """
    try:
        from . import adapter_config  # package-relative import
    except ImportError:
        _here = Path(__file__).resolve().parent
        if str(_here) not in sys.path:
            sys.path.insert(0, str(_here))
        import adapter_config  # type: ignore[no-redef]
    return adapter_config


def run_adapter(img_dir: Path, out_dir: Path, *, platform: str, config: dict) -> dict:
    assert platform in PLATFORMS, f"unknown platform: {platform}"
    adapter_config = _load_adapter_config()
    cfg = {**adapter_config.as_dict(), **config}
    out_dir.mkdir(parents=True, exist_ok=True)
    imgs = sorted(p for p in Path(img_dir).iterdir() if p.suffix.lower() in IMG_EXT)
    stats: list[PageStatus] = []
    backend = cfg.get("backend", "smoke")
    for i in imgs:
        try:
            if backend == "smoke":
                md = f"# {i.stem}\n\n(smoke output — wire your model here)\n"
            else:
                md = _infer(i, platform, cfg)  # TODO-replace: your model's inference
            (out_dir / f"{i.stem}.md").write_text(md, encoding="utf-8")
            stats.append(PageStatus(i.name, "ok", seconds=0.0, attempts=1))
        except Exception as e:  # per-page failure → record, continue, never raise
            stats.append(PageStatus(i.name, f"failed: {e}", error=str(e)))
    rs = RunSummary(len(imgs), sum(1 for s in stats if s.status == "ok"),
                    sum(1 for s in stats if s.status.startswith("failed")),
                    sum(1 for s in stats if s.status.startswith("fallback")),
                    cfg.get("limit_pages"), stats, engine=backend)
    rs.write(out_dir / "_run_stats.json")
    return rs.to_run_stats()


def _infer(img: Path, platform: str, config: dict) -> str:
    raise NotImplementedError("Replace _infer with your model's inference (img → markdown).")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--img-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--platform", required=True, choices=PLATFORMS)
    p.add_argument("--backend", default="smoke")
    p.add_argument("--server-url", default="")
    p.add_argument("--api-model-name", default="")
    a = p.parse_args()
    run_adapter(Path(a.img_dir), Path(a.out_dir), platform=a.platform,
                config={"backend": a.backend, "server_url": a.server_url, "api_model_name": a.api_model_name})
