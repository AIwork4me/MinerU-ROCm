"""MinerU-ROCm adapter dispatcher — implements the omnidocbench-amd contract.

Routes --backend to the right sub-adapter (pipeline | vlm-*). Keeps the
contract signature and the out_dir/<image_stem>.md + _run_stats.json output
convention. Per-page failures are caught and recorded (R2) — never raised.
"""
from __future__ import annotations
import argparse, importlib, time
from pathlib import Path
from mineru_rocm.types import RunSummary, PageStatus
from mineru_rocm import config as adapter_config

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
PLATFORMS = ("linux-rocm", "windows-hip")
SUB_ADAPTERS = {"pipeline": "pipeline",
                "vlm-vllm": "vlm", "vlm-transformers": "vlm"}


def _import_sub(name: str):
    """Import a backend module from mineru_rocm.backends (package-relative)."""
    return importlib.import_module(f"mineru_rocm.backends.{name}")


def run_adapter(img_dir: Path, out_dir: Path, *, platform: str, config: dict,
                skip_existing: bool = False) -> dict:
    assert platform in PLATFORMS, f"unknown platform: {platform}"
    cfg = {**adapter_config.as_dict(), **config}
    out_dir.mkdir(parents=True, exist_ok=True)
    imgs = sorted(p for p in Path(img_dir).iterdir() if p.suffix.lower() in IMG_EXT)
    stats: list[PageStatus] = []
    backend = cfg.get("backend", "smoke")
    try:
        sub = None if backend == "smoke" else _import_sub(SUB_ADAPTERS[backend])  # KeyError → ValueError below
    except KeyError:
        raise ValueError(f"unknown backend: {backend!r} (expected smoke|pipeline|vlm-vllm|vlm-transformers)")
    # Resume support: --skip-existing skips any image whose .md already exists in
    # out_dir (multi-hour VLM runs can be interrupted; this lets a re-invocation
    # pick up where it left off without re-inferring finished pages). Skipped pages
    # are NOT counted in ok/fail (they're already on disk; the scorer reads .md
    # files directly, not the run_stats counts).
    skipped = 0
    for i in imgs:
        out_md = out_dir / f"{i.stem}.md"
        if skip_existing and out_md.exists():
            skipped += 1
            continue
        t0 = time.time()
        try:
            if sub is None:
                md = f"# {i.stem}\n\n(smoke output — backend=smoke)\n"
            else:
                md = sub.infer_page(i, platform, cfg)
            out_md.write_text(md, encoding="utf-8")
            stats.append(PageStatus(i.name, "ok", seconds=time.time() - t0, attempts=1))
        except Exception as e:  # per-page failure → record, continue, never raise
            stats.append(PageStatus(i.name, f"failed: {e}", error=str(e), seconds=time.time() - t0))
    rs = RunSummary(len(imgs), sum(1 for s in stats if s.status == "ok"),
                    sum(1 for s in stats if s.status.startswith("failed")),
                    sum(1 for s in stats if s.status.startswith("fallback")),
                    cfg.get("limit_pages"), stats, engine=backend)
    rs.write(out_dir / "_run_stats.json")
    return rs.to_run_stats()


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--img-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--platform", required=True, choices=PLATFORMS)
    # Default comes from adapter_config.BACKEND (so the repo's configured primary
    # backend — `pipeline` for MinerU-ROCm — is used when the engine invokes the
    # adapter without --backend). Pass an explicit --backend to override.
    p.add_argument("--backend", default=None)
    p.add_argument("--server-url", default=None)
    p.add_argument("--api-model-name", default=None)
    p.add_argument("--skip-existing", action="store_true",
                   help="resume: skip images whose .md already exists in out-dir")
    a = p.parse_args(argv)
    # Resolve defaults from adapter_config, then let explicit CLI flags win.
    cfg = adapter_config.as_dict()
    if a.backend is not None:
        if a.backend != "smoke" and a.backend not in SUB_ADAPTERS:
            raise SystemExit(f"unknown backend: {a.backend!r} (expected smoke|pipeline|vlm-vllm|vlm-transformers)")
        cfg["backend"] = a.backend
    if a.server_url is not None:
        cfg["server_url"] = a.server_url
    if a.api_model_name is not None:
        cfg["api_model_name"] = a.api_model_name
    run_adapter(Path(a.img_dir), Path(a.out_dir), platform=a.platform, config=cfg,
                skip_existing=a.skip_existing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
