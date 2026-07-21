# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
"""MinerU-ROCm adapter dispatcher — implements the omnidocbench-rocm contract.
 
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
    count = len(imgs)
    stats: list[PageStatus] = []
    resumed_existing = 0
    backend = cfg.get("backend", "smoke")
    try:
        sub = None if backend == "smoke" else _import_sub(SUB_ADAPTERS[backend])
    except KeyError:
        raise ValueError(f"unknown backend: {backend!r} (expected smoke|pipeline|vlm-vllm|vlm-transformers)")

    for img in imgs:
        out_md = out_dir / f"{img.stem}.md"
        t0 = time.time()

        if skip_existing and out_md.exists():
            try:
                existing = out_md.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as e:
                stats.append(PageStatus(img.name, f"failed: existing file unreadable: {e}",
                                        error=str(e), seconds=0.0))
                continue
            if not existing.strip():
                stats.append(PageStatus(img.name, "failed: existing prediction is empty",
                                        error="existing prediction is empty", seconds=0.0))
                continue
            stats.append(PageStatus(img.name, "ok",
                                    seconds=0.0, attempts=0))
            resumed_existing += 1
            continue

        try:
            if sub is None:
                md = f"# {img.stem}\n\n(smoke output — backend=smoke)\n"
            else:
                md = sub.infer_page(img, platform, cfg)
            if not isinstance(md, str):
                raise TypeError(f"prediction is not a string (got {type(md).__name__})")
            if not md.strip():
                raise RuntimeError("empty prediction")
            out_md.write_text(md, encoding="utf-8")
            stats.append(PageStatus(img.name, "ok", seconds=time.time() - t0, attempts=1))
        except Exception as e:
            stats.append(PageStatus(img.name, f"failed: {e}", error=str(e), seconds=time.time() - t0))
            if out_md.exists():
                try:
                    out_md.unlink()
                except OSError:
                    pass

    ok = sum(1 for s in stats if s.status == "ok")
    fail = sum(1 for s in stats if s.status.startswith("failed"))
    fallback = sum(1 for s in stats if s.status.startswith("fallback"))

    if ok + fail + fallback != count:
        raise RuntimeError(
            f"stats conservation violation: ok={ok} fail={fail} fallback={fallback} "
            f"!= count={count} len(stats)={len(stats)}"
        )
    if len(stats) != count:
        raise RuntimeError(
            f"stats length mismatch: len(stats)={len(stats)} != count={count}"
        )

    extra = {}
    if resumed_existing > 0:
        extra["resumed_existing"] = resumed_existing

    rs = RunSummary(count, ok, fail, fallback,
                    cfg.get("limit_pages"), stats, engine=backend)
    d = rs.to_run_stats()
    if extra:
        d["_extra"] = extra
    (out_dir / "_run_stats.json").parent.mkdir(parents=True, exist_ok=True)
    import json
    (out_dir / "_run_stats.json").write_text(
        json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    return d


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--img-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--platform", required=True, choices=PLATFORMS)
    p.add_argument("--backend", default=None)
    p.add_argument("--server-url", default=None)
    p.add_argument("--api-model-name", default=None)
    p.add_argument("--skip-existing", action="store_true",
                   help="resume: skip images whose .md already exists in out-dir (non-empty only)")
    a = p.parse_args(argv)
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
