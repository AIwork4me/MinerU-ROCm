# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
"""OmniDocBench v1.6 scoring driver for MinerU-ROCm.

Writes an eval config, invokes the OmniDocBench scorer (pdf_validation.py) in its
own 3.11 venv, and parses the resulting metric_result.json / run_summary.json.
Overall = ((1 - text_EditDist)*100 + formula_CDM*100 + table_TEDS*100) / 3
(reading-order EditDist is reported separately, NOT part of Overall).
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path, PurePosixPath, PureWindowsPath

import yaml

# No host-path defaults: the OmniDocBench scorer venv/repo live at machine-local
# paths that must NOT leak into the public source tree. Set OMNIDOCBENCH_VENV /
# OMNIDOCBENCH_REPO (or pass --venv-python / --omnidocbench-repo) so `mineru-rocm
# score` works from a wheel install anywhere. score_directory() raises a clean
# ScoringError if neither is provided.
DEFAULT_VENV_PYTHON = os.environ.get("OMNIDOCBENCH_VENV")        # e.g. /path/to/OmniDocBench/.venv/bin/python
DEFAULT_OMNIDOCBENCH_REPO = os.environ.get("OMNIDOCBENCH_REPO")  # e.g. /path/to/OmniDocBench


class ScoringError(RuntimeError):
    """Raised when a prediction directory cannot be scored (validation failure or
    scorer non-zero exit). Carries a structured message so the CLI can present a
    friendly error instead of a traceback."""


def _load_eval_template() -> str:
    from importlib.resources import files
    return (files("mineru_rocm") / "data" / "eval_config.yaml").read_text(encoding="utf-8")


def overall_score(metrics: dict) -> float | None:
    """v1.6 Overall = ((1-text_edit)*100 + cdm*100 + teds*100)/3.

    Returns None when any of the three is missing (e.g. CDM is null on a subset
    with no display-formula pages), since the 3-metric Overall is undefined then.
    """
    text = metrics["text_edit_dist"]
    cdm = metrics["formula_cdm"]
    teds = metrics["table_teds"]
    if text is None or cdm is None or teds is None:
        return None
    return ((1.0 - text) * 100.0 + cdm * 100.0 + teds * 100.0) / 3.0


def _eval_config_path(value: str | os.PathLike[str]) -> str:
    """Return an absolute, slash-stable path for the scorer config.

    Preserve already-absolute paths in either POSIX or Windows syntax. This is
    important on Windows, where ``Path('/gt/full.json').resolve()`` would invent
    a drive prefix and silently change a caller-supplied POSIX absolute path.
    Relative paths are resolved on the current host, then serialized with
    forward slashes so generated YAML is stable across platforms.
    """
    raw = os.fspath(value)
    if PurePosixPath(raw).is_absolute():
        return PurePosixPath(raw).as_posix()
    if PureWindowsPath(raw).is_absolute():
        return PureWindowsPath(raw).as_posix()
    return Path(raw).resolve().as_posix()


def write_eval_config(*, gt_json: str, pred_dir: str, out_yaml: Path) -> None:
    """Materialize an eval config from the template, substituting GT + pred paths.

    Relative paths are **absolutized** so the scorer — which runs in its
    own cwd (the OmniDocBench repo) — resolves them correctly regardless of where
    the CLI was invoked. (This was the root cause of the P2/P3 0.00-score bug:
    a relative ``--pred-dir`` resolved wrong in the scorer's cwd.) Existing
    POSIX/Windows absolute paths retain their original root semantics.
    """
    cfg = yaml.safe_load(_load_eval_template())
    cfg["end2end_eval"]["dataset"]["ground_truth"]["data_path"] = _eval_config_path(gt_json)
    cfg["end2end_eval"]["dataset"]["prediction"]["data_path"] = _eval_config_path(pred_dir)
    out_yaml = Path(out_yaml)
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    out_yaml.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")


def run_scorer(
    *, omnidocbench_repo: str, config_yaml: str, venv_python: str | None = None
) -> subprocess.CompletedProcess:
    """Run pdf_validation.py --config <cfg> inside the OmniDocBench repo."""
    py = venv_python or DEFAULT_VENV_PYTHON
    cmd = [py, "pdf_validation.py", "--config", str(config_yaml)]
    return subprocess.run(cmd, cwd=omnidocbench_repo, capture_output=True, text=True, check=False)


def parse_run_summary(result_dir: str | Path, save_name: str) -> dict:
    """Read per-task numbers from OmniDocBench's ``run_summary.json``
    (``notebook_metric_summary.metrics`` is the notebook-aligned source of truth).
    ``save_name`` = ``basename(pred_dir) + '_quick_match'``. ``formula_cdm`` is
    ``None`` when the subset has no display-formula pages (CDM did not run)."""
    result_dir = Path(result_dir)
    summary = json.loads((result_dir / f"{save_name}_run_summary.json").read_text(encoding="utf-8"))
    ms = summary["notebook_metric_summary"]["metrics"]

    def raw(key: str) -> float | None:
        return ms.get(key, {}).get("raw")

    text = raw("text_block_Edit_dist")
    cdm = raw("display_formula_CDM")  # None when no formula pages
    teds = raw("table_TEDS")
    order = raw("reading_order_Edit_dist")
    return {
        "overall": overall_score({"text_edit_dist": text, "formula_cdm": cdm, "table_teds": teds}),
        "text_edit_dist": text,
        "formula_cdm": cdm,
        "table_teds": teds,
        "reading_order_edit": order,
    }


def score_directory(
    *,
    gt_json: str,
    pred_dir: str,
    omnidocbench_repo: str | None = None,
    venv_python: str | None = None,
    skip_validation: bool = False,
    strict: bool = True,
) -> dict:
    """Validate (unless skipped) and score a prediction directory end-to-end.

    The OmniDocBench eval config is written into a PRIVATE temp dir (never the
    prediction dir) so repeated scoring passes the strict validator. Runs the
    scorer in the pinned OmniDocBench venv and parses ``run_summary.json``.

    Returns ``{"validation_report": Report|None, "metrics": {...}}``. Raises
    :class:`ScoringError` on validation failure or a non-zero scorer exit — the
    message is user-facing (no raw traceback). Shared by the CLI ``score``
    command and ``scripts/score_predictions.py``.
    """
    from mineru_rocm.validation import validate_predictions

    repo = omnidocbench_repo or DEFAULT_OMNIDOCBENCH_REPO
    venv = venv_python or DEFAULT_VENV_PYTHON
    # Fail fast with a user-facing error (not an opaque TypeError from
    # subprocess.run([None, ...])) when neither env var nor CLI arg supplied the
    # scorer venv / repo. The defaults were removed for OPSEC (host paths leaked).
    if not repo:
        raise ScoringError(
            "OmniDocBench repo not configured: set OMNIDOCBENCH_REPO or pass --omnidocbench-repo"
        )
    if not venv:
        raise ScoringError(
            "scorer venv python not configured: set OMNIDOCBENCH_VENV or pass --venv-python"
        )
    report = None
    if not skip_validation:
        report = validate_predictions(gt_json, pred_dir, strict=strict)
        ok = report.ok_strict if strict else report.ok
        if not ok:
            lines = [
                f"  [{'ERROR' if p.severity == 'error' else 'WARN'}] {p.code}: {p.message}" for p in report.problems
            ]
            raise ScoringError(
                f"predictions invalid ({len(report.errors())} error(s), "
                f"{len(report.warnings())} warning(s)); refusing to score:\n" + "\n".join(lines)
            )
    save_name = f"{Path(pred_dir).name}_quick_match"
    with tempfile.TemporaryDirectory(prefix="mineru_rocm_eval_") as tmpd:
        cfg_path = Path(tmpd) / "_eval_config.yaml"
        write_eval_config(gt_json=gt_json, pred_dir=pred_dir, out_yaml=cfg_path)
        res = run_scorer(omnidocbench_repo=repo, config_yaml=str(cfg_path), venv_python=venv)
        if res.returncode != 0:
            tail = (res.stderr or "")[-2000:]
            raise ScoringError(f"scorer failed (rc={res.returncode})\n{tail}")
        metrics = parse_run_summary(Path(repo) / "result", save_name)
    return {"validation_report": report, "metrics": metrics}


def format_score_table(label: str, metrics: dict) -> str:
    """Render the per-task metrics as the human-readable score table."""

    def fmt(v, pct=False):
        if v is None:
            return "n/a"
        return f"{v * 100:.2f}" if pct else f"{v:.4f}"

    ov = metrics["overall"]
    recomputed = overall_score(
        {
            "text_edit_dist": metrics["text_edit_dist"],
            "formula_cdm": metrics["formula_cdm"],
            "table_teds": metrics["table_teds"],
        }
    )
    return (
        f"\n=== {label} -- OmniDocBench v1.6 ===\n"
        f"  Overall          : {'n/a (CDM missing on this subset)' if ov is None else f'{ov:.2f}'}\n"
        f"  text  EditDist   : {fmt(metrics['text_edit_dist'])}   -> {fmt(metrics['text_edit_dist'], pct=True)}\n"
        f"  formula CDM      : {fmt(metrics['formula_cdm'])}   -> {fmt(metrics['formula_cdm'], pct=True)}\n"
        f"  table  TEDS      : {fmt(metrics['table_teds'])}   -> {fmt(metrics['table_teds'], pct=True)}\n"
        f"  order  EditDist  : {fmt(metrics['reading_order_edit'])}\n"
        f"  (overall recomputed: {'n/a' if recomputed is None else f'{recomputed:.2f}'})\n"
    )
