# Changelog

All notable changes to MinerU-ROCm are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

Versions are tagged on `main`; see `docs/superpowers/specs/` for the design
behind each phase and `reproducibility.lock.yaml` for the exact provenance.

## [1.0.0] — 2026-07-23

First evidence-complete community release.

### Added

- A conformant Windows-HIP Phase 1 bundle for the MinerU 3.4.4 pipeline:
  1651/1651 pages, Overall 86.59, Windows ROCm PyTorch plus DirectML, with the
  DirectML-incompatible `slanet-plus.onnx` model recorded as an audited CPU
  override rather than a silent fallback.
- Platform-specific Windows model card, prediction manifest, dataset identity,
  scoring configuration, provenance, run summary, and run statistics.
- Release, architecture, and hardware-matrix documentation.

### Changed

- Package and citation version promoted to 1.0.0.
- The optional platform extra and CI now use the published, SHA256-pinned
  OmniDocBench-ROCm 0.3.2 wheel.

### Verified

- Linux ROCm: MinerU2.5-Pro VLM Overall 95.56 and pipeline Overall 86.48.
- Windows-HIP: pipeline Overall 86.59; VLM Phase 2 remains explicitly out of
  scope for this release.

## P1.1 — evidence consistency + self-contained bundles (2026-07-21)

Cross-repo consistency fix with OmniDocBench-ROCm P0.1 (platform 0.3.1).

### Changed — current canonical VLM result is now 95.56 (platform CDM)
- **The canonical current VLM Overall is 95.56** (platform CDM-scored,
  `omnidocbench-rocm run --cdm`), recomputed from the committed CDM metric via
  `((1−0.0359)·100 + 96.73 + 93.54)/3 = 95.5605`. **95.46 is the prior
  standalone `mineru-rocm score` result** (Formula CDM 96.46) on the **same
  1651 predictions** (commit `b75f788`). The Δ +0.10 pp is entirely the
  Formula-CDM submetric (CDM scoring configuration), not new inference. This
  reverses the 2026-07-20 demotion to 95.46-primary on the verified basis that
  95.56 recomputes from the CDM metric over the same prediction set.
- README + README.zh-CN badge/headline/results, `docs/reproducibility.md`,
  `docs/how-it-works.md`, `docs/benchmark-methodology.md` → 95.56-primary,
  95.46 retained only as the documented prior standalone score.

### Added — self-contained platform bundles
- `results/omnidocbench/v16/linux-rocm/` now holds self-contained CDM bundles
  for `mineru2.5` and `mineru-pipeline`: committed `metric_result` + `run_stats`
  + `scoring_config` + `dataset_identity` (revision `2b161d0`, GT sha
  `a45cd84b…`) + a run-driven SHA256 `prediction_manifest` (mineru2.5 1649 ok /
  2 failed; pipeline 1650 ok / 1 failed). Provenance splits
  `packaging_commit` from `prediction_source_commit` (mineru2.5 `b75f788` /
  pipeline `e05eec3`) and redacts runtime paths to `<eval-root>`. Generated via
  `omnidocbench-rocm publish` (migration_type `legacy_predictions_to_platform_artifacts`).
- `model_card.json` + `model_card.pipeline.json` repointed at the CDM bundles
  (fixes the prior cdm/non-cdm metric mismatch).

### Fixed
- `scripts/check_repo.py`: replaced the hard-coded `_STALE=95.56` /
  `_CURRENT=95.46` gate with a **data-driven** check sourced from
  `reproducibility.lock.yaml` (current overall) + the legacy v1.6 metric
  (prior overall): the README VLM badge must state the current overall, and the
  prior overall may appear only alongside the current one.
- `Makefile` `eval-*` targets: `--cdm` on by default, `--skip-existing` only
  with `RESUME=1` (was unconditional `--skip-existing`, no `--cdm`).
- `pyproject.toml`: `platform = ["omnidocbench-rocm>=0.3.1,<0.4"]` (was `>=0.2.0`).
- CI split into `core` + `platform-contract` jobs; the latter installs the
  pinned platform and runs `omnidocbench-rocm conformance .` +
  `scripts/validate_platform_artifacts.py`.
- `results/omnidocbench/v16/linux-rocm/README.md`: no longer claims the
  directory is empty / artifacts not yet generated.

## Upstream-readiness hardening (2026-07-20)

Hardening for upstream MinerU PR #5288 (ROCm docs contribution) — evidence-base consistency + OPSEC + falsifiability.
- P1 platform migration: `omnidocbench-amd` → `omnidocbench-rocm` (historical name: pre-P1 used omnidocbench-amd; now migrated to OmniDocBench-ROCm).

### Fixed
- `model_card.json` VLM Overall 95.56 → **95.46**; both model cards repointed from the superseded `v16/` engine artefacts to the authoritative `results/omnidocbench/v1.6/` set (`run_manifest` + `metric_result` + `sample_predictions`).
- `docs/reproducibility.md` rewritten to the standalone `mineru-rocm predict|score` path (was the pre-rewrite `omnidocbench-amd` workflow); quotes 95.46/86.48; no machine-local paths/IPs; documents `HSA_OVERRIDE` for both paths (pipeline = none, VLM = `11.0.0`).
- `docs/how-it-works.md` 95.56 → 95.46; standalone-CLI identity; `cuda`/HIP clarification.
- `Makefile` + README `Evaluation` drive `mineru-rocm predict|score`; dropped the machine-local `OMNIDOCBENCH_IMG_DIR` default.
- Pinned upstream commits in the lock: `mineru` @ `0dfc946`, `mineru_vl_utils` @ `cc467fa` (resolved via `git ls-remote`); recorded official anchors (pipeline 86.47, vlm-engine 95.30) from the upstream README.

### Changed
- Archived superseded `results/omnidocbench/v16/` under `results/_archive/v16-engine-superseded/` (provenance history; do not cite).
- Added a committed 10-page stratified prediction sample per backend under `results/omnidocbench/v1.6/{pipeline,vlm-vllm}/sample_predictions/` (the full-set predictions were already gitignored, never committed); `.gitignore` rules now document the sample-vs-fullset policy.
- `pipeline` backend's default `HF_ENDPOINT` is now the public `https://huggingface.co` (was an internal mirror unreachable by public users); `setdefault` still respects an exported `HF_ENDPOINT`.
- `mineru-rocm score` now requires the scorer venv/repo via `OMNIDOCBENCH_VENV`/`OMNIDOCBENCH_REPO` env vars or `--venv-python`/`--omnidocbench-repo` (was a host-specific default); fails fast with a clear `ScoringError` if unset.

### Security
- Redacted the internal HF-mirror IP and host eval-root/venv paths from all public artefacts under `results/` + `docs/`; added a `check_repo.py` no-leak gate.

### Added
- `scripts/check_repo.py` gates: modelcard↔lock tri-source agreement; no-stale-95.56; no-internal-infra leak (+ tests); no-withdrawn-anchor-claims (flags the withdrawn unofficial-anchor tokens across user-facing surfaces) across user-facing surfaces.
- `scripts/sample_predictions.py`, `scripts/redact_internal.py`.
- `docs/upstream-pr/` — staged docs-only contribution to `opendatalab/MinerU` (zh AMD.md section + README GPU row + #5288 process-gate comment).

### Resolved (previously open in [0.1.0])
- The **official upstream anchor** is now verified and sourced from the upstream README "Local Deployment" table (vlm-engine 95.30, pipeline 86.47) — see `reproducibility.lock.yaml` → `benchmark.official_reference: source: verified`. The prior withdrawn unofficial anchor is no longer cited anywhere in user-facing surfaces.
- `mineru`/`mineru_vl_utils` upstream commits are now pinned in the lock (resolved via `git ls-remote`).

## [0.1.0] — 2026-07-19

The first release of MinerU-ROCm as a **standalone, evaluation-backed AMD ROCm
port of [opendatalab/MinerU](https://github.com/opendatalab/MinerU)**. Previously
an omnidocbench-amd adapter; now a self-contained package with its own CLI,
scoring, and reproducibility lock.

### Added — standalone package + CLI + results

- **`src/mineru_rocm/`** — 12-module GPU-free core (`types`, `dispatcher`,
  `config`, `backends/{pipeline,vlm}`, `omnidbench`, `preflight`, `validation`,
  `canary`, `scoring`, `runner`, `driver`, `cli`). Imports with no `torch`/
  `mineru`/`omnidocbench_amd` at module top-level (all heavy deps lazy).
- **`mineru-rocm` CLI** — `doctor | validate | predict | score | canary
  materialize | manifest verify` (`[project.scripts]` entry).
- **`scripts/check_repo.py`** — CI consistency gate: AST no-engine-import scan,
  `pip install -e .` smoke (PEP 639 guard), SPDX-header check, README↔lock
  value cross-check (drift gate).
- **`reproducibility.lock.yaml`** — filled with byte-exact SHAs (code commit,
  model weights, GT json, eval config, scorer commit) + the re-run metrics +
  both venvs' full environment. The single source of truth.
- **Results reproduced** via `mineru-rocm predict | score` on AMD gfx1100:
  pipeline Overall **86.48** (byte-identical to prior), VLM Overall **95.46**
  (within ±0.5pp of prior 95.56; vLLM non-determinism).

### Changed — identity + decouple + license

- Reframed repository identity from "omnidocbench-amd adapter" → standalone
  evaluation-backed port of opendatalab/MinerU.
- `omnidocbench-amd` demoted from a core dependency to an optional `[platform]`
  extra; the local `types.py` replaces the `omnidocbench_amd.types` import.
- License posture corrected to Apache-2.0 (base) + MinerU Open Source License
  additional terms (`NOTICE`, `LICENSES/`, `REUSE.toml`).
- The engine subprocess contract (`adapter/run_adapter.py`, 8-key
  `_run_stats.json`, 7-flag CLI) is **byte-identical** — backward-compatible.

### Known limitations (alpha)

- Canary subset not yet materialized.

## [pre-rewrite] — prior to 2026-07-19

- Initial adapter repo (omnidocbench-amd platform integration): MinerU 3.4
  pipeline (Overall 86.48) and MinerU2.5-Pro VLM via vLLM-on-ROCm (Overall 95.56)
  on OmniDocBench v1.6, gfx1100.
