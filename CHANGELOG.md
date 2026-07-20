# Changelog

All notable changes to MinerU-ROCm are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

Versions are tagged on `main`; see `docs/superpowers/specs/` for the design
behind each phase and `reproducibility.lock.yaml` for the exact provenance.

## [Unreleased]

Hardening for upstream MinerU PR #5288 (ROCm docs contribution) — evidence-base consistency + OPSEC + falsifiability.

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
- `scripts/check_repo.py` gates: modelcard↔lock tri-source agreement; no-stale-95.56; no-internal-infra leak (+ tests).
- `scripts/sample_predictions.py`, `scripts/redact_internal.py`.
- `docs/upstream-pr/` — staged docs-only contribution to `opendatalab/MinerU` (zh AMD.md section + README GPU row + #5288 process-gate comment).

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

- The **official upstream anchor is unverified** (~95.69, NOT the prior
  unverified "95.75") — see `reproducibility.lock.yaml` →
  `benchmark.official_reference: not_verified`.
- Canary subset not yet materialized.
- `mineru`/`mineru_vl_utils` upstream commits not in the lock (pip-installed;
  commit not in dist-info).

## [pre-rewrite] — prior to 2026-07-19

- Initial adapter repo (omnidocbench-amd platform integration): MinerU 3.4
  pipeline (Overall 86.48) and MinerU2.5-Pro VLM via vLLM-on-ROCm (Overall 95.56)
  on OmniDocBench v1.6, gfx1100.
