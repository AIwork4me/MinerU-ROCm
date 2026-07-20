# Changelog

All notable changes to MinerU-ROCm are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

Versions are tagged on `main`; see `docs/superpowers/specs/` for the design
behind each phase and `reproducibility.lock.yaml` for the exact provenance.

## [Unreleased]

No changes since v0.1.0.

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
