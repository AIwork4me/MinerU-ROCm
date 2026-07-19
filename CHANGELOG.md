# Changelog

All notable changes to MinerU-ROCm are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

Versions are tagged on `main`; see the spec under `docs/superpowers/specs/` for
the design behind each phase.

## [Unreleased] — P0: identity, license, lock skeleton

### Changed
- Reframed the repository identity from "adapter for the omnidocbench-amd
  platform" to an evaluation-backed AMD ROCm port of opendatalab/MinerU.
- Demoted `omnidocbench-amd` from a core dependency to an optional `[platform]`
  extra; the core package is now GPU-free and platform-free.
- Corrected the license posture to Apache-2.0 (base) with the MinerU Open
  Source License additional terms recorded for the pipeline-wrapping path.

### Added
- `reproducibility.lock.yaml` skeleton (single source of truth; values populated in P3).
- `NOTICE`, `LICENSES/`, `REUSE.toml` (mixed-license + SPDX REUSE compliance).
- `CITATION.cff`, `SECURITY.md`, `SUPPORT.md`.

### Notes
- No inference behavior or published score changed in this phase.

## [0.1.0] — prior to P0

- Initial adapter repo (omnidocbench-amd platform integration): MinerU 3.4
  pipeline (Overall 86.48) and MinerU2.5-Pro VLM via vLLM-on-ROCm (Overall 95.56)
  on OmniDocBench v1.6, gfx1100.
