# MinerU-ROCm P0 — Identity, License & Lock Skeleton Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reframe MinerU-ROCm's identity from "adapter for the omnidocbench-amd platform" to a standalone, evaluation-backed AMD ROCm port of opendatalab/MinerU — fix the license declaration, demote the platform dependency to an optional extra, and lay the `reproducibility.lock.yaml` + governance scaffolding — with **no change to inference behavior or published scores**.

**Architecture:** Doc/config/structure-only phase (CPU-only, no GPU). The `src/mineru_rocm/` package, the `mineru-rocm` CLI, the local `types.py`, and the lock's verified values all arrive in later phases (P1 code, P3 re-run). P0 changes identity copy, packaging metadata, license files, a lock skeleton (all values `not_recorded`), and governance docs. CI stays green throughout because the not-yet-refactored `adapter/run_adapter.py` still imports `omnidocbench_amd.types`, so CI installs the new `[platform]` extra.

**Tech Stack:** Python 3.11+, setuptools, PyYAML, pytest, fsfe-reuse (SPDX), GitHub Actions.

## Global Constraints

(From the approved spec `docs/superpowers/specs/2026-07-19-mineru-rocm-standalone-port-design.md`; every task implicitly includes these.)

- **No behavior change.** P0 must not alter any inference path, scorer output, or published number. Current results (VLM 95.56, pipeline 86.48) stand unchanged.
- **Core package is GPU-free and platform-free.** `project.dependencies` MUST be `[]`; `omnidocbench-amd` lives ONLY in `optional-dependencies.platform`.
- **Identity hero links to `opendatalab/MinerU`** (the upstream model repo), NOT to `OmniDocBench-AMD`. The omnidocbench-amd engine is described as one *optional* consumer.
- **License is Apache-2.0 base**, with the MinerU Open Source License additional terms recorded for the pipeline-wrapping path, and the undeclared PDF-Extract-Kit-1.0 weights flagged. Verified 2026-07-19 — do NOT claim AGPL.
- **Lock skeleton uses `not_recorded` + a fill command** for every unverified field; never invent a value. Real (verified) values are filled in P3.
- **Bilingual READMEs** (`README.md` + `README.zh-CN.md`) stay in sync.
- **One concern per commit; commit after every task's validation passes.** Branch: `docs/standalone-port-design` (already holds the spec).

---

## File Structure (P0 scope)

| File | Action | Responsibility |
|---|---|---|
| `pyproject.toml` | Rewrite | Package metadata; `dependencies = []`; `[platform]`/`[dev]` extras; project URLs (Homepage + Upstream); Apache-2.0 license ref NOTICE |
| `.github/workflows/ci.yml` | Modify | Install line → `pip install -e ".[dev,platform]"`; drop redundant explicit install |
| `LICENSES/Apache-2.0.txt` | Create | Apache-2.0 full text (via `reuse download`) |
| `LICENSES/LicenseRef-MinerU-Open-Source-License.txt` | Create | MinerU OSL text (verbatim from local `mineru-3.4.4` dist-info) |
| `NOTICE` | Create | Mixed-license breakdown: Apache-2.0 base + MinerU additional terms + undeclared-weights caveat |
| `REUSE.toml` | Create | Bulk SPDX coverage of repo files as Apache-2.0 |
| `reproducibility.lock.yaml` | Create | Single-source-of-truth skeleton; all fields `not_recorded` + fill commands |
| `CHANGELOG.md` | Create | Release history (seeded with an `[Unreleased]` P0 entry) |
| `CITATION.cff` | Create | Cite this repo + MinerU2.5-Pro paper + MinerU |
| `SECURITY.md` | Create | Vulnerability reporting policy |
| `SUPPORT.md` | Create | Where to get help (issues, upstream, OmniDocBench) |
| `README.md` | Rewrite identity sections | Hero/tagline → opendatalab/MinerU; At a glance; license; Issues filed; reproducibility pointer; install uses `[platform]` |
| `README.zh-CN.md` | Mirror | Same reframing in Chinese |
| `docs/how-it-works.md` | Modify | First sentence → "benchmark infrastructure for evaluating MinerU on AMD ROCm" |

Out of P0 (later phases): `src/mineru_rocm/` package + CLI (P1), `scripts/check_repo.py` + `render_benchmark_tables.py` (P1/P3), verified lock values + canary (P3), `gpu-smoke.yml` + `release-artifact.md` (P4).

---

## Task 1: Reframe `pyproject.toml` + CI install line (demote omnidocbench-amd)

**Files:**
- Modify: `pyproject.toml` (full rewrite — currently 8 lines)
- Modify: `.github/workflows/ci.yml:9`
- Test: inline `python` validation (no new test file in P0; `check_repo.py` is P1)

**Interfaces:**
- Consumes: nothing (foundational).
- Produces: `project.dependencies == []`; `optional-dependencies.platform == ["omnidocbench-amd>=0.1.0"]`; `project.license == "Apache-2.0"`; `project.urls.Upstream`. Later tasks reference `[platform]` in install docs and `LICENSES/` in NOTICE.

- [ ] **Step 1: Write the validation check (will fail until pyproject is rewritten)**

Create `scripts/check_deps.py` (a tiny P0-only validator; superseded by `check_repo.py` in P1):

```python
#!/usr/bin/env python3
"""P0 validator: core is GPU/platform-free; omnidocbench-amd is only in [platform]."""
import sys, tomllib
from pathlib import Path

with open(Path(__file__).resolve().parents[1] / "pyproject.toml", "rb") as f:
    p = tomllib.load(f)
proj = p["project"]
assert proj["dependencies"] == [], f"core must have no deps, got {proj['dependencies']!r}"
extras = proj["optional-dependencies"]
assert extras.get("platform") == ["omnidocbench-amd>=0.1.0"], f"[platform] wrong: {extras.get('platform')!r}"
assert "omnidocbench-amd" not in extras.get("dev", []), "[dev] must not pull omnidocbench-amd (use [platform])"
assert proj["license"] == "Apache-2.0", f"license must be Apache-2.0, got {proj['license']!r}"
assert proj["urls"]["Upstream"] == "https://github.com/opendatalab/MinerU", "Upstream URL missing"
print("P0 pyproject OK")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python scripts/check_deps.py`
Expected: FAIL with `KeyError: 'optional-dependencies'` (current pyproject has none) or an assertion error.

- [ ] **Step 3: Rewrite `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "mineru-rocm"
version = "0.1.0"
description = "Evaluation-backed AMD ROCm port of opendatalab/MinerU (3.4 pipeline + 2.5-Pro VLM) for OmniDocBench v1.6"
readme = "README.md"
requires-python = ">=3.11"
# Base license is Apache-2.0. The MinerU Open Source License adds commercial-threshold
# and attribution terms that apply to the pipeline-wrapping path — see NOTICE and LICENSES/.
license = "Apache-2.0"
authors = [{ name = "AIwork4me" }]
keywords = [
  "mineru", "document-parsing", "amd-gpu", "rocm", "rdna3",
  "vllm", "omnidocbench", "vision-language-model",
]
classifiers = [
  "Development Status :: 3 - Alpha",
  "Intended Audience :: Science/Research",
  "Operating System :: POSIX :: Linux",
  "Programming Language :: Python :: 3",
  "Programming Language :: Python :: 3.11",
  "Programming Language :: Python :: 3.12",
  "Topic :: Scientific/Engineering :: Artificial Intelligence",
  "License :: OSI Approved :: Apache Software License",
]
# Core package is GPU-free and has NO platform dependency. The omnidocbench-amd
# engine integration is OPTIONAL (install [platform]); the standalone CLI lands in P1.
dependencies = []

[project.optional-dependencies]
# Optional: omnidocbench-amd platform integration. The current adapter/run_adapter.py
# imports omnidocbench_amd.types; this coupling is removed in P1 (local types.py).
platform = ["omnidocbench-amd>=0.1.0"]
dev = ["pytest>=8", "ruff>=0.6", "reuse>=1.3", "build"]

[project.urls]
Homepage = "https://github.com/AIwork4me/MinerU-ROCm"
Repository = "https://github.com/AIwork4me/MinerU-ROCm"
Issues = "https://github.com/AIwork4me/MinerU-ROCm/issues"
Upstream = "https://github.com/opendatalab/MinerU"
```

- [ ] **Step 4: Update the CI install line**

In `.github/workflows/ci.yml`, replace line 9:

```yaml
      - run: pip install -e ".[dev,platform]" && pip install omnidocbench-amd pytest
```

with:

```yaml
      # [dev] = pytest/ruff/reuse; [platform] = omnidocbench-amd (the adapter still
      # imports omnidocbench_amd.types until P1 lands a local types.py). The redundant
      # explicit installs are no longer needed once both extras are pulled.
      - run: pip install -e ".[dev,platform]"
```

- [ ] **Step 5: Run the validator + the existing test suite**

Run: `python scripts/check_deps.py`
Expected: `P0 pyproject OK`

Run: `pip install -e ".[dev,platform]" && pytest -q`
Expected: all existing tests pass (smoke/pipeline-routing/vlm-routing/unknown-backend/per-page-failure — unchanged behavior).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .github/workflows/ci.yml scripts/check_deps.py
git commit -m "build(p0): reframe pyproject — Apache-2.0, demote omnidocbench-amd to [platform] extra"
```

---

## Task 2: License files — `LICENSES/`, `NOTICE`, `REUSE.toml`

**Files:**
- Create: `LICENSES/Apache-2.0.txt`
- Create: `LICENSES/LicenseRef-MinerU-Open-Source-License.txt`
- Create: `NOTICE`
- Create: `REUSE.toml`

**Interfaces:**
- Consumes: nothing.
- Produces: a passing `reuse lint`; `NOTICE` is referenced by the README license section (Task 5) and by `pyproject.toml`'s license comment.

- [ ] **Step 1: Add the Apache-2.0 license text**

Run (requires the `reuse` tool from `[dev]`):
```bash
reuse download license Apache-2.0
```
Expected: creates `LICENSES/Apache-2.0.txt` with the full Apache-2.0 text. If `reuse download` is unavailable, fetch equivalent text from `https://www.apache.org/licenses/LICENSE-2.0.txt` into `LICENSES/Apache-2.0.txt`.

- [ ] **Step 2: Create `LICENSES/LicenseRef-MinerU-Open-Source-License.txt`**

Write this exact content (verbatim MinerU OSL, captured from the local `mineru-3.4.4` dist-info `licenses/LICENSE.md`):

```text
MinerU Open Source License

MinerU is licensed under Apache License 2.0 and is subject to the additional
terms below. Except to the extent expressly modified or supplemented by these
additional terms, your other rights and obligations are governed by Apache
License 2.0.

1. Commercial License and Thresholds

MinerU may be used for commercial purposes without a separate commercial
license. However, if you and your Affiliates, on a consolidated basis, meet
either of the following thresholds, you must obtain a separate commercial
license from [MinerU Team] before continuing such use:

a. monthly active users (MAU) exceed 100 million; or
b. total monthly revenue exceeds USD 20 million.

2. Online Service Attribution Obligation

If you provide online services to third parties based on MinerU, you must
clearly and prominently indicate, in the relevant product or service interface
or in publicly available documentation, that MinerU is used.

3. Termination

Where a separate commercial license is required under Section 1 but is not
obtained before continuing such use, or where the attribution obligation under
Section 2 is not complied with, this License and all rights granted under this
License will terminate automatically, and no further notice from the Licensor
is required.

4. Definitions

In these additional terms, "Affiliates" means any legal entity that directly or
indirectly controls, is controlled by, or is under common control with you.
"Control" means the power to direct the management and operating decisions of
an entity, whether through equity ownership, voting rights, contractual
arrangements, or otherwise.

The full text of Apache License 2.0 is available at
https://www.apache.org/licenses/LICENSE-2.0 .

Source: https://github.com/opendatalab/MinerU/blob/master/LICENSE.md
```

- [ ] **Step 3: Create `NOTICE`**

```text
MinerU-ROCm
Copyright 2026 AIwork4me

This repository is an evaluation-backed AMD ROCm port of opendatalab/MinerU.
It is NOT affiliated with, sponsored, or endorsed by the MinerU Team or
OpenDataLab.

LICENSE OVERVIEW
----------------
All original packaging, tooling, and documentation in this repository are
licensed under the Apache License 2.0 (see LICENSES/Apache-2.0.txt and the
top-level LICENSE).

COMPONENT LICENSES
------------------
1. Upstream `mineru` pipeline (opendatalab/MinerU, v3.4.x) — wrapped by the
   pipeline backend — is under the MinerU Open Source License: Apache-2.0 PLUS
   additional terms (commercial-use threshold: MAU > 100 million OR monthly
   revenue > USD 20 million requires a separate commercial license; online
   services built on MinerU must attribute MinerU; termination on breach).
   See LICENSES/LicenseRef-MinerU-Open-Source-License.txt.
   Source: https://github.com/opendatalab/MinerU

2. `mineru-vl-utils` (opendatalab/mineru-vl-utils, v1.0.x) — drives the VLM
   two-step inference — is Apache-2.0.
   Source: https://github.com/opendatalab/mineru-vl-utils

3. MinerU2.5-Pro-2605-1.2B model weights — Apache-2.0 (per HF cardData).
   Source: https://huggingface.co/opendatalab/MinerU2.5-Pro-2605-1.2B

4. PDF-Extract-Kit-1.0 pipeline weights (layout/formula/OCR/table sub-models)
   — NO license is declared on the Hugging Face model card (cardData license
   field is empty, verified 2026-07-19). Treat as license-ambiguous; do NOT
   redistribute these weights. Tracked in docs/known-gaps.md; an upstream
   clarification issue is planned.

5. omnidocbench-amd engine (OPTIONAL, install via the [platform] extra) —
   Apache-2.0. Used only by the optional platform-integration shim, not by the
   standalone evaluation path.

There is NO AGPL component and NO copyleft network-use source-disclosure
obligation in this repository's dependency set.
```

- [ ] **Step 4: Create `REUSE.toml`**

```toml
# SPDX-REUSE configuration. Bulk-covers original repo files as Apache-2.0.
# Vendored upstream files (added in P1) will carry their own annotations.
version = 1

[[annotations]]
path = [
  "**/*.py",
  "**/*.md",
  "**/*.yaml",
  "**/*.yml",
  "**/*.toml",
  "**/*.cff",
  "**/*.sh",
  "**/*.ps1",
  "**/*.jinja",
  "Makefile",
  "LICENSE",
]
precedence = "aggregate"
SPDX-FileCopyrightText = "2026 AIwork4me"
SPDX-License-Identifier = "Apache-2.0"

[[annotations]]
path = ["examples/*.png", "docs/assets/*"]
precedence = "aggregate"
SPDX-FileCopyrightText = "2026 AIwork4me"
SPDX-License-Identifier = "Apache-2.0"
```

- [ ] **Step 5: Run `reuse lint`**

Run: `pip install -e ".[dev]" && reuse lint`
Expected: `# Successfully finished reading REUSE.toml` and `Congratulations! Your project is compliant with version 3.0 of the REUSE Specification :-)`. If it flags specific files, either add them to `REUSE.toml` paths or drop them into a `.reuse/ignore` — fix until green.

- [ ] **Step 6: Commit**

```bash
git add LICENSES NOTICE REUSE.toml
git commit -m "license(p0): mixed-license NOTICE + SPDX REUSE (Apache-2.0 + MinerU OSL terms)"
```

---

## Task 3: `reproducibility.lock.yaml` skeleton (single source of truth)

**Files:**
- Create: `reproducibility.lock.yaml`

**Interfaces:**
- Consumes: nothing.
- Produces: the lock skeleton that Task 5's README reproducibility section points to, and that P3 fills with verified values. README result tables are NOT yet lock-generated (that wiring is P1/P3) — P0 only creates the skeleton.

- [ ] **Step 1: Create `reproducibility.lock.yaml`**

Every unverified field is `not_recorded` with a fill command in a comment; verified-by-construction fields (e.g. repo URL, gpu_arch) are filled now.

```yaml
# reproducibility.lock.yaml — machine-readable reproducibility snapshot for the
# MinerU-ROCm results. Verified fields carry a source comment; unverified fields
# are `not_recorded` with a fill command. Do not invent values.
#
# STATUS: P0 skeleton. Real values are populated in P3 after the full-set re-run
# and byte-exact SHA cross-checks. See
# docs/superpowers/specs/2026-07-19-mineru-rocm-standalone-port-design.md §3.3.
#
# Verification status legend:
#   # (verified)     = recomputed and recorded
#   # (not_recorded) = fill command provided; populated in P3

mineru_rocm:
  repo: https://github.com/AIwork4me/MinerU-ROCm
  commit: not_recorded   # (not_recorded) git rev-parse HEAD at the published-results commit (P3)

mineru:
  repo: https://github.com/opendatalab/MinerU
  commit: not_recorded   # (not_recorded) fill: cd <mineru source> && git rev-parse HEAD  (P3)
  version: not_recorded  # (not_recorded) fill: pip show mineru | grep Version  (P3; expect 3.4.x)

mineru_vl_utils:
  repo: https://github.com/opendatalab/mineru-vl-utils
  commit: not_recorded   # (not_recorded) fill: pip show mineru-vl-utils (P3)
  version: not_recorded  # (not_recorded) expect 1.0.x

model:
  vlm:                                   # MinerU2.5-Pro-2605-1.2B
    hf_repo: opendatalab/MinerU2.5-Pro-2605-1.2B
    benchmark_artifact:
      safetensors_sha256: not_recorded   # (not_recorded) fill: sha256sum <snapshot>/model-*.safetensors (P3)
      config_sha256: not_recorded        # (not_recorded) fill: sha256sum <snapshot>/config.json (P3)
    current_remote_artifact:             # cross-checked byte-for-byte vs upstream HF via hf-mirror (P3)
      hf_repo_revision: not_recorded     # (not_recorded) fill: HF_ENDPOINT=https://hf-mirror.com huggingface-cli download ... --revision
      safetensors_lfs_oid: not_recorded  # (not_recorded) must == benchmark_artifact.safetensors_sha256
      cross_check_source: "https://hf-mirror.com (HuggingFace mirror; official API data)"
  pipeline_weights:                      # opendatalab/PDF-Extract-Kit-1.0 sub-models
    layout_sha256: not_recorded          # (not_recorded) fill in P3
    formula_sha256: not_recorded
    ocr_sha256: not_recorded
    table_sha256: not_recorded

omnidocbench:
  version: v1.6
  scorer_repo_url: https://github.com/opendatalab/OmniDocBench
  scorer_commit: not_recorded            # (not_recorded) fill: cd <OmniDocBench> && git rev-parse HEAD (P3)
  gt_json_full_sha256: not_recorded      # (not_recorded) fill: sha256sum OmniDocBench.json (P3; 1651 pages)
  gt_json_canary_sha256: not_recorded    # (not_recorded) populated when canary lands (P3)
  eval_config_sha256: not_recorded       # (not_recorded) P3
  canary_manifest_sha256: not_recorded   # (not_recorded) P3
  metric:
    overall_formula: "((1 - text_EditDist) * 100 + table_TEDS + formula_CDM) / 3"
    aggregation: page.ALL                # OmniDocBench page_avg convention (NOT sample-weighted .all)
    match_method: quick_match
    note: "reading_order EditDist is reported separately and is NOT part of Overall"

environment:
  python: not_recorded                   # (not_recorded) fill: python -c "import sys; print(sys.version)" (P3)
  rocm_hip: not_recorded                 # (not_recorded) fill: python -c "import torch; print(torch.version.hip)" (P3)
  torch: not_recorded                    # (not_recorded) P3
  vllm: not_recorded                     # (not_recorded) P3
  transformers: not_recorded             # (not_recorded) P3
  mineru: not_recorded                   # (not_recorded) P3
  mineru_vl_utils: not_recorded          # (not_recorded) P3
  gpu_arch: gfx1100                      # (per repo docs) RDNA3
  rocm_smi_device_id: not_recorded       # (not_recorded) fill: rocm-smi --showproductname (P3)

benchmark:
  date: not_recorded                     # (not_recorded) P3
  hardware: "AMD gfx1100 (Radeon PRO W7900, 48 GB), ROCm 7.2"
  canary_N:                              # populated in P3 (canary size + page count finalized there)
    pipeline_overall: not_recorded
    vlm_vllm_overall: not_recorded
    vlm_transformers_overall: not_recorded
  full_1651:
    pipeline_overall: not_recorded       # (not_recorded) P3 re-run target ~86.48 (±1.0 pp of verified official)
    vlm_vllm_overall: not_recorded       # (not_recorded) P3 re-run target ~95.56 (±0.5 pp of verified official)
    vlm_transformers: not_run            # sample-only; ~44 h full impractical (diagnostic)
  official_reference:                    # re-verified against authoritative sources in P3
    source: not_recorded                 # (not_recorded) P3: opendatalab/OmniDocBench leaderboard + arXiv:2604.04771 + MinerU README
    vlm_overall: not_recorded            # (not_recorded) P3: authoritative value (95.69 vs 95.75 — re-verify)
    pipeline_overall: not_recorded       # (not_recorded) P3: authoritative value (~86.47; submetrics not published upstream)
    inference_engine: not_recorded       # (not_recorded) P3: upstream engine if stated (else "unspecified")
    provenance_note: "Third-party figures not in an official source are treated as not_verified."
```

- [ ] **Step 2: Validate it parses**

Run: `python -c "import yaml; d=yaml.safe_load(open('reproducibility.lock.yaml')); assert d['mineru_rocm']['repo']; assert d['environment']['gpu_arch']=='gfx1100'; print('lock skeleton parses OK,', len(d), 'top-level keys')"`
Expected: `lock skeleton parses OK, 7 top-level keys`

- [ ] **Step 3: Commit**

```bash
git add reproducibility.lock.yaml
git commit -m "repro(p0): reproducibility.lock.yaml skeleton — single source of truth (values not_recorded, P3 fills)"
```

---

## Task 4: Governance + release scaffolding — `CHANGELOG`, `CITATION`, `SECURITY`, `SUPPORT`

**Files:**
- Create: `CHANGELOG.md`, `CITATION.cff`, `SECURITY.md`, `SUPPORT.md`

**Interfaces:**
- Consumes: nothing.
- Produces: standard community files referenced by GitHub UI and by Task 5's README.

- [ ] **Step 1: Create `CHANGELOG.md`**

```markdown
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
```

- [ ] **Step 2: Create `CITATION.cff`**

```yaml
cff-version: 1.2.0
message: "If you use this work, please cite both this repository and the MinerU2.5-Pro technical report."
title: "MinerU-ROCm: an evaluation-backed AMD ROCm port of opendatalab/MinerU"
abstract: >-
  Tooling to run the MinerU 3.4 pipeline and MinerU2.5-Pro VLM on AMD gfx1100
  (RDNA3) and score them on OmniDocBench v1.6. Evaluation-backed, not
  precision-aligned.
authors:
  - alias: AIwork4me
type: software
version: "0.1.0"
date-released: "2026-07-19"
license: Apache-2.0
keywords:
  - document-parsing
  - AMD-ROCm
  - RDNA3
  - vision-language-model
  - OmniDocBench
repository-code: "https://github.com/AIwork4me/MinerU-ROCm"
preferred-citation:
  type: article
  title: "MinerU2.5-Pro Technical Report"
  url: "https://arxiv.org/abs/2604.04771"
```

- [ ] **Step 3: Create `SECURITY.md`**

```markdown
# Security Policy

## Reporting a vulnerability

Please DO NOT open a public GitHub issue for security problems.

Email the maintainer via the GitHub Security Advisories tab
(**Security → Report a vulnerability** on
https://github.com/AIwork4me/MinerU-ROCm), or open a private security advisory.

Include:
- A description of the issue and its impact.
- Steps to reproduce (minimal).
- Affected versions/commits.

You should receive an acknowledgement within a few days.

## Scope

This repository is **evaluation tooling**. Vulnerabilities in upstream
dependencies (MinerU, vLLM, OmniDocBench scorer, ROCm) should be reported to
their respective projects; please also file an issue here so we can pin or
document a workaround.
```

- [ ] **Step 4: Create `SUPPORT.md`**

```markdown
# Getting help

| Question type | Where |
|---|---|
| Bug or unexpected result in THIS repo | [GitHub Issues](https://github.com/AIwork4me/MinerU-ROCm/issues) — include the OmniDocBench page id, the backend, and the `reproducibility.lock.yaml` environment block. |
| ROCm / gfx1100 compatibility | Open an issue with `rocm-smi --showproductname` output; mark it `rocm-compat`. |
| Upstream MinerU behavior (model quality, pipeline options) | [opendatalab/MinerU](https://github.com/opendatalab/MinerU) — this repo wraps upstream, it does not change the models. |
| OmniDocBench scoring / metrics | [opendatalab/OmniDocBench](https://github.com/opendatalab/OmniDocBench). |
| Security | See [SECURITY.md](SECURITY.md) — do not open a public issue. |

Before filing, please run `python scripts/check_deps.py` (P0) / the future
`mineru-rocm doctor` (P1) and include its output.
```

- [ ] **Step 5: Validate**

Run: `python -c "import yaml; yaml.safe_load(open('CITATION.cff')); print('CITATION.cff parses OK')"`
Expected: `CITATION.cff parses OK`

Run: `ls -1 CHANGELOG.md CITATION.cff SECURITY.md SUPPORT.md`
Expected: all four files listed.

- [ ] **Step 6: Commit**

```bash
git add CHANGELOG.md CITATION.cff SECURITY.md SUPPORT.md
git commit -m "docs(p0): governance scaffolding — CHANGELOG, CITATION, SECURITY, SUPPORT"
```

---

## Task 5: Reframe README identity (EN + zh-CN) + `how-it-works`

**Files:**
- Modify: `README.md` (identity sections only — hero, badges, At a glance, Install, License, Issues filed, Reproducibility pointer; **keep the existing results tables unchanged**)
- Modify: `README.zh-CN.md` (mirror the same sections in Chinese)
- Modify: `docs/how-it-works.md` (first sentence)

**Interfaces:**
- Consumes: `NOTICE` + `LICENSES/` (Task 2), `reproducibility.lock.yaml` (Task 3), `[platform]` extra (Task 1).
- Produces: the user-facing identity that links to `opendatalab/MinerU` as the primary upstream.

- [ ] **Step 1: Replace the README header + first paragraph**

Replace the top of `README.md` (the `# MinerU-ROCm` heading through the first bullet block) with:

```markdown
# MinerU-ROCm

> Evaluation-backed AMD ROCm port of [MinerU](https://github.com/opendatalab/MinerU)
> — runs the **MinerU 3.4 pipeline** and the **MinerU2.5-Pro** VLM on AMD
> **gfx1100 (RDNA3)** and reports **OmniDocBench v1.6** results across multiple
> inference backends. **Not** a precision-aligned port: no same-page-set CUDA
> control exists, and the upstream headline may use a different engine. See
> [Benchmark methodology](docs/benchmark-methodology.md) *(lands in P2)*.

[![OmniDocBench v1.6](https://img.shields.io/badge/OmniDocBench-v1.6-blue)](https://github.com/opendatalab/OmniDocBench)
[![VLM full](https://img.shields.io/badge/MinerU2.5--Pro%20VLM%20(full)-95.56-green)](#results--mineru25-pro-vlm)
[![pipeline full](https://img.shields.io/badge/MinerU%203.4%20pipeline%20(full)-86.48-yellowgreen)](#results--mineru-34-pipeline)
[![status: evaluation-backed](https://img.shields.io/badge/status-evaluation--backed-blue)](reproducibility.lock.yaml)
[![license: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0%20(+MinerU%20terms)-blue)](NOTICE)

## At a glance

- **What it is.** Tooling to run opendatalab MinerU (3.4 pipeline + 2.5-Pro VLM) on AMD ROCm and score it on OmniDocBench v1.6.
- **Where verified.** AMD **gfx1100 (RDNA3, 48 GB ×4), ROCm 7.2**, bf16.
- **Most reliable results.** **MinerU2.5-Pro VLM (vLLM-on-ROCm) full 1651 = 95.56 Overall**; **MinerU 3.4 pipeline full 1651 = 86.48 Overall**.
- **Most important limitation.** **Not precision-aligned.** No same-engine CUDA control exists; the upstream headline may be measured with a different engine. The "official 95.75" anchor is being re-verified (upstream points to ~95.69) — see `reproducibility.lock.yaml` once populated.
- **Upstream.** This is a port OF [opendatalab/MinerU](https://github.com/opendatalab/MinerU); the [omnidocbench-amd](https://github.com/AIwork4me/OmniDocBench-AMD) engine is one *optional* consumer (install the `[platform]` extra), not the definition of this repo.
```

- [ ] **Step 2: Update the Install section**

Replace the existing Install block with:

```markdown
## Install

The core package is GPU-free and has no platform dependency.

```bash
pip install -e ".[dev]"          # core + dev/CI tooling (pytest, ruff, reuse)
# optional: omnidocbench-amd engine integration (the adapter/run_adapter.py path)
pip install -e ".[platform]"
```

For platform provisioning (weights, ROCm runtime), run `make setup-linux` (or
`make setup-windows`). GPU backends additionally need a ROCm torch + (VLM)
vLLM-on-ROCm, installed separately from a verified ROCm wheel source — see
`docs/reproducibility.md`.
```

- [ ] **Step 3: Update the License + Reproducibility + add Issues-filed sections**

Append/replace the License section (keep the existing Results sections verbatim) with:

```markdown
## License — read before downloading weights

This repo is **Apache-2.0** (original packaging/tooling). The MinerU pipeline is
under the **MinerU Open Source License** (Apache-2.0 + additional terms:
commercial use above MAU 100M or USD 20M/mo revenue needs a separate license;
online services must attribute MinerU). `mineru-vl-utils` and the MinerU2.5-Pro
weights are Apache-2.0. The **PDF-Extract-Kit-1.0** pipeline weights declare **no
license** on their HF card — treat as license-ambiguous, do not redistribute. Full
breakdown in [NOTICE](NOTICE) and [LICENSES/](LICENSES). Not affiliated with the
MinerU Team / OpenDataLab.

## Reproducibility

[`reproducibility.lock.yaml`](reproducibility.lock.yaml) is the single source of
truth — pinned commits, byte-exact weight/GT SHA256 cross-checked against the
upstream HF repos, environment versions, and the metric formula. *(P0 ships the
skeleton; verified values are populated in P3 after the full-set re-run.)* See
[docs/reproducibility.md](docs/reproducibility.md).

## Issues filed

- **[ROCm/AMDMIGraphX#5078](https://github.com/ROCm/AMDMIGraphX/issues/5078)** — Loop-subgraph parser bug affecting ONNX table recognition on ROCm.
- Upstream `opendatalab/MinerU` AMD.md contribution + PDF-Extract-Kit-1.0 license clarification are planned (P4).
```

- [ ] **Step 4: Mirror the same reframing in `README.zh-CN.md`**

Apply the Chinese equivalents of Steps 1–3 to `README.zh-CN.md`. Use this exact hero + At a glance copy (keep the existing results tables unchanged):

```markdown
# MinerU-ROCm

> [opendatalab/MinerU](https://github.com/opendatalab/MinerU) 的**评估背书 AMD ROCm 移植** ——
> 在 AMD **gfx1100 (RDNA3)** 上运行 **MinerU 3.4 pipeline** 与 **MinerU2.5-Pro** VLM，
> 跨多个推理后端报告 **OmniDocBench v1.6** 结果。**非**精度对齐移植：不存在同页集 CUDA 对照，
> 且上游 headline 可能用不同引擎。见 [基准方法学](docs/benchmark-methodology.md) *（P2 落地）*。

## 概览（At a glance）

- **是什么。** 在 AMD ROCm 上运行 opendatalab MinerU（3.4 pipeline + 2.5-Pro VLM）并在 OmniDocBench v1.6 上评分的工具。
- **在哪验证。** AMD **gfx1100 (RDNA3, 48 GB ×4)，ROCm 7.2**，bf16。
- **最可靠结果。** **MinerU2.5-Pro VLM (vLLM-on-ROCm) 全量 1651 = 95.56 Overall**；**MinerU 3.4 pipeline 全量 1651 = 86.48 Overall**。
- **最重要限制。** **非精度对齐。** 无同引擎 CUDA 对照；上游 headline 可能用不同引擎测量。所谓"官方 95.75"锚点正在重新核实（上游指向 ~95.69）—— 见填充后的 `reproducibility.lock.yaml`。
- **上游。** 本仓是 [opendatalab/MinerU](https://github.com/opendatalab/MinerU) 的移植；[omnidocbench-amd](https://github.com/AIwork4me/OmniDocBench-AMD) 引擎只是**可选**消费者（装 `[platform]` extra），不是本仓的定义。
```

Also translate the Install (使用 `pip install -e ".[dev]"` / `".[platform]"`)、License（Apache-2.0 + MinerU 附加条款 + PDF-Extract-Kit 权重未声明）、Reproducibility（指向 lock）、Issues filed 段。The license summary in Chinese:

```markdown
## License —— 下载权重前必读

本仓为 **Apache-2.0**（原创打包/工具）。MinerU pipeline 遵循 **MinerU Open Source License**（Apache-2.0 + 附加条款：MAU 超 1 亿或月营收超 2000 万美元需另获商业授权；在线服务须标注 MinerU）。`mineru-vl-utils` 与 MinerU2.5-Pro 权重为 Apache-2.0。**PDF-Extract-Kit-1.0** pipeline 权重在 HF 卡片上**未声明** license —— 视为授权不明，请勿再分发。完整分解见 [NOTICE](NOTICE) 与 [LICENSES/](LICENSES)。本仓与 MinerU Team / OpenDataLab 无隶属关系。
```

- [ ] **Step 5: Reframe `docs/how-it-works.md` first sentence**

Replace its opening line:

```text
`MinerU-ROCm` is a per-model adapter repo for the **omnidocbench-amd** engine. The engine drives the OmniDocBench v1.6 pipeline; this repo only supplies the model-specific inference step.
```

with:

```text
`MinerU-ROCm` is **benchmark infrastructure for evaluating opendatalab MinerU on AMD ROCm** — not a model port. It runs the MinerU 3.4 pipeline and the MinerU2.5-Pro VLM, and scores them on OmniDocBench v1.6. The standalone CLI (`mineru-rocm`, lands in P1) is the primary interface; the omnidocbench-amd engine remains an *optional* consumer via the `[platform]` extra.
```

- [ ] **Step 6: Validate — links + smoke demo still work**

Run: `grep -nE 'opendatalab/MinerU|omnidocbench-amd' README.md | head`
Expected: the first `opendatalab/MinerU` reference appears ABOVE the first `omnidocbench-amd` reference (upstream leads, platform is secondary).

Run: `bash examples/run_demo.sh` (smoke backend, no GPU)
Expected: completes; writes a `.md` per image in a temp dir (behavior unchanged).

Run: `pytest -q`
Expected: all green (no behavior change).

- [ ] **Step 7: Commit**

```bash
git add README.md README.zh-CN.md docs/how-it-works.md
git commit -m "docs(p0): reframe identity — port of opendatalab/MinerU; omnidocbench-amd demoted to optional consumer"
```

---

## Definition of Done (P0)

- [ ] `python scripts/check_deps.py` → `P0 pyproject OK`.
- [ ] `reuse lint` → compliant.
- [ ] `reproducibility.lock.yaml` parses; every leaf is a real value or `not_recorded` + fill command.
- [ ] `CHANGELOG`/`CITATION`/`SECURITY`/`SUPPORT` present; `CITATION.cff` parses.
- [ ] README hero links to `opendatalab/MinerU` first; existing results tables unchanged.
- [ ] `pytest -q` green; `examples/run_demo.sh` smoke works; `pip install -e ".[dev,platform]"` succeeds.
- [ ] All five tasks committed on `docs/standalone-port-design` (push uses the gh-API fallback per the known env limitation).

## Follow-on phases (separate plans)

- **P1** — `src/mineru_rocm/` package + `mineru-rocm` CLI + port `runner`/`validation`/`scoring`/`omnidocbench`/`endpoint_pool`/`preflight` from Hunyuan; local `types.py` replaces `omnidocbench_amd.types`; remove the `[dev]`→platform coupling; add `scripts/check_repo.py`.
- **P2** — `benchmark-methodology.md`, `architecture.md`, `hardware-matrix.md`, expand `known-gaps.md`.
- **P3** — re-verify the official anchor; build the canary; re-run pipeline + VLM full-set with the new runner (resume fix); score twice; populate `reproducibility.lock.yaml` with verified SHAs; render README results from the lock.
- **P4** — cut v1.0.0 (wheel + SHA256SUMS + tags); `gpu-smoke.yml` GPU-CI bridge; deepen tests; file upstream `opendatalab/MinerU` AMD.md contribution + PDF-Extract-Kit license clarification.
