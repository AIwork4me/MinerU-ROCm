# MinerU-ROCm — Standalone Evaluation-Backed Port (Quality Alignment to HunyuanOCR-ROCm)

| | |
|---|---|
| **Date** | 2026-07-19 |
| **Status** | Approved (design); awaiting implementation plan |
| **Author** | Claude (AIwork4me) |
| **Upstream model** | [opendatalab/MinerU](https://github.com/opendatalab/MinerU) v3.4.4 (pipeline) + [MinerU2.5-Pro-2605-1.2B](https://huggingface.co/opendatalab/MinerU2.5-Pro-2605-1.2B) (VLM) |
| **Reference repo (quality bar)** | [AIwork4me/HunyuanOCR-ROCm](https://github.com/AIwork4me/HunyuanOCR-ROCm) — "Evaluation-backed AMD ROCm port of HunyuanOCR-1.5" |
| **Supersedes** | [`2026-07-17-mineru-rocm-design.md`](2026-07-17-mineru-rocm-design.md) §5.2 / §6 framing (adapter-for-platform → port-of-upstream). The 2026-07-17 spec's technical content (two-model dispatcher, two-step VLM, ONNX-tables-on-CPU, ROCm recipe) is retained and carried forward; only the **identity / ownership / rigor posture** changes here. |
| **Lives in** | this repo, `docs/superpowers/specs/` |

---

## 0. TL;DR

MinerU-ROCm currently presents itself as **a per-model adapter for the `omnidocbench-amd` platform** — its README hero links to `OmniDocBench-AMD`, its `pyproject.toml` hard-depends `omnidocbench-amd`, and scoring / provenance / CDM are **engine-owned** ("the adapter ends at `.md` + `_run_stats.json`"). This is the opposite posture from the sibling reference repo HunyuanOCR-ROCm, which is a **standalone, evaluation-backed port of the upstream model** that owns its own scoring, lock file, methodology, and CLI.

This spec rewrites MinerU-ROCm into that same standalone posture: **"Evaluation-backed AMD ROCm port of [opendatalab/MinerU](https://github.com/opendatalab/MinerU) — runs the MinerU 3.4 pipeline and MinerU2.5-Pro VLM on AMD gfx1100 (RDNA3) and reports OmniDocBench v1.6 results."** The upstream model repo is the identity; `omnidocbench-amd` becomes one **optional** consumer, not the definition.

Four decisions were locked in the design review:

1. **Decouple depth — full standalone** (mirror HunyuanOCR-ROCm exactly; port its `src/hunyuan_ocr/` blueprint).
2. **Model scope — keep both models, VLM primary** (MinerU2.5-Pro VLM is the hero / OmniDocBench #1; MinerU 3.4 pipeline is the secondary card).
3. **Results strategy — re-run both full-set evals + re-verify the official anchor + byte-exact SHAs into the lock** (the current "official 95.75" anchor is unverified; upstream points to ~95.69).
4. **Landing mode — incremental phased on `main`** (P0→P4, five phases, each a reviewable, green PR; the engine integration is demoted to an optional shim, not deleted).

---

## 1. Background — the quality gap (diagnosis)

A side-by-side audit of MinerU-ROCm vs the reference HunyuanOCR-ROCm (local clones + GitHub behavior, README/spec/repro/packaging/CI/tests/license/history) surfaced one **root** gap and a tiered list of consequences.

### 1.1 The root gap — identity misalignment

| | HunyuanOCR-ROCm (reference) | MinerU-ROCm (current) |
|---|---|---|
| First sentence | *"Evaluation-backed AMD ROCm port of **HunyuanOCR-1.5**…"* — hero links to `Tencent-Hunyuan/HunyuanOCR` | *"A per-model adapter repo for the **omnidocbench-amd** platform. Rendered from the cookiecutter template"* — hero links to `OmniDocBench-AMD` |
| Core runtime deps | GPU-free core (pillow/pyyaml/tqdm/requests); **no platform dep** | `dependencies = ["omnidocbench-amd>=0.1.0"]` — **hard dep** on the platform engine |
| Who owns scoring/provenance | **This repo** (`scoring.py`, `validation.py`, `reproducibility.lock.yaml`, CLI) | **The engine** (spec §5.2: "Adapter ends at `.md` + `_run_stats.json`. Eval/scoring/CDM/provenance are engine-owned") |
| Badge identity | maintainer-verified on gfx1100/ROCm 7.2 | platform `hub/registry.yaml` badge (lives in a **separate** repo) |

### 1.2 Tiered gap list

**Tier 1 — trust-defining (must close to match the reference):**
1. **No `reproducibility.lock.yaml`.** Hunyuan's lock is a single source of truth — commits + GT/weight SHA256 cross-checked **byte-for-byte** against the official HF repos + env + metric formula; README tables auto-render from it; CI fails on drift. MinerU's provenance is scattered across engine-generated `provenance.json` + hand-written README tables; no single source, no weight cross-check.
2. **No `benchmark-methodology.md`.** The "evaluation-backed vs precision-aligned" framing lives only in the 2026-07-17 spec §4, not as a user-facing doc; no "what is/isn't comparable" discipline; no formal/diagnostic/invalid classification (Hunyuan explicitly excludes the invalid vLLM 46.31). **Side finding:** the README's "official 95.75" anchor is unverified — upstream sources point to ~95.69; this is exactly the provenance lapse Hunyuan caught with 94.74-vs-94.10.
3. **License declaration is likely wrong.** `pyproject` declares `Apache-2.0`, but `opendatalab/MinerU` upstream is **AGPL-3.0**; the 2026-07-17 spec §16.9 itself flags "confirm upstream license compatibility before distributing." Hunyuan handles this meticulously (mixed-license `NOTICE` + `LICENSES/` + SPDX `REUSE.toml`).

**Tier 2 — engineering maturity:**
4. **Packaging** — no `src/` package, no wheel, no CLI (Hunyuan: `hunyuan-ocr` CLI + `dist/` wheel + `SHA256SUMS`, GPU-free core, `v0.1.3`).
5. **Reproduce scripts / canary** — no `scripts/` dir, no `reproduce_*.sh`, no canary manifest; reproduction is copy-paste bash in `docs/reproducibility.md` full of **machine-local paths** (`/root/ocr-eval/…`, `/opt/venv`) — precisely what Hunyuan's HANDOFF warns against.
6. **Releases / version / citation** — 0 git tags, 26 commits, no `CHANGELOG` / `CITATION.cff` (Hunyuan: v0.1.0/v0.1.2/v0.1.3 tags + CHANGELOG + CITATION + `release-artifact.md`).
7. **CI depth** — has `ci.yml` (good) but no GPU-CI bridge, no dependabot/CODEOWNERS/issue templates (Hunyuan: `gpu-smoke` commit-status bridge + 4 issue templates + dependabot).
8. **Testing** — 3 files / 128 lines vs Hunyuan's 10 files (incl. `check_repo`, `score_gate`, `scoring`, `cli`).

**Tier 3 — polish:**
9. **Docs depth** — missing `architecture` / `hardware-matrix` / `release-artifact` / `release-checklist`; `known-gaps` is 3 bullets.
10. **Upstream engagement** — filed MIGraphX#5078 (good) but **zero feedback to `opendatalab/MinerU`**; Hunyuan filed ROCm#6416 + Tencent#114 (three follow-ups).
11. **Hardware honesty matrix** — uses platform badges; lacks Hunyuan's ✅/❌/❔ "do not assume VRAM" per-GPU matrix.

---

## 2. Decisions (locked in design review)

| # | Decision | Choice | Rationale |
|---|---|---|---|
| D1 | Decouple depth | **Full standalone** (mirror Hunyuan) | Most faithful to "link to the original repo, not omnidocbench-amd"; Hunyuan's `src/hunyuan_ocr/` is a ready blueprint, so the cost is bounded. |
| D2 | Model scope | **Both models, VLM primary** | Both are official MinerU products; VLM (95.56, OmniDocBench #1) is the headline; pipeline (86.48) is the secondary card. Preserves existing intent. |
| D3 | Results strategy | **Re-run both full-set + verify anchor + byte-exact SHAs** | Closes the resume-`ok_pages` fragility structurally; matches Hunyuan's "scored twice" bar; environment has 4× idle gfx1100 + ready venvs/weights. |
| D4 | Landing mode | **Incremental phased on `main`** | Preserves git history + currently-working results; each phase reviewable + green; engine integration demoted to optional shim (not deleted). Matches how HunyuanOCR-ROCm was actually built. |

**Out of scope (non-goals):**
- Bit-exact parity with upstream CUDA (impossible — no same-engine CUDA control; official engine may differ).
- Re-implementing MinerU's models (we wrap upstream `mineru[all]` + `mineru-vl-utils` + upstream weights).
- Windows verification in this environment (handoff only — retained from the 2026-07-17 spec §14).

---

## 3. Design

### 3.1 Identity & repo posture

- **README hero/tagline** rewritten to Hunyuan's posture: *"Evaluation-backed AMD ROCm port of [MinerU](https://github.com/opendatalab/MinerU) — runs the MinerU 3.4 pipeline and MinerU2.5-Pro VLM on AMD gfx1100 (RDNA3) and reports OmniDocBench v1.6 results. **Not** a precision-aligned port: … See Benchmark methodology."* First link = `opendatalab/MinerU`. Add badges (OmniDocBench v1.6 / status: evaluation-backed / license: mixed / VLM 95.56 / pipeline 86.48) and an "At a glance" block (what / where-verified / most-reliable-result / most-important-limitation / fastest-path).
- **`omnidocbench-amd` demoted to optional:** remove from core `dependencies`; move to `optional-dependencies.platform = ["omnidocbench-amd>=0.1.0"]`. Replace the `omnidocbench_amd.types` import with a local `src/mineru_rocm/types.py`. Keep the existing dispatcher as a thin **optional shim** (`src/mineru_rocm/platform_shim.py`, `[platform]` extra) — clearly labeled "optional omnidocbench-amd platform integration," not a core path.
- **`how-it-works.md`** leads with: *"benchmark infrastructure for evaluating MinerU on AMD ROCm — not a model port."*
- **Directory restructure** to mirror Hunyuan: new `src/mineru_rocm/` (GPU-free-core package), `scripts/`, `eval/canary_*.manifest.json`, `reproducibility.lock.yaml`, `reports/`, `LICENSES/`, `NOTICE`, `CHANGELOG.md`, `CITATION.cff`; `results/` reorganized to lock-driven `<pageset>/<backend>/` layout; `adapter/` absorbed/demoted.

### 3.2 Standalone package + CLI + scoring architecture + data flow

Module responsibilities (porting Hunyuan's `src/hunyuan_ocr/` blueprint):

| Module | Responsibility | Source |
|---|---|---|
| `contract.py` | Frozen inference contract: pipeline config (`MINERU_DEVICE_MODE=cuda`, formula-CH off, ONNX-table policy) + VLM two-step params (`MinerULogitsProcessor` no-repeat-100, OTSL→HTML, server flags) | New (locks upstream recipe) |
| `tasks.py` / `postprocess.py` | Two-step type-specific prompts + OTSL→HTML conversion | **Verbatim port from `mineru_vl_utils`** (AGPL — §3.5) |
| `backends/pipeline.py` | Wraps upstream `mineru[all]` in-process on cuda → md | Refactor of current `pipeline_adapter.py` |
| `backends/vlm_vllm.py` | Drives `mineru-vl-utils` two-step → vLLM-on-ROCm server | Refactor of current `vlm_adapter.py` |
| `backends/vlm_transformers.py` | transformers fallback (logits processor ported into `generate`) | Current |
| `endpoint_pool.py` | Circuit-breaking OpenAI endpoint pool (VLM http-client path) | Port from Hunyuan |
| `preflight.py` | Fail-fast input validation + sharding, before model load | Port from Hunyuan |
| `runner.py` | Atomic writes, resumability, run-manifest schema (conservation laws), writer lock | **Port from Hunyuan — structurally fixes the resume `ok_pages` fragility** |
| `validation.py` | Pre-score prediction-dir validation (missing/empty/ERROR/.partial) | Port from Hunyuan |
| `scoring.py` | OmniDocBench eval-config writer + scorer + result parser; private temp config dir | **Port from Hunyuan — scoring reclaimed into this repo** |
| `omnidocbench.py` | Dataset iteration + prediction filename mapping | Port from Hunyuan |
| `canary.py` | Rebuild the canary byte-identically from full GT | Port from Hunyuan (new for MinerU) |
| `cli.py` | Unified CLI | Port from Hunyuan |
| `types.py` | `RunSummary` / `PageStatus` (replaces `omnidocbench_amd.types`) | Port from engine / Hunyuan `runner` |
| `platform_shim.py` | Optional: omnidocbench-amd engine contract (`[platform]` extra) | Demoted current dispatcher |

**CLI surface** (mirrors `hunyuan-ocr`):
```
mineru-rocm doctor [--strict --backend {pipeline|vlm-vllm|vlm-transformers} --json]
mineru-rocm predict  --backend {pipeline|vlm-vllm|vlm-transformers} ...
mineru-rocm validate ...
mineru-rocm score    ...
mineru-rocm canary materialize ...
mineru-rocm manifest verify ...
```

**Data flow** (backend-agnostic downstream shared, isomorphic to Hunyuan's three-backend diagram):
```
            OmniDocBench v1.6 (GT + images)
                        |
                        v
          Evaluation Driver (mineru-rocm predict / scripts/run_inference.py)
                        |
      +-----------------+------------------+------------------+
      |                 |                  |
  pipeline          vlm-vllm         vlm-transformers
 (mineru[all]      (mineru-vl-utils   (mineru-vl-utils
  in-proc cuda)     → vLLM-on-ROCm)    → transformers)
      |                 |                  |
      +-----------------+------------------+
                        |
                        v
        Prediction artifacts (<stem>.md/page, atomic, resumable, .run.lock'd)
                        |
                        v
              Validator (mineru-rocm validate)
                        |
                        v
          Scorer (mineru-rocm score; OmniDocBench v1.6 scorer venv)
                        |
                        v
     Reproducibility Manifest (run_manifest.json + reproducibility.lock.yaml)
```

**Design principles** (mirror Hunyuan; principle 2 is the structural fix):
1. **CPU-installable core** — `pip install mineru-rocm` runs on plain CPU; GPU deps (torch / vllm / `mineru[all]`) are opt-in extras; CLI/validation/scoring-wrapper never import torch.
2. **Atomic + resumable** — one `<stem>.md` per page written atomically; resumability skips only genuinely-complete pages; `.run.lock` prevents two writers. **This structurally eliminates** the current VLM run's `ok_pages` regeneration-from-disk-truth (incremental per-page stats survive resume; the manifest is authoritative, not reconstructed).
3. **One source of truth** — `reproducibility.lock.yaml`; `check_repo.py` cross-checks README ↔ lock; CI fails on drift.
4. **Evaluation-backed, not precision-aligned.**

**Error handling** (carry over the existing R2 contract): per-page failure → record `PageStatus`, continue, never raise (a missing page scores zero); VLM server-down → all pages recorded failed, never raise; `smoke` backend retained for no-GPU CI.

### 3.3 reproducibility.lock + benchmark methodology + canary

**`reproducibility.lock.yaml`** — single source of truth; every field annotated `# (verified)` / `# (not_recorded)` + a fill command; never invent values. Skeleton (adapted to MinerU's two models):

```yaml
mineru_rocm: { repo, commit }                                   # published-results anchor
mineru:        { repo: opendatalab/MinerU, commit, version: 3.4.4 }            # (verified) pipeline upstream
mineru_vl_utils: { repo: opendatalab/mineru-vl-utils, commit, version }        # (verified) VLM two-step upstream
model:
  vlm:                                                          # MinerU2.5-Pro-2605-1.2B
    hf_repo: opendatalab/MinerU2.5-Pro-2605-1.2B
    benchmark_artifact: { safetensors_sha256, config_sha256 }                   # (verified) local sha256sum
    current_remote_artifact: { hf_repo_revision, lfs_oid_*, cross_check_source: hf-mirror.com }  # (verified) byte-for-byte vs upstream
  pipeline_weights:                                             # opendatalab/PDF-Extract-Kit-1.0 sub-models
    { layout, formula, ocr, table }_sha256                                       # (verified)
omnidocbench:
  version: v1.6
  scorer_repo_url + scorer_commit                                               # (verified)
  gt_json_full_sha256 / gt_json_canary_sha256                                   # (verified)
  eval_config_sha256 / canary_manifest_sha256                                   # (verified)
  metric: { overall_formula, match_method: quick_match, aggregation: page.ALL, note }
environment: { python, rocm_hip, torch, vllm, transformers, mineru, mineru_vl_utils, gpu_arch: gfx1100, rocm_smi_device_id }
benchmark:
  date, hardware
  canary_N:   { pipeline_overall, vlm_vllm_overall, vlm_transformers_overall }
  full_1651:  { pipeline_overall: 86.48, vlm_vllm_overall: 95.56, vlm_transformers: invalid|not_run }
  official_reference: { source, vlm_overall, pipeline_overall, inference_engine: <unspecified|...>, provenance_note }
```

**README results auto-generated from the lock** — `<!-- BEGIN GENERATED RESULTS -->` blocks rendered by `scripts/render_benchmark_tables.py`; `check_repo.py` fails CI on drift. **No hand-edited result numbers in the README** (both current tables are hand-written).

**Official anchor re-verification** (closes the 95.69-vs-95.75 lapse, Hunyuan's 94.74-vs-94.10 discipline):
1. Fetch `opendatalab/OmniDocBench` README/leaderboard at the pinned scorer commit → MinerU2.5-Pro Overall + submetrics + stated engine.
2. Cross-check the MinerU2.5-Pro paper (arXiv:2604.04771).
3. Cross-check `opendatalab/MinerU` README benchmark table.
4. Record the **authoritative** number in `lock.official_reference` (source URL); flag 95.75 as `not_verified` if absent from official sources. Same for pipeline 86.47 (submetrics not published upstream → note).

**`benchmark-methodology.md`** (new, port Hunyuan's structure): (1) what "precision-aligned" would require and why we cannot claim it; (2) result tables never mixed (canary vs full-set, pipeline vs VLM — never one column); (3) scoring pipeline fully reproducible + Overall formula + **`page.ALL` aggregation convention** (MinerU already hit the `.all` vs `.page.ALL` discrepancy — codify it); (4) performance numbers are diagnostic, not ranked; (5) provenance table per formal number. MinerU-specific honesty: the official engine is unspecified (upstream serves via vLLM/LMDeploy), so "全量对齐 = reproduce on gfx1100 full-set with provenance naming our engine"; two engines reported with explicit deltas.

**Canary (new for MinerU):** a fixed ~150-page subset stratified across the 10 doc types, materialized byte-identically from full GT via `mineru-rocm canary materialize`; manifest `eval/canary_N.manifest.json` (with source-GT SHA256) committed. Purpose: (a) minute-scale pre-full-set sanity for both models; (b) same-page comparison across pipeline / vlm-vllm / vlm-transformers (Hunyuan Table A analog); (c) CI GPU-smoke. `scripts/create_canary_manifest.py` regenerates; `check_repo.py` verifies manifest SHA.

**Reproduce scripts:** `scripts/reproduce_{pipeline_full, vlm_vllm_full, canary}.sh` — locked commit, bind `127.0.0.1`, `RESUME`/`OVERWRITE` env vars, absolute paths derived from env vars only (no machine-local paths in user-facing repro).

### 3.4 Results & re-run plan + provenance + gate

**Re-run plan (D3 — fresh artifacts on 4× idle gfx1100):**
- **Pipeline (MinerU 3.4, ~2.9 h / GPU):** fresh `mineru-rocm-venv` / `mineru 3.4.4` / ROCm torch, `HIP_VISIBLE_DEVICES` pinned → `mineru-rocm predict --backend pipeline` (atomic, resumable via new `runner.py`) → `validate` → `score`. Target: reproduce within **±1.0 pp** of the verified official anchor (was 86.48).
- **VLM vLLM (MinerU2.5-Pro, ~4.5 h / GPU):** `examples/serve_vlm_vllm.sh` (vLLM-on-ROCm, `MinerULogitsProcessor`, `--enforce-eager`) → `mineru-rocm predict --backend vlm-vllm --skip-existing` (new `runner.py` → clean structured per-page stats surviving resume; empty-page rate monitored <2%) → `validate` → `score`. Target: within **±0.5 pp** of the verified official anchor (was 95.56).
- **VLM transformers:** sample-only (clean but ~44 h full not run) → reported as diagnostic, no full Overall (Hunyuan transformers "~40 h impractical" analog).

**Structural resume fix:** the current VLM run's `ok_pages` was regenerated from disk truth (1074 → 1651) across a resume; the ported `runner.py` makes per-page stats incrementally persistent and resume-surviving, with the manifest authoritative under conservation laws (count/ok/fail reconcile). Recorded as a dedicated note in the new `run_manifest.json` + HANDOFF.

**Artifact layout (Hunyuan-style, lock-driven):**
```
results/omnidocbench/v1.6/
  pipeline/         full-1651: run_manifest.json, metric_result.json, _run_stats.json, sample_predictions/
  vlm-vllm/         full-1651: same
  vlm-transformers/ sample-only: sample_predictions/ (no full Overall)
results/canary/{pipeline, vlm-vllm, vlm-transformers}/   canary metric_result.json
reports/{HANDOFF, canary-baseline, project-stage-summary}.md
reproducibility.lock.yaml   # the anchor
```
1651 raw `.md` stay out of the repo (too heavy); committed: run_manifest / metric_result / run_stats / 10-page sample + full SHA set.

**Evidence bundle per formal run** (mirror `release-artifact.md`): `run_manifest.json` + `metric_result.json` + `environment.json` + lock ref + `commands.txt` + `checksums.sha256`.

**Gate / tolerance (with the verified anchor):** VLM ±0.5 pp, pipeline ±1.0 pp; honest PASS/FAIL. Example: if the verified anchor is 95.69 and we score 95.56 → Δ −0.13 pp → PASS. **"Scored twice" discipline** (Hunyuan bar): re-score each full run a second time, record both. **formal / diagnostic / invalid** classification (invalid = ERROR pages above threshold; analog to Hunyuan excluding vLLM 46.31).

**Self-attestation → maintainer-verified:** adopt Hunyuan's posture — results "maintainer-verified on gfx1100/ROCm 7.2" with the evidence bundle; the platform badge (if `[platform]` is kept) becomes a downstream consequence, not the definition.

### 3.5 License + release + CI + testing + upstream engagement

**License correction (Tier-1 #3):**
- **Verify first** (P0) the upstream license of each component: MinerU (AGPL-3.0) / `mineru-vl-utils` / MinerU2.5-Pro weights / PDF-Extract-Kit-1.0 weights — confirm against upstream, do not assert.
- `pyproject.license` → mixed declaration: `Mixed: AGPL-3.0 (code ported from MinerU/mineru-vl-utils + pipeline-wrapped builds) AND Apache-2.0 (original packaging/tooling). Weights under <verified>. See NOTICE and LICENSES/.`
- `NOTICE` (full breakdown) + `LICENSES/{Apache-2.0.txt, LicenseRef-MinerU-AGPL.txt, <weight-license>}` + `REUSE.toml` + CI `reuse lint`. README license section: read-before-download; AGPL network-use disclosure; upstream non-affiliation disclaimer.

**Release / version / CHANGELOG / CITATION:** cut **v1.0.0** on completion; add `CHANGELOG.md`, `CITATION.cff` (cite this repo + MinerU2.5-Pro paper + MinerU), `dist/` wheel + `SHA256SUMS`, `docs/{release-artifact,release-checklist}.md`, git tags.

**CI deepening (mirror Hunyuan):**
- `ci.yml`: pytest (CPU, no torch) + `ruff check` + `scripts/check_repo.py` + `reuse lint`.
- **`gpu-smoke.yml` — GPU-CI bridge** (commit-status based, Hunyuan `poller.py` pattern): per-commit canary-on-a-few-pages on real gfx1100 — real GPU validation, not just CPU conformance.
- `rocm-runner-preflight.yml`.
- `.github/`: dependabot, CODEOWNERS, PR template, 4 issue templates (benchmark_report / bug_report / rocm_compatibility / config).

**Testing depth (3 files → 10, mirror Hunyuan):** add `test_check_repo` (lock/README consistency), `test_scoring` + `test_score_gate`, `test_omnidocbench`, `test_postprocess` + `test_tasks` (OTSL→HTML + prompts, upstream-ported, verbatim-trackable with per-file ruff ignores), `test_cli_extras`, `test_{vlm,pipeline}_wiring` (no GPU); `gpu` marker for end-to-end (CI-deselected). Principle: behavior change → CPU unit test; score-affecting change → re-baseline.

**Upstream engagement (Tier-3 #10 — currently absent):**
- Draft an issue/PR to upstream `docs/zh/usage/acceleration_cards/AMD.md`: the verified full-set OmniDocBench numbers on gfx1100 (VLM 95.56 / pipeline 86.48), the locked-commit recipe, the resume/runner lessons, the ONNX-tables-on-CPU finding.
- `docs/upstream-issue-drafts/` (mirror Hunyuan `docs/tencent-*-draft.md`).
- README "Issues filed" section (MIGraphX#5078 + the new MinerU upstream issue).

**Governance files:** MinerU already has CODE_OF_CONDUCT / CONTRIBUTING / LICENSE / Makefile; add SECURITY, SUPPORT, NOTICE, REUSE.toml, CHANGELOG, CITATION, dependabot, CODEOWNERS, issue/PR templates (all mirror Hunyuan).

---

## 4. Phased implementation plan (high level — detailed via writing-plans)

Each phase lands green on `main` as a reviewable PR; the repo stays usable throughout.

- **P0 — Identity + license + lock skeleton (no behavior change):** rewrite README hero/tagline + "At a glance"; demote `omnidocbench-amd` to `[platform]` extra (core deps GPU-free); verify upstream licenses → mixed `NOTICE`/`LICENSES`/`REUSE.toml`; add `reproducibility.lock.yaml` skeleton (fields present, values `not_recorded` + fill commands); add `CHANGELOG`/`CITATION`/`SECURITY`/`SUPPORT`/governance files. *No score changes; current results stand.*
- **P1 — Standalone package + CLI + scoring/validation port:** create `src/mineru_rocm/` porting `runner`/`validation`/`scoring`/`omnidocbench`/`endpoint_pool`/`preflight`/`cli`/`types` from Hunyuan; refactor `pipeline_adapter`/`vlm_adapter` into `backends/`; add the `mineru-rocm` CLI; move the dispatcher to `platform_shim.py` (`[platform]` extra). CPU tests + `check_repo.py` green.
- **P2 — Methodology + architecture + hardware-matrix docs:** write `benchmark-methodology.md` (with the `page.ALL` convention codified), `architecture.md`, `hardware-matrix.md` (✅/❌/❔), expand `known-gaps.md`.
- **P3 — Re-run + populate lock + canary:** re-verify the official anchor (95.69 vs 95.75); build the canary manifest; re-run pipeline + VLM-vLLM full-set with the new `runner.py` (structural resume fix) → fresh artifacts → score twice → populate `reproducibility.lock.yaml` with verified SHAs; render README results from the lock.
- **P4 — Release + CI + upstream:** cut **v1.0.0** (wheel + `SHA256SUMS` + tags + CHANGELOG); add `gpu-smoke.yml` GPU-CI bridge + `ci.yml` (`check_repo`/`ruff`/`reuse lint`); deepen tests; draft + file the upstream `opendatalab/MinerU` AMD.md contribution; add "Issues filed" section.

P0 and P2 are doc/structure-only and CPU-only (no GPU). P1 is CPU-only. P3 is the GPU-heavy phase (≈8 h GPU: pipeline ~2.9 h + VLM ~4.5 h + scoring ×2). P4 is CPU + one GPU smoke.

---

## 5. Success criteria (definition of done)

- [ ] README hero links to `opendatalab/MinerU`; `omnidocbench-amd` is an optional `[platform]` extra; core deps GPU-free.
- [ ] `src/mineru_rocm/` importable package + `mineru-rocm` CLI (`doctor`/`predict`/`validate`/`score`/`canary materialize`/`manifest verify`); scoring/validation owned in-repo.
- [ ] `reproducibility.lock.yaml` populated with **verified** values (commits + byte-exact weight/GT SHAs cross-checked vs upstream HF + env + metric formula); README results auto-rendered from it; `check_repo.py` green.
- [ ] `benchmark-methodology.md` present (precision-aligned caveat, never-mixed tables, `page.ALL` convention, provenance table).
- [ ] Both models re-run on gfx1100 full-set; results within tolerance of the **verified** official anchor (VLM ±0.5 pp, pipeline ±1.0 pp); scored twice; formal/diagnostic/invalid classified.
- [ ] License correctly declared (mixed: AGPL-3.0 + Apache-2.0 + weight license); `NOTICE`/`LICENSES`/`REUSE.toml`; `reuse lint` green.
- [ ] Canary introduced; CI green (`ci.yml` + `gpu-smoke.yml`); ≥10 test files; v1.0.0 released (tag + wheel + SHA256SUMS + CHANGELOG + CITATION).
- [ ] Upstream `opendatalab/MinerU` engagement filed; "Issues filed" section in README.

---

## 6. Risks & open questions

1. **Upstream license exact text** — must verify `mineru-vl-utils`, MinerU2.5-Pro weights, and PDF-Extract-Kit-1.0 weight licenses before finalizing `NOTICE` (P0). If any is more restrictive than AGPL-3.0, the mixed-license breakdown adjusts.
2. **Official anchor** — the verified number may be 95.69 (not 95.75); the gate delta and the "PASS at +0.31 pp" claim in the current README will both be restated. If the authoritative source itself disagrees across OmniDocBench leaderboard vs paper vs MinerU README, the lock records the primary source and flags the others.
3. **Porting fidelity from Hunyuan** — the Hunyuan blueprint is single-model-three-server-backends; MinerU is two-models with an in-process pipeline backend + http-client VLM. The `runner`/`scoring`/`validation` ports are model-agnostic and carry over cleanly; the `predict` driver must handle both in-process (pipeline) and server (vlm-*) modes (richer than Hunyuan, same shape).
4. **GPU-CI bridge** — Hunyuan's `gpu-smoke` relies on a self-hosted gfx1100 runner + commit-status plumbing; this env has 4× idle gfx1100, so the bridge is reproducible, but the exact runner topology must be confirmed (P4).
5. **Push caveat** — git-push of updates from this environment has historically failed (gh API works as a fallback); commits land locally and push via the documented fallback. Non-blocking for design/plan; flagged for the release phase.

---

## 7. References

- Reference repo (quality bar): [AIwork4me/HunyuanOCR-ROCm](https://github.com/AIwork4me/HunyuanOCR-ROCm) — `src/hunyuan_ocr/` blueprint, `reproducibility.lock.yaml`, `docs/benchmark-methodology.md`, `docs/architecture.md`, `docs/release-artifact.md`, GPU-CI bridge.
- Upstream model: [opendatalab/MinerU](https://github.com/opendatalab/MinerU) v3.4.4; [opendatalab/mineru-vl-utils](https://github.com/opendatalab/mineru-vl-utils) (`two_step_extract`, `MinerULogitsProcessor`, `otsl2html.py`); [HF opendatalab/MinerU2.5-Pro-2605-1.2B](https://huggingface.co/opendatalab/MinerU2.5-Pro-2605-1.2B); pipeline weights [opendatalab/PDF-Extract-Kit-1.0](https://huggingface.co/opendatalab/PDF-Extract-Kit-1.2); ROCm guide `docs/zh/usage/acceleration_cards/AMD.md` + Discussion #3662 + fork [healy-hub/MinerU-AMD-RDNA](https://github.com/healy-hub/MinerU-AMD-RDNA).
- Benchmark: [opendatalab/OmniDocBench](https://github.com/opendatalab/OmniDocBench) v1.6 (1651 pages); MinerU2.5-Pro paper [arXiv:2604.04771](https://arxiv.org/abs/2604.04771).
- Prior spec (superseded on framing, retained on technical content): [`2026-07-17-mineru-rocm-design.md`](2026-07-17-mineru-rocm-design.md).
- Sibling: [AIwork4me/Unlimited-OCR-ROCm](https://github.com/AIwork4me/Unlimited-OCR-ROCm).
