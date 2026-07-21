# Windows-HIP evaluation handoff — revision design

**Date:** 2026-07-21
**Status:** Approved (design).
**Target repo / file:** `MinerU-ROCm` `docs/HANDOFF-windows-hip.md` (rewritten in place).
**Branch:** `docs/windows-hip-handoff-revision` (from `main` @ `4f44ae4`).
**Goal:** a colleague on a Ryzen AI MAX+ 395 (Strix Halo) / Windows 11 box can open this one document and **start verification immediately** — produce a real OmniDocBench v1.6 score for both the MinerU 3.4 pipeline and the MinerU2.5-Pro VLM on `windows-hip`, then land a self-contained platform bundle.

---

## 1. Why the current handoff blocks the colleague (audit, verified)

| # | Defect | Evidence |
|---|--------|----------|
| F1 | **Scoring is impossible as written.** §6 runs `omnidocbench-rocm score --platform windows-hip` + `cdm setup --platform windows-hip`, but the platform's `get_backend("windows-hip")` raises `NotImplementedError` (only `linux_rocm.py` exists). | `OmniDocBench-ROCm/engine/omnidocbench_rocm/backends/__init__.py:24-25` |
| F2 | **Overall formula is wrong.** §6: `((1−Text_EditDist)×100 + Table_TEDS + Formula_CDM)/3` — TEDS/CDM miss `×100`. Raw metric stores them 0–1 (pipeline TEDS≈0.82, CDM≈0.83) → formula yields ≈32, not 86. Colleague would think the run failed. | compare `reproducibility.lock.yaml` `metric.overall_formula` + `bundle_validator.recompute_overall` |
| F3 | **Stale path.** Uses `results/omnidocbench/v1.6/windows-hip/`; current platform layout is `results/omnidocbench/v16/<platform>/`. | `results/omnidocbench/v16/linux-rocm/` (the P1.1 bundles) |
| F4 | **Stale revision.** `--revision v1.6`; the pinned dataset+scorer revision is `2b161d0`. | `reproducibility.lock.yaml` `omnidocbench.scorer_commit` / platform `_refs.OMNIDOCBENCH_V16_REF` |
| F5 | **Stale platform install.** `pip install omnidocbench-rocm` pulls pre-0.3.1 from PyPI, which lacks the self-contained `publish` + new flags (`--backend`, `--metric-result`, `--prediction-source-*`, `--gt-sha256`). | `OmniDocBench-ROCm` 0.3.1 just merged to main (PR #15 → `ce081db`); not on PyPI yet. |
| F6 | **Nonexistent script.** §7.3 `python scripts\check_conformance.py .` — that file does not exist in MinerU-ROCm (it is a platform script). | `ls scripts/` (MinerU) |
| F7 | **VLM on Windows has no concrete path.** §10 lists transformers-DirectML / llama.cpp-GGUF as "future"; the adapter's vlm backend assumes a vLLM HTTP server (Linux-only). Not actionable today. | `src/mineru_rocm/backends/vlm.py` (http client) |

---

## 2. The scoring solution: `omnidocbench-amd-windows`

Do **not** force `mineru-rocm score` and do **not** use the broken `omnidocbench-rocm score --platform windows-hip`. Use the sibling repo **`AIwork4me/omnidocbench-amd-windows`** — a one-command OmniDocBench v1.6 full-evaluation stack for Windows + AMD (ROCm/HIP), **CDM formula scoring included**, model-agnostic via adapters.

Why it fits (verified from its README + architecture):
- Same adapter contract as `omnidocbench-rocm` — an adapter writes one `<image_stem>.md` per page into a predictions dir. **MinerU-ROCm's existing `adapter/run_adapter.py` already satisfies this and already writes `_run_stats.json`**, so it drops in.
- It runs OmniDocBench's `pdf_validation.py`, so its `metric_result.json` is **the same shape** `omnidocbench-rocm`'s `validate-bundle` recomputes Overall from.
- It provisions the CDM toolchain two ways — **native Windows** (TeX Live + ImageMagick 7 + Ghostscript via `patches/omnidocbench/windows-cdm.patch`) or a **WSL reference path** — so "full CDM from the start" is achievable without the colleague hand-provisioning TeX.
- 4 idempotent phases (`setup.*` + `verify.*` each): `01-omnidocbench` (code + dataset) → `02-cdm-environment` → adapter inference → `03-scoring` (`score.ps1 -Config v16-cdm.yaml` native, or `score-cdm.sh` WSL).
- Per-model plugins live in `adapters/<model>/` (there is an `_template` + a `paddleocr-vl-1.6` reference with `00-install-deps`/`01-vlm-server`/`02-layout-model`/`run_adapter.py`).

---

## 3. Revised handoff structure (Pipeline + VLM, full CDM)

Rewrite `docs/HANDOFF-windows-hip.md` to this section plan:

- **§0 — What you are verifying + honest framing.** Two targets: pipeline Overall ≈ **86.47** (±1.0 pp) and VLM Overall ≈ **95.56 CDM** (±0.5 pp — but framed as "reproduce the linux-rocm number", not precision-aligned: a *different* Windows inference engine may shift VLM by >0.5 pp). Overall = `((1−text_EditDist)·100 + formula_CDM·100 + table_TEDS·100)/3` (note: raw TEDS/CDM are 0–1 fractions). Staged: **pipeline = Phase 1 (solid), VLM = Phase 2 (exploratory)**.

- **§1 — Target environment.** Strix Halo (Radeon RDNA 3.5 iGPU), Windows 11, WSL2 Ubuntu 22.04, Python 3.10/3.11 (not 3.12), PowerShell. ~50 GB disk.

- **§2 — Install.** Two checkouts: (a) `git clone AIwork4me/omnidocbench-amd-windows` (the scorer); (b) `git clone AIwork4me/MinerU-ROCm` at the pinned commit (the model + its adapter + `omnidocbench-rocm` publish). Install `omnidocbench-rocm` ≥0.3.1 pinned to the merged main commit (`ce081db…`) until 0.3.1 ships to PyPI; then `pip install -e .` MinerU-ROCm.

- **§3 — Run the omnidocbench-amd-windows 4 phases.** env+WSL → `01-omnidocbench` (dataset at revision `2b161d0`; note the images dir) → `02-cdm-environment` (native Windows CDM via `verify-windows.ps1`, else the WSL reference path). Each phase has a `verify.*`.

- **§4 — Add the MinerU adapter (the one per-model step).** Create `adapters/mineru/` in the `omnidocbench-amd-windows` checkout, mirroring `_template` + `paddleocr-vl-1.6`. Its `run_adapter.py` delegates to MinerU-ROCm's dispatcher (`python -m mineru_rocm.dispatcher` / the repo's `adapter/run_adapter.py` with `--backend {pipeline,vlm-vllm}`). Because MinerU-ROCm's adapter already emits `<stem>.md` + `_run_stats.json`, the framework scores it directly.

- **§5 — Inference.**
  - **Phase 1 — pipeline:** `python adapters\mineru\run_adapter.py --backend pipeline --img-dir <images> --out-dir predictions\mineru_pipeline`. Target ≈86.47. Validate on a 10-page subset first.
  - **Phase 2 — VLM (exploratory, flagged):** MinerU2.5-Pro needs a Windows serving runtime — **no Windows vLLM-HIP**. Primary attempt: the framework's `01-vlm-server` layer with a **transformers-DirectML** (or **llama.cpp-GGUF**) build of MinerU2.5-Pro; the MinerU adapter's vlm backend must branch on `--platform windows-hip` to call the local Windows server instead of a vLLM HTTP endpoint. **Honest note:** this path is untested here (no Strix Halo host); if it blocks, finish pipeline first and report back.

- **§6 — Score (omnidocbench-amd-windows, full CDM).** Native: `eval-infra\03-scoring\score.ps1 -Config v16-cdm.yaml` (after `02-cdm-environment\verify-windows.ps1` passes). WSL: `wsl … eval-infra/03-scoring/score-cdm.sh`. Output `metric_result.json` (text Edit_dist, reading-order Edit_dist, table TEDS, formula CDM). **Compute Overall with the corrected formula.** PASS gates: pipeline ≥85.47; VLM within ±0.5 pp of 95.56 (report the engine + flag if outside).

- **§7 — Publish the self-contained bundle (omnidocbench-rocm 0.3.1, in MinerU-ROCm).** Bring `metric_result.json` + the predictions dir + `_run_stats.json` back to the MinerU-ROCm checkout and:
  ```
  omnidocbench-rocm publish `
    --model-id <mineru-pipeline|mineru2.5> --platform windows-hip --version v16 --cdm `
    --backend <engine matching _run_stats.json["engine"]> `
    --run-stats <..._run_stats.json> --metric-result <...metric_result.json> `
    --predictions-dir <predictions dir> `
    --results-dir results\omnidocbench\v16\windows-hip `
    --git-commit <sha> --adapter-command "<full windows argv>" `
    --dataset-revision 2b161d0 `
    --gt-sha256 a45cd84b04ad8b793e775089640e6b681209abea33ead54c1828ddca35fae496 `
    --prediction-source-commit <sha> --migration-type legacy_predictions_to_platform_artifacts
  ```
  This needs **no platform backend** (publish only assembles artifacts). Then `omnidocbench-rocm validate-bundle results\omnidocbench\v16\windows-hip --model-card model_card.{pipeline,}.json` → CONFORMANT.

- **§8 — Land artifacts.** Commit the bundle under `results/omnidocbench/v16/windows-hip/`; set `badge.windows-hip → "community"` + hardware (`AMD Ryzen AI MAX+ 395 (Strix Halo)`) in `model_card.json` + `model_card.pipeline.json`; `omnidocbench-rocm conformance .` → CONFORMANT; open a PR / report to the linux-rocm owner to update `hub/registry.yaml`.

- **§9 — What Claude provides vs. what you provide.** (Table; same shape as today, updated.)

- **§10 — Escalation.** Score drift (pipeline >1.0 pp / VLM >0.5 pp): don't publish as community; diff a few table/formula pages vs the linux `metric_result.json`; check CDM provisioning; report. VLM Windows serving blocked: report, finish pipeline, descope VLM to a follow-up.

- **§11 — Future: fold `omnidocbench-amd-windows` into `omnidocbench-rocm` (Phase B, separate spec).** After a real Windows score exists, port `omnidocbench-amd-windows`'s `eval-infra/` (01-omnidocbench, 02-cdm-environment, 03-scoring) into `OmniDocBench-ROCm` as a real `engine/omnidocbench_rocm/backends/windows_hip.py`, so `get_backend("windows-hip")` works and `omnidocbench-rocm run/score --platform <auto>` **auto-detects** the host (Linux → LinuxRocmBackend; Windows → WindowsHipBackend) and launches the matching scorer. Retire the standalone repo; close the F1 `NotImplementedError`. Not in this handoff's scope — but the colleague's repro notes feed it.

---

## 4. Acceptance criteria (the colleague can "start smoothly")

- [ ] §6 no longer references `omnidocbench-rocm score --platform windows-hip` or `cdm setup --platform windows-hip` (F1 fixed).
- [ ] Overall formula everywhere is `((1−text)·100 + CDM·100 + TEDS·100)/3` with the 0–1 fraction note (F2 fixed).
- [ ] All paths are `results/omnidocbench/v16/windows-hip/` (F3); revision `2b161d0` (F4); platform pinned ≥0.3.1 commit (F5); conformance via `omnidocbench-rocm conformance .` (F6).
- [ ] Scoring route is `omnidocbench-amd-windows` (§3–§6), with a concrete `adapters/mineru/` step reusing MinerU-ROCm's adapter.
- [ ] Pipeline = Phase 1 with copy-paste commands; VLM = Phase 2 with the runtime risk flagged and an escalation.
- [ ] Publish section produces a self-contained bundle at `v16/windows-hip/` + a `validate-bundle` CONFORMANT check.
- [ ] No stale 95.46-as-current, no `v1.6/` paths, no `pip install omnidocbench-rocm` unpinned, no `scripts/check_conformance.py` reference.
- [ ] `check_repo` clean on MinerU after the rewrite.

## 5. Out of scope

- Implementing `WindowsHipBackend` itself (Phase B, separate spec after a real score).
- The VLM-on-Windows serving runtime (transformers-DirectML / llama.cpp-GGUF) — documented as the Phase-2 exploratory step; not built here.
- Changing the linux-rocm story (already landed in P1.1).

## 6. Risks

- **VLM on Windows** is the only piece that may not "just work"; the doc stages it so a stuck VLM doesn't block the pipeline start.
- The `adapters/mineru/` bridge assumes MinerU-ROCm's adapter writes `_run_stats.json` in the shape `publish` consumes — it does (verified: `src/mineru_rocm/runner.py` + the linux-rocm bundles), but the colleague should confirm on the 10-page subset first.
- `omnidocbench-amd-windows` is pre-rebrand-named (it's in the platform's `check_brand` forbidden list); Phase B (folding it in) resolves the naming split. The handoff doc itself must not introduce that token into MinerU's user-facing surface beyond citing the repo URL.
