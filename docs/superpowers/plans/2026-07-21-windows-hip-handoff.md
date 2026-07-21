# Windows-HIP handoff revision — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (inline). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite `MinerU-ROCm/docs/HANDOFF-windows-hip.md` so a colleague on Strix Halo/Windows 11 can start Windows-HIP verification immediately — scoring via `omnidocbench-amd-windows`, then landing a self-contained platform bundle.

**Architecture:** Single-file rewrite of the handoff doc to the approved spec (`docs/superpowers/specs/2026-07-21-windows-hip-handoff-design.md`). The doc points the colleague at `omnidocbench-amd-windows` (which already does dataset + CDM + scoring on Windows), adds a MinerU adapter to it, then bridges to `omnidocbench-rocm publish` (0.3.1, no platform backend) for the bundle. Pipeline = Phase 1 (solid), VLM = Phase 2 (exploratory).

**Tech Stack:** Markdown; PowerShell + WSL2 (colleague's box); `omnidocbench-amd-windows` (sibling repo); `omnidocbench-rocm` ≥0.3.1.

## Global Constraints (verbatim from spec)

- Overall formula = `((1−text_EditDist)·100 + formula_CDM·100 + table_TEDS·100)/3`; raw TEDS/CDM are 0–1 fractions; reading-order excluded.
- Bundle path = `results/omnidocbench/v16/windows-hip/` (NOT `v1.6/`).
- Dataset + scorer revision = `2b161d0`.
- Platform install pinned to `omnidocbench-rocm` merged main commit `ce081dbd3848d84cb0622ceee57c8f054845fcf3` until 0.3.1 ships to PyPI; then `>=0.3.1`.
- Conformance via `omnidocbench-rocm conformance .` (NOT the nonexistent `scripts/check_conformance.py`).
- Do NOT cite `omnidocbench-rocm score --platform windows-hip` / `cdm setup --platform windows-hip` — they raise `NotImplementedError` (no windows-hip backend). Score via `omnidocbench-amd-windows`.
- GT sha256 (for publish `--gt-sha256`) = `a45cd84b04ad8b793e775089640e6b681209abea33ead54c1828ddca35fae496`.
- The token `omnidocbench-amd-windows` may appear only as the repo URL citation (it is in the platform `check_brand` forbidden list); keep MinerU's user-facing surface clean.

---

### Task 1: Rewrite `docs/HANDOFF-windows-hip.md`

**Files:**
- Modify (full rewrite): `docs/HANDOFF-windows-hip.md`

**Interfaces:** none (documentation only).

- [ ] **Step 1: Replace the entire file with the content below**

Write this exact content to `docs/HANDOFF-windows-hip.md`:

````markdown
# Windows-HIP verification handoff — MinerU-ROCm (pipeline + VLM)

> **Who does what.** The **Linux / `linux-rocm`** side is verified (community:
> pipeline Overall **86.48**, VLM Overall **95.56** CDM — see `model_card.json` /
> `model_card.pipeline.json` + the self-contained bundles under
> `results/omnidocbench/v16/linux-rocm/`). The **Windows / `windows-hip`** side is
> handed off to a colleague on **Ryzen AI MAX+ 395 (Strix Halo)** / Windows 11.
> This document is self-contained.
>
> Scope: **both** model cards — the MinerU 3.4 **pipeline** (Phase 1, target
> Overall ≈ **86.47**, ±1.0 pp) and the **MinerU2.5-Pro VLM** `mineru2.5`
> (Phase 2, target ≈ **95.56** CDM, ±0.5 pp — but framed as "reproduce the
> linux-rocm number", not precision-aligned: a *different* Windows inference
> engine can shift the VLM by more than 0.5 pp).

---

## 0. What you are verifying + the honest framing

You reproduce OmniDocBench v1.6 on Strix Halo/Windows and produce real artifacts
this repo can publish under `results/omnidocbench/v16/windows-hip/`. The bar is
**evaluation-backed, not bit-exact-CUDA**: same model + same dataset + same
scorer; a small cross-device delta is expected.

**Overall** = `((1 − text_EditDist) × 100 + formula_CDM × 100 + table_TEDS × 100) / 3`,
OmniDocBench `page.ALL` aggregation. **Note:** the raw metric stores TEDS and CDM
as 0–1 *fractions* (e.g. TEDS ≈ 0.82, CDM ≈ 0.83 for the pipeline) — the `× 100`
on each is required, or the number is garbage. Reading-order EditDist is reported
separately and is **not** part of Overall.

**Staged.** Do the **pipeline first** (Phase 1 — CPU-runnable, solid); the **VLM**
is Phase 2 and exploratory on Windows (no Windows vLLM — see §5).

---

## 1. Target environment

- **Hardware:** AMD Ryzen AI MAX+ 395 (Strix Halo) — integrated Radeon (RDNA 3.5).
- **OS:** Windows 11 + **WSL2 (Ubuntu 22.04)**.
- **Python:** 3.10 or 3.11 (**not** 3.12 — OmniDocBench breaks). 3.11 preferred.
- **Disk:** ~50 GB (dataset ~3 GB + weights + TeX Live ~5 GB + IM7 + WSL rootfs).

---

## 2. Install — two checkouts

```powershell
# A. The Windows scoring stack (dataset + CDM + scoring; model-agnostic adapters)
git clone https://github.com/AIwork4me/omnidocbench-amd-windows
cd omnidocbench-amd-windows

# B. This model repo + the platform publish tool (separate dir)
git clone https://github.com/AIwork4me/MinerU-ROCm
cd MinerU-ROCm
git checkout <pinned commit recorded in docs/reproducibility.md>
python -m venv .venv ; .\.venv\Scripts\Activate.ps1
python -m pip install -U pip
# Platform engine (publish/conformance/validate-bundle) — pin to merged main 0.3.1
# until omnidocbench-rocm 0.3.1 ships to PyPI, then use: pip install "omnidocbench-rocm>=0.3.1,<0.4"
pip install "git+https://github.com/AIwork4me/OmniDocBench-ROCm.git@ce081dbd3848d84cb0622ceee57c8f054845fcf3#egg=omnidocbench-rocm"
pip install -e .          # MinerU-ROCm (adapter + pyproject)
pip install -U "mineru[all]"
```

> The platform's `get_backend("windows-hip")` is **not implemented yet** — so
> `omnidocbench-rocm score --platform windows-hip` / `cdm setup --platform
> windows-hip` will raise `NotImplementedError`. That is why scoring goes through
> `omnidocbench-amd-windows` (§3–§6) and only the bundle assembly through
> `omnidocbench-rocm publish` (§7), which needs no backend.

---

## 3. Run the omnidocbench-amd-windows phases (dataset + CDM)

From the `omnidocbench-amd-windows` checkout root (each `setup.*` is idempotent;
run the matching `verify.*` after):

```powershell
# Step 0 — environment + WSL2 + mirrors
powershell -ExecutionPolicy Bypass -File scripts\detect-mirrors.ps1
powershell -ExecutionPolicy Bypass -File scripts\wsl-ensure.ps1

# Step 1 — OmniDocBench code + v1.6 dataset (PIN the revision to 2b161d0)
powershell -ExecutionPolicy Bypass -File eval-infra\01-omnidocbench\setup.ps1
powershell -ExecutionPolicy Bypass -File eval-infra\01-omnidocbench\verify.ps1
```
The dataset revision **must** be `2b161d0` (the pinned dataset + scorer revision
this repo's `reproducibility.lock.yaml` records). Note the images directory path
(`eval-infra\01-omnidocbench\data\images`); you'll pass it to the adapter.

```powershell
# Step 2 — CDM environment (full CDM, up front). Two options:
#   native Windows CDM (TeX Live + ImageMagick 7 + Ghostscript):
powershell -ExecutionPolicy Bypass -File eval-infra\02-cdm-environment\verify-windows.ps1
#   OR the WSL reference path (Ubuntu TeX Live 2026 + IM7 + gs):
wsl -d Ubuntu2204 bash /mnt/c/<path-to-omnidocbench-amd-windows>/eval-infra/02-cdm-environment/setup.sh
wsl -d Ubuntu2204 bash /mnt/c/<path-to-omnidocbench-amd-windows>/eval-infra/02-cdm-environment/verify.sh
```

---

## 4. Add the MinerU adapter (the one per-model step)

`omnidocbench-amd-windows` is model-agnostic: a model plugs in via
`adapters/<model>/run_adapter.py` whose only job is to write one
`<image_stem>.md` per page (+ `_run_stats.json`). MinerU-ROCm's
`adapter/run_adapter.py` already satisfies that contract and already writes
`_run_stats.json`, so create:

```
omnidocbench-amd-windows/adapters/mineru/
  run_adapter.py   # thin shim that calls MinerU-ROCm's dispatcher
  README.md        # how to run (mirror adapters/paddleocr-vl-1.6/README.md)
```

`run_adapter.py` shim (delegates to MinerU-ROCm's installed dispatcher; keeps the
`--img-dir`/`--out-dir`/`--platform`/`--backend` interface the framework expects):

```python
"""omnidocbench-amd-windows adapter for MinerU-ROCm.

Delegates to MinerU-ROCm's dispatcher (src/mineru_rocm/dispatcher.py), which
writes <out-dir>/<image_stem>.md per page + _run_stats.json — the contract this
framework scores. Reuse, don't reimplement."""
import sys
from mineru_rocm.dispatcher import main  # from the MinerU-ROCm install (pip install -e .)
if __name__ == "__main__":
    raise SystemExit(main())
```

Run the adapter on a **10-page subset first** to confirm wiring, then the full
1651-page set (see §5).

---

## 5. Inference

### Phase 1 — pipeline (solid; do this first)

```powershell
python adapters\mineru\run_adapter.py `
  --backend  pipeline `
  --platform windows-hip `
  --img-dir  eval-infra\01-omnidocbench\data\images `
  --out-dir  predictions\mineru_pipeline
```
- `--platform windows-hip` is required (the adapter branches on it; never infer
  from the OS).
- On Windows without a CUDA/ROCm GPU, mineru's PyTorch models run on CPU (correct,
  slower). DirectML for the ONNX table models is an optional speed-up (see
  `omnidocbench-amd-windows/docs`); skip it for the first run.
- Expect `ok ≈ 1651, fail ≈ 0, limit_pages: null` in `_run_stats.json`.

### Phase 2 — VLM `mineru2.5` (exploratory; flagged)

MinerU2.5-Pro has **no Windows vLLM-HIP path**. Two candidate runtimes, both
untested here (no Strix Halo host in this org's Linux eval env):

1. **transformers + `torch-directml`** (primary attempt) — stock
   `Qwen2VLForConditionalGeneration` serving MinerU2.5-Pro-2605-1.2B on the iGPU,
   via the framework's `01-vlm-server` layer (mirror `adapters/paddleocr-vl-1.6/01-vlm-server`).
2. **llama.cpp-GGUF** (fallback) — a Windows HIP/Vulkan llama.cpp build + a GGUF
   of MinerU2.5-Pro.

The MinerU vlm backend currently expects a vLLM HTTP server; for Windows it must
branch on `--platform windows-hip` to call the local Windows server instead, and
report a matching `_run_stats.json["engine"]` (e.g. `vlm-dml`). **If this blocks,
finish Phase 1 (pipeline) and report back** — don't let it stall the start.

---

## 6. Score (omnidocbench-amd-windows, full CDM)

With `_run_stats.json` present (`limit_pages: null`):

```powershell
# Native Windows CDM (after 02-cdm-environment\verify-windows.ps1 passes):
powershell -ExecutionPolicy Bypass -File eval-infra\03-scoring\score.ps1 -Config v16-cdm.yaml
# WSL CDM reference path:
wsl -d Ubuntu2204 bash /mnt/c/<path>/eval-infra/03-scoring/score-cdm.sh
powershell -ExecutionPolicy Bypass -File eval-infra\03-scoring\verify.ps1
```
This emits `metric_result.json` (text Edit_dist, reading-order Edit_dist, table
TEDS, formula CDM) — the same shape `omnidocbench-rocm`'s `validate-bundle`
recomputes Overall from.

**Compute Overall** = `((1 − text_EditDist) × 100 + formula_CDM × 100 + table_TEDS × 100) / 3`.
**PASS** gates: pipeline ≥ **85.47** (within ±1.0 pp of 86.47); VLM within ±0.5 pp
of **95.56** (report the engine; flag if outside).

---

## 7. Publish the self-contained bundle (omnidocbench-rocm 0.3.1)

In the **MinerU-ROCm** checkout, copy the run's `metric_result.json`,
`_run_stats.json`, and the predictions dir over, then assemble the bundle (no
platform backend needed — `publish` only assembles artifacts):

```powershell
omnidocbench-rocm publish `
  --model-id <mineru-pipeline | mineru2.5> `
  --platform windows-hip --version v16 --cdm `
  --backend <the engine your adapter wrote in _run_stats.json, e.g. pipeline | vlm-dml> `
  --run-stats     <..._run_stats.json> `
  --metric-result <...metric_result.json> `
  --predictions-dir <the predictions dir> `
  --results-dir   results\omnidocbench\v16\windows-hip `
  --git-commit    <your commit sha> `
  --adapter-command "python adapters\mineru\run_adapter.py --backend <b> --platform windows-hip --img-dir <images> --out-dir <preds>" `
  --dataset-revision 2b161d0 `
  --gt-sha256 a45cd84b04ad8b793e775089640e6b681209abea33ead54c1828ddca35fae496 `
  --prediction-source-commit <sha that produced the predictions> `
  --migration-type legacy_predictions_to_platform_artifacts

omnidocbench-rocm validate-bundle results\omnidocbench\v16\windows-hip `
  --model-card model_card.json          # or model_card.pipeline.json for the pipeline
```
Expect `CONFORMANT`. Verify the GT sha matches your downloaded `OmniDocBench.json`
(`certutil -hashfile OmniDocBench.json SHA256` on Windows).

---

## 8. Land the artifacts

1. **Commit** the bundle under `results/omnidocbench/v16/windows-hip/` to this
   repo (provenance + `metric_result.json` + `run_stats.json` + prediction
   manifest + dataset identity; bulk `.md` stay gitignored per repo policy).
2. **Update** `model_card.json` (VLM) and/or `model_card.pipeline.json`:
   `badge.windows-hip → "community"`, fill `eval_date` + `overall` + `submetrics`
   + `hardware` (`gpu: "AMD Ryzen AI MAX+ 395 (Strix Halo)"`). (`verified` still
   needs a maintainer Docker reproduction on both platforms.)
3. `omnidocbench-rocm conformance .` → **CONFORMANT**.
4. **Open a PR** (or send `metric_result.json` + commit sha to the linux-rocm
   owner) so `hub/registry.yaml` in `OmniDocBench-ROCm` gets the windows-hip row.

---

## 9. What Claude (Linux side) provides vs. what you provide

| Provided by Claude (already in the repo) | Provided by you (colleague) |
|---|---|
| MinerU-ROCm adapter/dispatcher (writes `.md` + `_run_stats.json`, branches on `--platform`) | The Windows run + real artifacts |
| `model_card{,.pipeline}.json` windows-hip badge slot + hardware | Filled windows-hip badge + result |
| `omnidocbench-rocm publish`/`validate-bundle`/`conformance` (0.3.1) | The `metric_result.json` / Overall |
| This handoff + the linux-rocm reference bundles | Any Windows-specific patch (e.g. DirectML EP) + repro notes |

---

## 10. Escalation

- **Pipeline Overall drifts > 1.0 pp from 86.47** (or VLM > 0.5 pp from 95.56):
  don't publish as community. Diff a few table/formula pages' `.md` against the
  linux `results/omnidocbench/v16/linux-rocm/` metric; check CDM provisioning
  (native vs WSL); confirm the page set / manifest match. Open an issue with both
  `metric_result.json` side by side.
- **VLM Windows serving blocked:** report; finish Phase 1 (pipeline); descope VLM
  to a follow-up. A future spec folds `omnidocbench-amd-windows` into
  `omnidocbench-rocm` as a real auto-detecting `windows-hip` backend (so
  `omnidocbench-rocm score --platform windows-hip` works and the standalone repo
  can retire).
````

- [ ] **Step 2: Verify the fatal fixes are present + stale refs are gone**

Run:
```bash
cd /workspace/MinerU-ROCm
# F2 fixed: correct formula present, wrong formula absent
grep -n "formula_CDM × 100 + table_TEDS × 100" docs/HANDOFF-windows-hip.md   # expect a hit
! grep -n "Table_TEDS + Formula_CDM) / 3" docs/HANDOFF-windows-hip.md         # expect no hit
# F1 fixed: no broken score/cdm-setup on windows-hip as an instruction (only the explanatory note)
! grep -n "omnidocbench-rocm score --platform windows-hip$\|cdm setup --platform windows-hip$" docs/HANDOFF-windows-hip.md
# F3 fixed: v16/windows-hip path present; no v1.6/windows-hip instruction path
grep -n "results\\\\omnidocbench\\\\v16\\\\windows-hip\|results/omnidocbench/v16/windows-hip" docs/HANDOFF-windows-hip.md
# F4 fixed: revision 2b161d0
grep -n "2b161d0" docs/HANDOFF-windows-hip.md
# F5 fixed: pinned platform install
grep -n "ce081dbd3848d84cb0622ceee57c8f054845fcf3" docs/HANDOFF-windows-hip.md
# F6 fixed: no nonexistent script reference
! grep -n "scripts\\\\check_conformance.py\|scripts/check_conformance.py" docs/HANDOFF-windows-hip.md
```
Expected: every `grep` prints ≥1 line; every `! grep` (negated) prints nothing.

- [ ] **Step 3: Commit**

```bash
git add docs/HANDOFF-windows-hip.md
git commit -m "docs(handoff): rewrite Windows-HIP handoff — score via omnidocbench-amd-windows

Fix 2 fatal blockers (scoring step hit platform NotImplementedError; Overall
formula missing x100 on TEDS/CDM) + stale path/revision/platform-version/
script refs. Score on Windows via omnidocbench-amd-windows (CDM included,
model-agnostic adapters; MinerU-ROCm adapter drops in), then assemble the
self-contained bundle with omnidocbench-rocm publish 0.3.1. Pipeline=Phase1,
VLM=Phase2 exploratory.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: Verify no regressions in the repo gate

**Files:** none (verification only).

- [ ] **Step 1: Run the MinerU gate**

```bash
cd /workspace/MinerU-ROCm
python -m pytest -q                       # expect: 156 passed (or current count, 0 failed)
python scripts/check_repo.py              # expect: check_repo: clean
ruff check .                              # expect: All checks passed!
```
Expected: all clean. The handoff doc is under `docs/` and is scanned by
`check_repo.check_no_internal_infra` (no host paths), `check_no_withdrawn_anchor_claims`,
`check_version_consistency` (must state `ROCm 7.2` — the doc says Strix Halo/Windows,
no ROCm version claim, so fine), and `check_readme_lock_values` (only README.md, not
this doc). If `check_repo` flags anything in the handoff, fix the flagged line.

- [ ] **Step 2: Confirm the doc cites the repo only as a URL (brand-cleanliness)**

```bash
# 'omnidocbench-amd-windows' should appear only as the github URL + the cloned dir name,
# not as a product/CLI identity. MinerU's own check_repo has no brand gate, but keep it clean.
grep -c "omnidocbench-amd-windows" docs/HANDOFF-windows-hip.md   # expect a small number (URLs + dir refs)
```

---

### Task 3: Push + open PR

**Files:** none.

- [ ] **Step 1: Push the branch**

```bash
cd /workspace/MinerU-ROCm
git push -u origin docs/windows-hip-handoff-revision
# If the env push-lock blocks the update (cannot lock ref ... reference already exists),
# use the throwaway-ref + gh API workaround:
#   HEAD=$(git rev-parse HEAD)
#   git push origin "$HEAD:refs/heads/throwhandoff"
#   gh api repos/AIwork4me/MinerU-ROCm/git/refs/heads/docs/windows-hip-handoff-revision -X PATCH -f sha="$HEAD"
#   gh api repos/AIwork4me/MinerU-ROCm/git/refs/heads/throwhandoff -X DELETE
```

- [ ] **Step 2: Open the PR**

```bash
gh pr create --base main --head docs/windows-hip-handoff-revision \
  --title "docs(handoff): rewrite Windows-HIP verification handoff" \
  --body "Rewrites docs/HANDOFF-windows-hip.md so a Strix Halo colleague can start Windows-HIP verification immediately. Scores via omnidocbench-amd-windows (CDM included); assembles the self-contained bundle via omnidocbench-rocm publish 0.3.1. Fixes 2 fatal blockers (scoring step hit platform NotImplementedError; Overall formula missing x100) + stale path/revision/platform-version/script refs. Pipeline=Phase1, VLM=Phase2 exploratory. Spec: docs/superpowers/specs/2026-07-21-windows-hip-handoff-design.md."
```

- [ ] **Step 3: Confirm CI green (core job; platform-contract is unaffected by a docs change)**

```bash
gh pr view <PR> --json statusCheckRollup -q '[.statusCheckRollup[] | {name,conclusion}]'
# expect core: SUCCESS
```

---

## Self-Review

**Spec coverage:** spec §1 (audit) → Task 1 Step 2 grep checks cover all 7 defects (F1–F7). F7 (VLM path) → §5 Phase 2 in the new doc. spec §2–§3 (scoring via omnidocbench-amd-windows + structure) → Task 1 doc §3–§7. spec §4 (acceptance) → Task 1 Step 2 + Task 2. spec §11 (Phase B contribution) → noted in doc §10 + a one-line future pointer (kept out of scope). ✓

**Placeholder scan:** the `<...>` tokens in the doc are intentional colleague-fill parameters (paths, shas, engine name), not plan placeholders; each is labeled. No TBD/TODO. ✓

**Type consistency:** doc-internal only — formula, paths, revision, sha, CLI flags are identical across §0/§6/§7. ✓
