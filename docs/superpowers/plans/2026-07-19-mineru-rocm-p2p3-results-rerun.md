# MinerU-ROCm P2/P3 — Full Results Re-run + Reproducibility Lock Fill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: a HYBRID — code tasks (1, 5) via superpowers:subagent-driven-development; the GPU re-run tasks (2, 3) + lock-fill (4) are a **controller-executed runbook** (long background GPU jobs the controller runs + monitors; they are NOT dispatchable to a fresh subagent). Steps use checkbox (`- [ ]`) syntax.

**Goal:** Prove the new `mineru-rocm predict | score` path reproduces **86.48 (pipeline)** + **95.56 (VLM)** on AMD ROCm via a FULL fresh re-run on OmniDocBench v1.6 (1651 pages), then fill `reproducibility.lock.yaml` with byte-exact SHAs + the re-run metrics + add the README↔lock value cross-check. This closes "evaluation-backed": code-ready → numbers-reproduced.

**Architecture:** A gold-standard full re-run (the user's explicit choice over the sample-parity fast path). Task 1 (code) first reworks the `predict` CLI so the full-run invocation is clean (no literal `--`). Tasks 2-3 (GPU runbook) run each backend over the full 1651-page set via `mineru-rocm predict`, then score via `mineru-rocm score` (the OmniDocBench `pdf_validation.py` scorer), verifying Overall within tolerance (pipeline ±1.0pp of 86.48; VLM ±0.5pp of 95.56). Task 4 fills the lock from the fresh runs. Task 5 (code) adds the README↔lock value cross-check to `check_repo.py` + activates it. The existing Plan 1/Plan 2 results (`/root/ocr-eval/mineru-{pipeline,vlm-vllm}-preds/`) serve as the **anchor** to verify against — the new runs must reproduce them.

**Tech Stack:** `mineru-rocm` CLI (P1d), MinerU 3.4 pipeline + MinerU2.5-Pro VLM (vLLM-on-ROCm), OmniDocBench v1.6 `pdf_validation.py` scorer, AMD ROCm (gfx1100), PyYAML.

## Global Constraints

(From the approved spec `docs/superpowers/specs/2026-07-19-mineru-rocm-standalone-port-design.md` §3.2/§3.4/D3.)

- **Reproduce the recorded scores (no regressions).** Pipeline Overall must land within **±1.0pp of 86.48** (i.e. ∈ [85.48, 87.48]); VLM Overall within **±0.5pp of 95.56** (i.e. ∈ [95.06, 96.06]). Overall formula `((1-text)*100 + cdm*100 + teds*100)/3`.
- **Full-set, fresh, via the NEW path.** Predict via `mineru-rocm predict` (the P1c.2 driver → `backends.{pipeline,vlm}.infer_page` → the P1c runner with atomic writes + `run_manifest.json`); score via `mineru-rocm score` (`scoring.score_directory` → `pdf_validation.py`). The new runs land in `results/omnidocbench/v1.6/{pipeline,vlm-vllm}/` (spec §3.4 layout). The OLD `/root/ocr-eval/mineru-*-preds/` dirs are the ANCHOR, not the deliverable.
- **Lock becomes the single source of truth.** `reproducibility.lock.yaml` is filled with the fresh repo commits + model/weight SHA256 + scorer commit + GT/eval-config SHA256 + the re-run metrics + env versions. README results render from it; `check_repo` cross-checks (P1d structural check + the new Task-5 value check); CI fails on drift.
- **Engine subprocess contract untouched.** P2/P3 does NOT modify `dispatcher.py`/`adapter/`/`backends/` logic (the inference code is fixed from P1). Only `cli.py` (Task 1 predict rework), `reproducibility.lock.yaml` (Task 4 fill), `scripts/check_repo.py` + `README.md` (Task 5 cross-check) change.
- **One concern per commit on branch `feat/p2p3-results` off `main` @ `6c57273`.** GPU-run commits include the prediction dir + `run_manifest.json` + `metric_result.json` (or a reference + the manifest, if the dir is too large for git — see Task 2).
- **Validation environment:** `/opt/venv/bin/python` + the `mineru-rocm` console script (py3.12; has `mineru` + `mineru_vl_utils`); scorer venv `/root/ocr-eval/OmniDocBench/.venv/bin/python` (Py3.11; `scoring.score_directory` uses this by default); GPU for inference. Dataset GT `/root/ocr-eval/OmniDocBench_data/OmniDocBench.json` (1651 pages); images `/root/ocr-eval/OmniDocBench_v16_images/`.

---

## File Structure (P2/P3 scope)

| File | Action | Responsibility |
|---|---|---|
| `src/mineru_rocm/driver.py` | Modify (Task 1) | Extract `add_arguments(parser)` from `parse_args` (DRY; shared with the CLI) |
| `src/mineru_rocm/cli.py` | Modify (Task 1) | `predict` subparser uses `driver.add_arguments` (drops the literal-`--` requirement) |
| `tests/test_cli.py` | Modify (Task 1) | Test the clean `predict` invocation (no `--`) |
| `results/omnidocbench/v1.6/pipeline/` | Create (Task 2, GPU) | Fresh full-1651 pipeline predictions + `run_manifest.json` + `metric_result.json` |
| `results/omnidocbench/v1.6/vlm-vllm/` | Create (Task 3, GPU) | Fresh full-1651 VLM predictions + `run_manifest.json` + `metric_result.json` |
| `reproducibility.lock.yaml` | Modify (Task 4) | Fill the `not_recorded` fields with fresh SHAs + metrics + env |
| `scripts/check_repo.py` | Modify (Task 5) | Add `check_readme_lock_values` (the value cross-check) |
| `tests/test_check_repo.py` | Modify (Task 5) | Test the value cross-check |
| `README.md` | Modify (Task 5) | Results tables rendered from the lock (the cross-check target) |

---

## Task 1: `predict` CLI rework — drop the literal-`--` requirement (code/SDD)

**Files:**
- Modify: `src/mineru_rocm/driver.py` (extract `add_arguments`), `src/mineru_rocm/cli.py` (predict subparser), `tests/test_cli.py`
**Interfaces:**
- Produces: `driver.add_arguments(parser: argparse.ArgumentParser) -> None` (adds the 11 driver flags to any parser; `parse_args` uses it internally). The CLI `predict` subparser calls it so `mineru-rocm predict --backend pipeline --gt-json g --images-dir i --pred-dir p` (no `--`) works.

- [ ] **Step 1: Write the failing test** (append to `tests/test_cli.py`):

```python
def test_predict_natural_invocation_reaches_driver(tmp_path, monkeypatch):
    """predict WITHOUT a literal '--' reaches driver.run (the rework drops the REMAINDER limitation)."""
    gt = tmp_path / "gt.json"
    gt.write_text(json.dumps([{"page_info": {"image_path": "a.png"}}]), encoding="utf-8")
    img = tmp_path / "images"; img.mkdir(); (img / "a.png").write_bytes(b"x")
    seen = {}
    from mineru_rocm import driver
    def _fake_run(dargs, command=None):
        seen["backend"] = dargs.backend
        return 0
    monkeypatch.setattr(driver, "run", _fake_run)
    rc = cli.main(["predict", "--backend", "pipeline",
                   "--gt-json", str(gt), "--images-dir", str(img), "--pred-dir", str(tmp_path / "pred")])
    assert rc == 0 and seen.get("backend") == "pipeline"  # reached driver.run, no '--' needed
```

- [ ] **Step 2: Run to verify it fails** — `mineru-rocm predict --backend pipeline --gt-json ...` (no `--`) currently exits 2 (argparse REMAINDER). Run `/opt/venv/bin/python -m pytest tests/test_cli.py::test_predict_natural_invocation_reaches_driver -q` → FAIL (SystemExit 2).

- [ ] **Step 3: Implement the rework:**
  - In `driver.py`: extract the 11 `p.add_argument(...)` calls out of `parse_args` into a new module-level `def add_arguments(parser): ...` that adds them to the given parser. `parse_args` becomes `p = argparse.ArgumentParser(prog="mineru_rocm.driver", ...); add_arguments(p); return p.parse_args(argv)`. (No behavior change to `parse_args` itself.)
  - In `cli.py`: the `predict` subparser — replace `extra = nargs=argparse.REMAINDER` with `driver.add_arguments(pr)` (so `--gt-json`, `--images-dir`, `--pred-dir`, `--model`, `--platform`, `--lang`, `--max-retries`, `--retry-backoff`, `--overwrite`, `--retry-failed` are first-class predict args) + keep `--backend`. The `_predict` handler becomes `return driver.run(args, command=["mineru-rocm", "predict", args.backend])` (the cli `args` Namespace now carries all the driver flags). Drop the `_clean_extra` helper + the `--` epilog (no longer needed). Lazy-import `driver` inside `_predict` (preserve GPU-free module import).

- [ ] **Step 4: Run the tests** — `/opt/venv/bin/python -m pytest tests/test_cli.py tests/test_driver.py -q` → all pass (the new natural-invocation test + the existing predict-separator test may need updating: the separator test now also passes without `--`, so either keep it as "both forms work" or replace it with the natural-invocation test). `/opt/venv/bin/mineru-rocm predict --help` shows the flags without the `--` epilog.

- [ ] **Step 5: Validate** — `/opt/venv/bin/python scripts/check_repo.py` → clean; `/opt/venv/bin/python -c "import mineru_rocm.cli; import sys; print('engine:', 'omnidocbench_amd' in sys.modules, 'torch:', 'torch' in sys.modules)"` → `engine: False | torch: False`; `/opt/venv/bin/python -m pytest -q` → 113+ passed (count unchanged or +1 net).

- [ ] **Step 6: Commit** — `git add src/mineru_rocm/driver.py src/mineru_rocm/cli.py tests/test_cli.py` → `fix(p2p3): predict CLI forwards driver flags directly (drop literal-'--' requirement); extract driver.add_arguments` (+ `Co-Authored-By` trailer).

---

## Task 2: Pipeline full re-run + score + verify 86.48 (GPU runbook — controller-executed)

**Files:** creates `results/omnidocbench/v1.6/pipeline/` (predictions + `run_manifest.json` + `metric_result.json`). **~2.9h GPU.**

- [ ] **Step 1: Pre-flight** — confirm GPU free (`rocm-smi --showuse`); confirm pipeline weights cached (the Plan 1 run used them — check the mineru cache; if missing, `mineru-models-download` first). Confirm the GT (1651 pages) + images dir:
```bash
/opt/venv/bin/python -c "import json; print('GT pages:', len(json.load(open('/root/ocr-eval/OmniDocBench_data/OmniDocBench.json'))))"
ls /root/ocr-eval/OmniDocBench_v16_images/ | wc -l   # expect 1651
```

- [ ] **Step 2: Run the full pipeline predict** (background; ~2.9h). The new path produces a CLEAN 1651-page result with a conservation-checked `run_manifest.json` (resolving the old 1742-page orphan mess):
```bash
cd /workspace/MinerU-ROCm
/opt/venv/bin/mineru-rocm predict --backend pipeline \
  --gt-json /root/ocr-eval/OmniDocBench_data/OmniDocBench.json \
  --images-dir /root/ocr-eval/OmniDocBench_v16_images \
  --pred-dir results/omnidocbench/v1.6/pipeline \
  --platform linux-rocm \
  > results/omnidocbench/v1.6/pipeline/predict.log 2>&1 &
```
Monitor: `tail -f` the log; check `run_manifest.json` appears at the end with `status: ok`. The runner's `select_todo`/`commit_success`/`write_run_manifest` drive the run; resume via `--retry-failed` if interrupted.

- [ ] **Step 3: Score the pipeline predictions** via the new `mineru-rocm score` (wraps `scoring.score_directory` → `pdf_validation.py` in the scorer venv):
```bash
/opt/venv/bin/mineru-rocm score \
  --gt-json /root/ocr-eval/OmniDocBench_data/OmniDocBench.json \
  --pred-dir results/omnidocbench/v1.6/pipeline \
  --label pipeline
```
Capture the Overall/text/CDM/TEDS from the table it prints.

- [ ] **Step 4: Verify the anchor** — Overall MUST be within **±1.0pp of 86.48** (anchor: text 0.05658 / CDM 83.07 / TEDS 82.04). If out of tolerance → STOP, investigate (systematic-debugging: is it the weights? the page set? a regression in the moved backend?). Record the actual metrics in `.superpowers/sdd/progress.md`.

- [ ] **Step 5: Verify the manifest + page count** — `/opt/venv/bin/mineru-rocm manifest verify --pred-dir results/omnidocbench/v1.6/pipeline` → `[OK]` (conservation laws hold); the prediction count is 1651 (clean — no orphans). Copy the scorer's `metric_result.json` into the pred dir (or note its path).

- [ ] **Step 6: Commit** — the predictions are ~12MB (1742 .md last time); decide: commit the full pred dir (if repo size permits) OR commit the `run_manifest.json` + `metric_result.json` + a `predict.log` tail + `.gitignore` the bulk .md (reference the on-disk dir). Prefer committing the manifest + metric_result + a sample (the full dir is regenerable from the lock inputs). Commit: `results(p2p3): pipeline full-1651 re-run via mineru-rocm predict|score — Overall <X.XX> (anchor 86.48, Δ<±>)` (+ trailer).

---

## Task 3: VLM full re-run + score + verify 95.56 (GPU runbook — controller-executed)

**Files:** creates `results/omnidocbench/v1.6/vlm-vllm/`. **~4.5h GPU** (vLLM serve + 1651-page infer). Uses GPU 0.

- [ ] **Step 1: Boot the vLLM server** (Plan 2's serve script; ~675M ViT encoder, MinerULogitsProcessor, GPU 0):
```bash
cd /workspace/MinerU-ROCm
HIP_VISIBLE_DEVICES=0 HSA_OVERRIDE_GFX_VERSION=11.0.0 VLLM_USE_V1=1 \
  bash examples/serve_vlm_vllm.sh > /tmp/vllm.log 2>&1 &
# wait for "Application startup complete" / the /v1/models endpoint to respond
```
Confirm: `curl -s http://127.0.0.1:8265/v1/models` returns the served model name.

- [ ] **Step 2: Run the full VLM predict** (background; ~4.5h). The vlm backend's `infer_page` connects to the vLLM server via `MinerUClient(backend="http-client")`; expect ~0.12% empty-page rate (Plan 2 baseline — the runner's `record_error`/`is_complete` handles them; `--retry-failed` can re-attempt empties):
```bash
/opt/venv/bin/mineru-rocm predict --backend vlm-vllm \
  --gt-json /root/ocr-eval/OmniDocBench_data/OmniDocBench.json \
  --images-dir /root/ocr-eval/OmniDocBench_v16_images \
  --pred-dir results/omnidocbench/v1.6/vlm-vllm \
  --platform linux-rocm \
  > results/omnidocbench/v1.6/vlm-vllm/predict.log 2>&1 &
```

- [ ] **Step 3: Score** — `/opt/venv/bin/mineru-rocm score --gt-json <GT> --pred-dir results/omnidocbench/v1.6/vlm-vllm --label vlm-vllm`. Capture Overall/text/CDM/TEDS.

- [ ] **Step 4: Verify the anchor** — Overall within **±0.5pp of 95.56** (anchor: text 0.03589 / CDM 96.73 / TEDS 93.54). Note the empty-page rate (expected ~0.12%; if much higher → the vLLM EOS-regression from Plan 2, investigate). `manifest verify` → `[OK]`; count 1651.

- [ ] **Step 5: Stop the vLLM server** — free GPU 0 (`pkill -f serve_vlm_vllm` + `pkill -f vllm` + confirm `rocm-smi` shows GPU 0 free).

- [ ] **Step 6: Commit** — same manifest+metric_result+sample pattern as Task 2. `results(p2p3): VLM full-1651 re-run via mineru-rocm predict|score — Overall <X.XX> (anchor 95.56, Δ<±>); empty <X%>` (+ trailer).

---

## Task 4: Fill `reproducibility.lock.yaml` (code — controller-executed)

**Files:** `reproducibility.lock.yaml`.

- [ ] **Step 1: Gather the values** (run these; capture output):
```bash
cd /workspace/MinerU-ROCm
echo "mineru_rocm commit: $(git rev-parse HEAD)"
echo "mineru version: $(/opt/venv/bin/python -c 'import mineru; print(mineru.__version__)')"
echo "mineru_vl_utils: $(/opt/venv/bin/python -c 'import mineru_vl_utils; print(mineru_vl_utils.__version__)')"
echo "torch: $(/opt/venv/bin/python -c 'import torch; print(torch.__version__, torch.version.hip)')"
echo "vllm: $(/opt/venv/bin/python -c 'import vllm; print(vllm.__version__)')"
echo "transformers: $(/opt/venv/bin/python -c 'import transformers; print(transformers.__version__)')"
echo "python: $(/opt/venv/bin/python --version)"
echo "rocm: $(rocminfo | grep -A1 'Name: GPU' | head -1); $(hipconfig --version 2>/dev/null)"
# GT + eval-config SHA256:
sha256sum /root/ocr-eval/OmniDocBench_data/OmniDocBench.json
sha256sum src/mineru_rocm/data/eval_config.yaml
# scorer commit:
git -C /root/ocr-eval/OmniDocBench rev-parse HEAD 2>/dev/null || echo "(scorer repo not git)"
# VLM model safetensors SHA256:
sha256sum /root/.cache/huggingface/models--opendatalab--MinerU2.5-Pro-2605-1.2B/snapshots/*/model.safetensors
# pipeline weights SHA256s (layout/formula/ocr/table) — sha256sum the cached weight files
```
Plus the **re-run metrics** from Tasks 2-3 (Overall/text/CDM/TEDS for both), the run date, `rocm-smi` device id.

- [ ] **Step 2: Fill the lock** — replace every `not_recorded` in `reproducibility.lock.yaml` with the gathered value (under the right section: `mineru_rocm`, `mineru`, `mineru_vl_utils`, `model`, `omnidocbench`, `environment`, `benchmark`). Use exact values; do NOT invent. For any field genuinely not derivable (e.g. the official 95.75-vs-95.69 anchor — flag it as `unverified` with a note, don't guess).

- [ ] **Step 3: Verify** — `/opt/venv/bin/python -c "import yaml; yaml.safe_load(open('reproducibility.lock.yaml'))"` (valid YAML); `/opt/venv/bin/python scripts/check_repo.py` → clean (lock-sections check passes; no `not_recorded` left in required fields).

- [ ] **Step 4: Commit** — `git add reproducibility.lock.yaml` → `repro(p2p3): fill reproducibility.lock.yaml — fresh byte-exact SHAs + re-run metrics (pipeline <X.XX>, VLM <X.XX>) + env` (+ trailer).

---

## Task 5: README↔lock value cross-check in `check_repo` (code/SDD)

**Files:** `scripts/check_repo.py`, `tests/test_check_repo.py`, `README.md`.

- [ ] **Step 1: Write the failing test** (append to `tests/test_check_repo.py`):

```python
def test_check_readme_lock_values_pass_when_consistent():
    """The README results tables match the lock values (the drift gate)."""
    import scripts.check_repo as cr
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    lock = cr._load_lock()
    findings = cr.check_readme_lock_values(readme, lock)
    assert findings == [], findings


def test_check_readme_lock_values_flags_drift():
    """A README number that disagrees with the lock is a finding."""
    import scripts.check_repo as cr
    lock = {"benchmark": {"full_1651": {"pipeline_overall": 99.99, "vlm_vllm_overall": 99.99}}}
    readme = "Pipeline Overall 86.48 | VLM 95.56"
    findings = cr.check_readme_lock_values(readme, lock)
    assert findings  # non-empty: the README 86.48/95.56 disagree with the lock 99.99
```

- [ ] **Step 2: Run to verify it fails** — `ModuleNotFoundError: check_readme_lock_values` (not yet implemented).

- [ ] **Step 3: Implement `check_readme_lock_values(readme, lock)` in `scripts/check_repo.py`** — extract the README's Pipeline + VLM Overall (regex on the results tables/badges; the spec §3.4 says README renders from the lock) + compare to `lock["benchmark"]["full_1651"]["pipeline_overall"]` / `["vlm_vllm_overall"]` (whatever schema Task 4 used). Return a finding per mismatch. Wire it into `main()` (alongside the structural checks). The check is skipped (returns `[]`) if the lock values aren't filled yet (defensive — but after Task 4 they are).

- [ ] **Step 4: Render the README from the lock** — update `README.md`'s results tables to the re-run metrics (matching the lock). `check_repo` now enforces they stay in sync.

- [ ] **Step 5: Validate** — `/opt/venv/bin/python -m pytest tests/test_check_repo.py -q` → pass; `/opt/venv/bin/python scripts/check_repo.py` → `check_repo: clean` (the README↔lock cross-check passes).

- [ ] **Step 6: Commit** — `git add scripts/check_repo.py tests/test_check_repo.py README.md` → `feat(p2p3): README↔lock value cross-check in check_repo (CI drift gate); README rendered from lock; 2 tests` (+ trailer).

---

## Definition of Done (P2/P3)

- [ ] `mineru-rocm predict --backend pipeline` ran the FULL 1651-page set via the new driver/runner (clean `run_manifest.json`, no orphans); `mineru-rocm score` → Overall within ±1.0pp of **86.48**.
- [ ] `mineru-rocm predict --backend vlm-vllm` ran the FULL 1651-page set (vLLM serve + infer); `mineru-rocm score` → Overall within ±0.5pp of **95.56**; empty-page rate ≈ 0.12%.
- [ ] `mineru-rocm manifest verify` → `[OK]` for both runs (conservation laws hold; 1651 pages each).
- [ ] `reproducibility.lock.yaml` filled — no `not_recorded` in required fields; fresh repo commits + weight/GT/eval-config/scorer SHA256 + re-run metrics + env versions.
- [ ] `scripts/check_repo.py` clean; includes the README↔lock VALUE cross-check (CI drift gate active).
- [ ] `README.md` results tables match the lock.
- [ ] `mineru-rocm predict` no longer requires `--` (Task 1 rework).
- [ ] Engine subprocess contract untouched; `pytest -q` green.

## Follow-on

- **The official anchor (95.75 vs ~95.69)**: if upstream verification is needed, that's a research task (web search the MinerU2.5-Pro paper/model card); recorded as `unverified` in the lock rather than guessed.
- **Score-twice discipline** (spec D3): optionally re-score each full run a second time (the scorer is deterministic on fixed predictions, so this is a no-op confirmation; defer unless a discrepancy surfaces).
- **`results/` size**: if committing the full prediction dirs bloats the repo, `.gitignore` the bulk `.md` + commit only `run_manifest.json` + `metric_result.json` + a sample (the predictions are regenerable from the lock inputs).
