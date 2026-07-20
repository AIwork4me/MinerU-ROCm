# ROCm Upstream PR Readiness — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden `MinerU-ROCm` into a consistent, OPSEC-clean, falsifiable evidence base and stage a docs-only upstream PR, so upstream `opendatalab/MinerU` quickly accepts the AMD ROCm contribution (issue #5288).

**Architecture:** Four buckets executed in dependency order — (3) lock provenance → (2) Tier-1 consistency → (4) repo cleanup + OPSEC → (1) upstream-PR content — with each invariant locked in by a `scripts/check_repo.py` gate + test written TDD-style in the same task that satisfies it.

**Tech Stack:** Python 3.11 stdlib + PyYAML (core); pytest (tests); bash/git (moves); markdown (docs). No GPU, no new deps.

**Spec:** [`docs/superpowers/specs/2026-07-20-rocm-upstream-pr-readiness-design.md`](../specs/2026-07-20-rocm-upstream-pr-readiness-design.md) (commit `ab95330`).

## Global Constraints

- **Headline numbers are frozen:** VLM Overall **95.46**, pipeline Overall **86.48** (submetrics: VLM Text 0.0360 / CDM 96.46 / TEDS 93.54 / read-order 0.1236; pipeline Text 0.0566 / CDM 83.07 / TEDS 82.04 / read-order 0.1534). Official anchors: pipeline **86.47**, vlm-engine **95.30** (upstream README).
- **Verified upstream commits:** `mineru` 3.4.4 → `0dfc9460cd9ab693b9af60ae3fbffd7bc111b062`; `mineru_vl_utils` 1.0.5 → `cc467faaddb53d8b276cedf88f09302f540a7b83`.
- **HSA_OVERRIDE truth:** pipeline = none (PyTorch auto-detects gfx1100); VLM/vLLM = `11.0.0` (required).
- **OPSEC:** no literal `134.199.133.77` or `/root/ocr-eval` in `results/**` or `docs/**` (excl. `docs/superpowers/**`) after Task 8.
- **Determinism:** any selection/redaction script uses `hashlib` (no `random`/`Date`).
- **Branch:** `feat/rocm-upstream-pr-readiness` (already created; spec committed at `ab95330`). One commit per task.
- **Green throughout:** after each task, `python -m pytest -q` and `python scripts/check_repo.py` must pass (the `check_install_smoke` step in `check_repo.py` runs `pip install -e .` — slow; the pytest suite skips it).

---

## File Map

| File | Responsibility | Task |
|---|---|---|
| `reproducibility.lock.yaml` | pin commits, official anchors, recipe, deferred annotations | 1 |
| `model_card.json`, `model_card.pipeline.json` | headline numbers + repoint to `v1.6/` artefacts | 2 |
| `scripts/check_repo.py` | new gates: model_card↔lock agreement; upstream-commit pinned; OPSEC no-leak | 2, 8 |
| `tests/test_check_repo.py` | tests for the new gates | 2, 8 |
| `docs/how-it-works.md` | 95.46 + standalone identity + cuda/HIP note | 3 |
| `Makefile`, `README.md` (Evaluation) | drive `mineru-rocm`; drop machine paths | 4 |
| `docs/reproducibility.md` | full rewrite to `mineru-rocm` path, no host paths/IPs | 5 |
| `results/_archive/v16-engine-superseded/` | archived old engine artefacts + README | 6 |
| `results/omnidocbench/v1.6/{pipeline,vlm-vllm}/sample_predictions/` | 10-page stratified sample + manifest | 7 |
| `.gitignore` | ignore `page-*.md` under `v1.6/` | 7 |
| `results/**/*.json`, `docs/spike-*.md`, `docs/vlm-engine-sample.md` | OPSEC redaction | 8 |
| `docs/upstream-pr/{README.md, AMD.md.section.zh.md, README.row.md, issue-5288-comment.md}` | staged upstream PR content | 9 |
| `CHANGELOG.md`, `docs/known-gaps.md` | record hardening + deferred backlog | 10 |

---

## Task 1: Pin upstream commits + official anchors + recipe in the lock

**Files:**
- Modify: `reproducibility.lock.yaml` (sections `mineru`, `mineru_vl_utils`, `benchmark.official_reference`; add `rocm_recipe`; annotate deferred fields)
- Test: `tests/test_check_repo.py` (new test)

**Interfaces:**
- Produces: `lock["mineru"]["commit"]`, `lock["mineru_vl_utils"]["commit"]` are real SHAs; `lock["benchmark"]["official_reference"]["source"] == "verified"`; `lock["rocm_recipe"]` block. Later tasks' gates read these.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_check_repo.py`:

```python
def test_upstream_commits_pinned_in_lock():
    """mineru + mineru_vl_utils carry real git commit SHAs (not 'not_recorded')."""
    import scripts.check_repo as cr
    lock = cr._load_lock()
    for dep in ("mineru", "mineru_vl_utils"):
        commit = (lock.get(dep) or {}).get("commit", "")
        assert commit != "not_recorded" and len(commit) == 40, f"{dep}.commit not pinned: {commit!r}"


def test_official_reference_verified():
    """The official anchor is sourced from the upstream README (not not_verified)."""
    import scripts.check_repo as cr
    lock = cr._load_lock()
    ref = (lock.get("benchmark") or {}).get("official_reference") or {}
    assert ref.get("source") == "verified", f"official_reference.source not verified: {ref.get('source')!r}"
    assert ref.get("pipeline_overall") == 86.47
    assert ref.get("vlm_overall") == 95.30
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_check_repo.py::test_upstream_commits_pinned_in_lock tests/test_check_repo.py::test_official_reference_verified -q`
Expected: FAIL (`commit == 'not_recorded'`; `source == 'not_verified'`).

- [ ] **Step 3: Edit the lock**

In `reproducibility.lock.yaml`, replace the `mineru:` block's commit line:

```yaml
mineru:
  repo: https://github.com/opendatalab/MinerU
  commit: 0dfc9460cd9ab693b9af60ae3fbffd7bc111b062   # (verified) git ls-remote --tags … refs/tags/mineru-3.4.4-released
  version: "3.4.4"                                    # (verified) pip show mineru (reuse venv)
```

Replace the `mineru_vl_utils:` block's commit line:

```yaml
mineru_vl_utils:
  repo: https://github.com/opendatalab/mineru-vl-utils
  commit: cc467faaddb53d8b276cedf88f09302f540a7b83   # (verified) git ls-remote --tags … refs/tags/mineru_vl_utils-1.0.5-released
  version: "1.0.5"                                    # (verified) both venvs
```

Replace the entire `benchmark.official_reference:` block with:

```yaml
  official_reference:                    # (verified 2026-07-20) upstream README "Local Deployment" table
    source: verified
    source_url: https://github.com/opendatalab/MinerU/blob/master/README.md
    pipeline_overall: 86.47              # (verified) README table, pipeline row
    vlm_overall: 95.30                   # (verified) README table, vlm-engine row (closest match to our vlm-vllm path)
    hybrid_engine_high: 95.39            # (verified) README table, for reference
    inference_engine: vlm-engine         # (verified) upstream label for the vLLM-served VLM path we mirror
    provenance_note: "Official anchors are OmniDocBench v1.6 Overall from the upstream README 'Local Deployment' table. The prior 'official 95.75' was unverified and is withdrawn."
```

Add a new top-level `rocm_recipe:` block (insert immediately before `environment:`):

```yaml
rocm_recipe:                             # (verified) the canonical gfx1100/RDNA3 recipe (mirrors the upstream doc)
  gpu_arch: gfx1100                      # RDNA3 — Radeon PRO W7900 (also gfx1101/1102: 7900 XTX/XT/GRE, 7800 XT, 7700 XT, 7600)
  hsa_override:
    pipeline: none                       # in-process PyTorch auto-detects gfx1100
    vlm_vllm: "11.0.0"                   # vLLM AoT-compiled kernels require it
  install: |
    pip install -U "mineru[all]"         # + ROCm torch wheel; VLM additionally needs vLLM-on-ROCm
  cli: |
    mineru-rocm predict --backend {pipeline|vlm-vllm} --gt-json <gt> --images-dir <imgs> --pred-dir <out> --platform linux-rocm
    mineru-rocm score --gt-json <gt> --pred-dir <out> --label {pipeline|vlm-vllm}
```

Annotate the deferred fields — change the trailing comments on these four lines from `# (not_recorded) …` to `# (deferred → docs/known-gaps.md)`:
- `gt_json_canary_sha256`
- `canary_manifest_sha256`
- the three lines under `benchmark.canary_N:` (`pipeline_overall`, `vlm_vllm_overall`, `vlm_transformers_overall`)

**OPSEC (pre-flight amendment):** redact host-specific venv paths from the lock's `environment.venvs` comments — the lock is a public artefact linked from issue #5288. Change:
- the `pipeline:` comment from `# /root/ocr-eval/mineru-rocm-venv (Py3.11) — has mineru` to `# mineru infer venv (Py3.11) — has mineru`
- the `vlm_vllm:` comment from `# /opt/venv (Py3.12) — has vllm 0.16.1 + mineru_vl_utils` to `# vLLM VLM venv (Py3.12) — has vllm 0.16.1 + mineru_vl_utils`
- any other `/root/ocr-eval` / `/opt/venv` literal under `cross_check_source` (e.g. the vlm `cross_check_source: "local HF cache (/root/.cache/huggingface)…"` is fine — standard HF cache, keep).

- [ ] **Step 4: Run test to verify it passes + OPSEC grep**

Run: `python -m pytest tests/test_check_repo.py::test_upstream_commits_pinned_in_lock tests/test_check_repo.py::test_official_reference_verified -q`
Expected: PASS.
Run: `grep -nE "/root/ocr-eval|/opt/venv" reproducibility.lock.yaml` → expect **no matches**.

- [ ] **Step 5: Commit**

```bash
git add reproducibility.lock.yaml tests/test_check_repo.py
git commit -m "fix(lock): pin mineru@0dfc946 + mineru_vl_utils@cc467fa; record official anchors (86.47/95.30); add rocm_recipe"
```

---

## Task 2: Fix `model_card.json` + `model_card.pipeline.json` + tri-source gate

**Files:**
- Modify: `model_card.json`, `model_card.pipeline.json`
- Modify: `scripts/check_repo.py` (add `check_modelcard_lock_agreement`)
- Test: `tests/test_check_repo.py` (new test)

**Interfaces:**
- Consumes: `lock["benchmark"]["full_1651"]["{vlm_vllm,pipeline}"]["overall"]` (Task 1).
- Produces: model cards show 95.46/86.48 and point at `results/omnidocbench/v1.6/…`; `check_modelcard_lock_agreement(lock)` gate.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_check_repo.py`:

```python
def test_modelcard_lock_agreement():
    """model_card.json (VLM) + model_card.pipeline.json Overall match the lock (tri-source)."""
    import json
    import scripts.check_repo as cr
    lock = cr._load_lock()
    full = (lock.get("benchmark") or {}).get("full_1651") or {}
    expected = {
        "model_card.json": (full.get("vlm_vllm") or {}).get("overall"),
        "model_card.pipeline.json": (full.get("pipeline") or {}).get("overall"),
    }
    for fname, exp in expected.items():
        if exp is None:
            continue
        card = json.loads((cr.REPO / fname).read_text(encoding="utf-8"))
        assert card["overall"] == exp, f"{fname}.overall {card['overall']} != lock {exp}"
        # artefacts must point at the authoritative v1.6 tree, not the old v16 engine tree
        arts = json.dumps(card.get("artifacts", {}))
        assert "omnidocbench/v1.6/" in arts and "omnidocbench/v16/" not in arts, f"{fname} still points at v16/"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_check_repo.py::test_modelcard_lock_agreement -q`
Expected: FAIL (`model_card.json overall 95.56 != 95.46`; artefacts point at `v16/`).

- [ ] **Step 3: Rewrite `model_card.json`**

Replace the entire file content with:

```json
{
  "schema_version": 1,
  "model_id": "mineru2.5",
  "model_version": "2605",
  "platforms": ["linux-rocm", "windows-hip"],
  "badge": {"linux-rocm": "community", "windows-hip": "community-wanted"},
  "eval_date": "2026-07-19",
  "omnidocbench_version": "v1.6",
  "overall": 95.46,
  "submetrics": {
    "text_edit_dist": 0.0360,
    "text_accuracy_percent": 96.40,
    "table_teds_percent": 93.54,
    "table_teds_structure_only_percent": 95.89,
    "formula_cdm_percent": 96.46,
    "reading_order_edit_dist": 0.1236
  },
  "hardware": {
    "gpu": "AMD gfx1100 (Radeon PRO W7900)",
    "vram": "48 GB",
    "rocm_driver": "7.2"
  },
  "official_reference": {
    "source": "upstream README 'Local Deployment' table",
    "source_url": "https://github.com/opendatalab/MinerU/blob/master/README.md",
    "vlm_engine_overall": 95.30,
    "delta_pp": 0.16
  },
  "artifacts": {
    "run_manifest": "results/omnidocbench/v1.6/vlm-vllm/run_manifest.json",
    "metric_result": "results/omnidocbench/v1.6/vlm-vllm/metric_result.json",
    "predict_log": "results/omnidocbench/v1.6/vlm-vllm/predict.log",
    "sample_predictions": "results/omnidocbench/v1.6/vlm-vllm/sample_predictions/"
  }
}
```

> Note: `table_teds_structure_only_percent: 95.89` is a placeholder-free estimate only if unverifiable — **verify the real value** from `results/omnidocbench/v1.6/vlm-vllm/metric_result.json` before committing (see Step 3b). If unavailable, drop the `table_teds_structure_only_percent` key entirely rather than guess.

- [ ] **Step 3b: Verify the structure-only submetric from the authoritative metric file**

Run: `python -c "import json;d=json.load(open('results/omnidocbench/v1.6/vlm-vllm/metric_result.json'));print(json.dumps({k:v for k,v in d.items() if 'TEDS' in k or 'teds' in k},indent=2))"` — read the `table_teds_structure_only_percent` (or `TEDS_structure_only`) value for the v1.6 vlm-vllm run and put the real number in `model_card.json`. If the field is absent from the v1.6 metric file, **remove** the `table_teds_structure_only_percent` line from the card.

- [ ] **Step 4: Edit `model_card.pipeline.json`**

It already shows `overall: 86.48` and correct submetrics — only repoint artefacts + eval_date. Replace the `"eval_date"` line and the `"artifacts"` block:

```json
  "eval_date": "2026-07-19",
```

```json
  "artifacts": {
    "run_manifest": "results/omnidocbench/v1.6/pipeline/run_manifest.json",
    "metric_result": "results/omnidocbench/v1.6/pipeline/metric_result.json",
    "predict_log": "results/omnidocbench/v1.6/pipeline/predict.log",
    "sample_predictions": "results/omnidocbench/v1.6/pipeline/sample_predictions/"
  }
```

- [ ] **Step 5: Add the gate to `scripts/check_repo.py`**

Add this function (after `check_readme_lock_values`):

```python
def check_modelcard_lock_agreement(lock) -> list[str]:
    """model_card.json (VLM) + model_card.pipeline.json Overall match the lock, and
    artefacts point at the authoritative v1.6 tree (not the superseded v16 engine tree)."""
    if lock is None:
        return []
    full = (lock.get("benchmark") or {}).get("full_1651") or {}
    findings = []
    mapping = {"model_card.json": "vlm_vllm", "model_card.pipeline.json": "pipeline"}
    for fname, key in mapping.items():
        exp = (full.get(key) or {}).get("overall")
        if exp is None:
            continue
        path = REPO / fname
        if not path.is_file():
            findings.append(f"{fname} missing")
            continue
        import json
        card = json.loads(path.read_text(encoding="utf-8"))
        if card.get("overall") != exp:
            findings.append(f"{fname}.overall {card.get('overall')} != lock full_1651.{key}.overall {exp}")
        arts = json.dumps(card.get("artifacts", {}))
        if "omnidocbench/v1.6/" not in arts or "omnidocbench/v16/" in arts:
            findings.append(f"{fname} artefacts do not point at the authoritative results/omnidocbench/v1.6/ tree")
    return findings
```

Wire it into `main()` — add after the `findings += check_readme_lock_values(readme, lock)` line:

```python
    findings += check_modelcard_lock_agreement(lock)
```

And wire it into the integration test `test_check_repo_clean_on_repo` — add before the final `assert`:

```python
    findings += cr.check_modelcard_lock_agreement(cr._load_lock())
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_check_repo.py -q`
Expected: PASS (all checks, including the new agreement gate + integration).

- [ ] **Step 7: Commit**

```bash
git add model_card.json model_card.pipeline.json scripts/check_repo.py tests/test_check_repo.py
git commit -m "fix(model_card): VLM 95.46 + repoint both cards to v1.6/ artefacts; add tri-source modelcard↔lock gate"
```

---

## Task 3: Reconcile `docs/how-it-works.md` (95.46 + identity + cuda/HIP note)

**Files:**
- Modify: `docs/how-it-works.md`
- Modify: `scripts/check_repo.py` (add `check_no_stale_overall`), `tests/test_check_repo.py`

**Interfaces:** none new.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_check_repo.py`:

```python
def test_no_stale_vlm_overall_in_docs():
    """No user-facing doc under docs/ (excl. superpowers/) still quotes the stale 95.56."""
    import scripts.check_repo as cr
    findings = cr.check_no_stale_overall()
    assert findings == [], findings
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_check_repo.py::test_no_stale_vlm_overall_in_docs -q`
Expected: FAIL (`docs/how-it-works.md` contains `95.56`).

- [ ] **Step 3: Add the gate to `scripts/check_repo.py`**

Add after `check_modelcard_lock_agreement`:

```python
_STALE_VLM_OVERALL = "95.56"
def check_no_stale_overall(repo=REPO) -> list[str]:
    """No user-facing doc under docs/ (excluding docs/superpowers/ design records)
    still carries the withdrawn stale VLM Overall (95.56)."""
    errs = []
    for md in (repo / "docs").rglob("*.md"):
        if "superpowers" in md.parts:
            continue
        if _STALE_VLM_OVERALL in md.read_text(encoding="utf-8"):
            errs.append(f"{md.relative_to(repo)} still quotes stale Overall {_STALE_VLM_OVERALL}")
    return errs
```

Wire into `main()` (after the modelcard line) and into `test_check_repo_clean_on_repo`:

```python
    findings += check_no_stale_overall()
```

- [ ] **Step 4: Fix `docs/how-it-works.md`**

Edit the "Two model cards, one repo" table rows. Replace `**95.56**` (two occurrences in that section) with `**95.46**`. Replace any `95.56` in the "Registry update note" paragraph with `95.46` and update the parenthetical to read `(Overall **95.46**, badge linux-rocm `community`)`.

In the "Backends" table, change the `pipeline` row's "what it does" cell from:

```
wraps upstream `mineru[all]` in-process on cuda → markdown
```

to:

```
wraps upstream `mineru[all]` in-process on `cuda` (PyTorch-ROCm exposes the HIP device as `cuda`) → markdown
```

In the "Stages (engine-side)" subsection heading, the body still says the `omnidocbench-amd` CLI runs the stages — prepend a one-line clarification as the first line of that subsection:

```
> **Primary interface is the `mineru-rocm` CLI** (`predict` → `validate` → `score` → `manifest verify`). The `omnidocbench-amd` stages below apply only when using the optional `[platform]` engine extra.
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_check_repo.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add docs/how-it-works.md scripts/check_repo.py tests/test_check_repo.py
git commit -m "docs(how-it-works): 95.46 + standalone-CLI identity + cuda/HIP clarification; gate against stale 95.56"
```

---

## Task 4: `Makefile` + README `Evaluation` → drive `mineru-rocm` (drop machine paths)

**Files:**
- Modify: `Makefile`
- Modify: `README.md` (the `## Evaluation` section, ~lines 54–63)
- Modify: `scripts/check_repo.py` (`check_readme_scripts_exist` already covers script refs)

**Interfaces:** none new.

- [ ] **Step 1: Replace the `Makefile`**

Replace the whole file with:

```make
PLATFORM ?= linux-rocm
BACKEND  ?= pipeline
GT_JSON  ?= OmniDocBench.json
IMAGES_DIR ?= images
PRED_DIR ?= $(IMAGES_DIR)-preds-$(BACKEND)
SCORER_VENV ?= $(VIRTUAL_ENV)

setup-linux:
	bash adapter/setup/00-install-deps.sh
setup-windows:
	powershell -ExecutionPolicy Bypass -File adapter\setup\00-install-deps.ps1

demo:
	OUT=$$(mktemp -d); python adapter/run_adapter.py --img-dir examples --out-dir $$OUT --platform $(PLATFORM) --backend smoke; ls $$OUT

predict:
	mineru-rocm predict --backend $(BACKEND) \
	  --gt-json $(GT_JSON) --images-dir $(IMAGES_DIR) \
	  --pred-dir $(PRED_DIR) --platform $(PLATFORM)

score:
	mineru-rocm score --gt-json $(GT_JSON) --pred-dir $(PRED_DIR) \
	  --label $(BACKEND) --venv-python $(SCORER_VENV)/bin/python

# Full OmniDocBench v1.6 eval = predict + score for both backends (linux-rocm).
# Override paths via env: make eval-linux GT_JSON=x IMAGES_DIR=y PRED_DIR=z SCORER_VENV=/path
eval-linux eval-windows:
	$(MAKE) predict BACKEND=pipeline
	$(MAKE) score    BACKEND=pipeline
	$(MAKE) predict BACKEND=vlm-vllm
	$(MAKE) score    BACKEND=vlm-vllm

publish:
	omnidocbench-amd conformance . && echo CONFORMANT

smoke-test:
	python -m pytest
```

This drops the machine-local `OMNIDOCBENCH_IMG_DIR ?= /root/ocr-eval/...` default (OPSEC) and routes eval through the standalone CLI.

- [ ] **Step 2: Edit README `## Evaluation`**

Replace the block (currently):

```
Run the full OmniDocBench v1.6 pipeline (download → infer → score → publish) once `_infer` is wired up:

```bash
make eval-linux      # linux-rocm
# make eval-windows  # windows-hip (run on Windows)
```

Eval config: [`eval/configs/omnidocbench_v16.yaml`](eval/configs/omnidocbench_v16.yaml).
```

with:

```
Run the full OmniDocBench v1.6 eval (infer + score for both backends) via the standalone CLI:

```bash
# predict → score (set GT_JSON / IMAGES_DIR / PRED_DIR / SCORER_VENV to your paths)
mineru-rocm predict --backend pipeline \
  --gt-json OmniDocBench.json --images-dir images/ --pred-dir out/ --platform linux-rocm
mineru-rocm score --gt-json OmniDocBench.json --pred-dir out/ --label pipeline \
  --venv-python <scorer-venv>/bin/python
# repeat with --backend vlm-vllm for the VLM
```

Or via make (overrides via env): `make eval-linux GT_JSON=… IMAGES_DIR=… PRED_DIR=… SCORER_VENV=…`.
See [`docs/reproducibility.md`](docs/reproducibility.md) for the full recipe and [`docs/benchmark-methodology.md`](docs/benchmark-methodology.md) for the reproduce commands.
```

- [ ] **Step 3: Verify**

Run: `make -n eval-linux` (dry-run) → expect lines invoking `mineru-rocm predict` / `mineru-rocm score` (not `omnidocbench-amd run`).
Run: `grep -n "/root/ocr-eval" Makefile README.md` → expect no matches.

- [ ] **Step 4: Run the gate suite**

Run: `python -m pytest tests/test_check_repo.py -q`
Expected: PASS (the README no longer references a nonexistent script; `make` targets are not script-refs so `check_readme_scripts_exist` is unaffected).

- [ ] **Step 5: Commit**

```bash
git add Makefile README.md
git commit -m "fix(makefile,readme): eval drives mineru-rocm predict|score; drop machine-local OMNIDOCBENCH_IMG_DIR default"
```

---

## Task 5: Rewrite `docs/reproducibility.md` (standalone path, no host paths/IPs)

**Files:**
- Modify: `docs/reproducibility.md` (full rewrite — the 218-line engine-workflow doc)

**Interfaces:** none new. This task is verified by the OPSEC gate added in Task 8 (no `/root/ocr-eval`, no `134.199.133.77`) + the stale-Overall gate (Task 3).

- [ ] **Step 1: Replace the entire file**

Replace `docs/reproducibility.md` with:

````markdown
# Reproducibility

A score is only meaningful if someone else can reproduce it from the committed repo. The standalone `mineru-rocm` CLI is the primary path; `reproducibility.lock.yaml` is the single source of truth (pinned commits, byte-exact weight/GT SHAs, scorer commit, both venvs' environment, the metric formula, the official anchors, and the ROCm recipe).

## Results (OmniDocBench v1.6, full 1651 pages, gfx1100 / ROCm 7.2)

| Backend | Overall | Text EditDist ↓ | Formula CDM ↑ | Table TEDS ↑ | read-order EditDist |
|---|---:|---:|---:|---:|---:|
| MinerU 3.4 pipeline | **86.48** | 0.0566 | 83.07 | 82.04 | 0.1534 |
| MinerU2.5-Pro VLM (vLLM-on-ROCm) | **95.46** | 0.0360 | 96.46 | 93.54 | 0.1236 |

Official anchors (upstream README "Local Deployment" table): pipeline **86.47** (Δ +0.01 pp), vlm-engine **95.30** (Δ +0.16 pp — within vLLM non-determinism).

**Overall** = `((1 − text_EditDist) × 100 + formula_CDM × 100 + table_TEDS × 100) / 3`, OmniDocBench `page.ALL` aggregation; reading-order EditDist is reported separately and is **not** part of Overall.

## The ROCm recipe (the only gfx1100-specific fact)

- GPU: AMD gfx1100 (Radeon PRO W7900, 48 GB). ROCm 7.2, bf16.
- `HSA_OVERRIDE_GFX_VERSION`:
  - **pipeline backend** (in-process PyTorch): **not required** — PyTorch-ROCm auto-detects gfx1100.
  - **VLM backend via vLLM**: **required** — `export HSA_OVERRIDE_GFX_VERSION=11.0.0` (vLLM's AoT-compiled kernels need it; applies to gfx1100/1101/1102).
- Performance: pipeline ~3–6 s/page (no patches). VLM via vLLM is **correct without patches but slow** (~15–16 s/page); for speed, community Triton patches for the `qwen2_vl.py` Conv3d exist upstream — see the upstream `docs/zh/usage/acceleration_cards/AMD.md`.

## The two venvs (reality)

Inference and scoring need different environments (MinerU pulls a ROCm torch; OmniDocBench's scorer pins its own deps and uses **no** torch for CDM). Use two venvs:

- **infer venv** — Python 3.11/3.12, `mineru[all]` 3.4.4 (+ ROCm torch wheel); for the VLM also `mineru_vl_utils` 1.0.5 + a vLLM-on-ROCm wheel. Versions pinned in the lock.
- **scorer venv** — OmniDocBench's pinned scoring deps (`bs4`, `apted`, `Levenshtein`, `pylatexenc`, `scipy`, …). CDM shells out to `pdflatex`/`magick`.

## Reproduce (commands; substitute your own paths)

```bash
# 1. pipeline (infer venv)
export HSA_OVERRIDE_GFX_VERSION=11.0.0   # harmless for pipeline; REQUIRED for VLM
mineru-rocm predict --backend pipeline \
  --gt-json "$GT_JSON" --images-dir "$IMAGES_DIR" \
  --pred-dir "$PRED_DIR_PIPELINE" --platform linux-rocm
mineru-rocm validate --gt-json "$GT_JSON" --pred-dir "$PRED_DIR_PIPELINE"
mineru-rocm score --gt-json "$GT_JSON" --pred-dir "$PRED_DIR_PIPELINE" \
  --label pipeline --venv-python "$SCORER_VENV/bin/python"
mineru-rocm manifest verify --pred-dir "$PRED_DIR_PIPELINE"

# 2. VLM via vLLM — first serve the model, then:
mineru-rocm predict --backend vlm-vllm \
  --gt-json "$GT_JSON" --images-dir "$IMAGES_DIR" \
  --pred-dir "$PRED_DIR_VLM" --platform linux-rocm
mineru-rocm score --gt-json "$GT_JSON" --pred-dir "$PRED_DIR_VLM" \
  --label vlm-vllm --venv-python "$SCORER_VENV/bin/python"
```

The authoritative artefacts land under each pred-dir: `run_manifest.json` (conservation laws), `metric_result.json`, `_errors.jsonl`, `predict.log`. The committed copies live under `results/omnidocbench/v1.6/{pipeline,vlm-vllm}/`.

## Serving the VLM (vLLM-on-ROCm)

```bash
HIP_VISIBLE_DEVICES=0 HSA_OVERRIDE_GFX_VERSION=11.0.0 VLLM_USE_V1=1 \
  bash examples/serve_vlm_vllm.sh     # serves --served-model-name mineru-pro, bf16, --enforce-eager
bash examples/wait_vlm.sh             # polls /v1/models
```

Server flags are recorded in the lock's `rocm_recipe`. Empty-page rate ~0.12% (2/1651) — vLLM EOS-first-token behaviour on a few sparse pages; absorbed by the 1651-page average.

## Non-determinism

- **pipeline**: deterministic across runs (byte-identical predictions).
- **VLM (vLLM)**: ~0.1 pp run-to-run drift (the P2/P3 re-run scored 95.46 vs a prior 95.56 — Δ −0.10 pp, within the ±0.5 pp gate). bf16 matmul kernel non-determinism.

## Provenance in the lock

`reproducibility.lock.yaml` records: the `mineru-rocm` results commit, the **upstream `mineru`/`mineru_vl_utils` git commits** (resolved via `git ls-remote` against the release tags), byte-exact weight + GT SHAs, the scorer commit, both venvs' full environment, the official anchors, and the metric formula. Deferred fields (`canary_*`, `table_sha256`) are annotated `→ docs/known-gaps.md`.
````

- [ ] **Step 2: Verify (manual + upcoming gate)**

Run: `grep -nE "/root/ocr-eval|134\.199\.133\.77|95\.56|omnidocbench-amd (run|infer|publish)" docs/reproducibility.md` → expect **no matches**.
Run: `python -m pytest tests/test_check_repo.py -q` → PASS.

- [ ] **Step 3: Commit**

```bash
git add docs/reproducibility.md
git commit -m "docs(reproducibility): rewrite to mineru-rocm predict|score path; 95.46/86.48; no host paths/IPs; HSA_OVERRIDE for both paths"
```

---

## Task 6: Archive the superseded `results/omnidocbench/v16/` engine artefacts

**Files:**
- Move: `results/omnidocbench/v16/` → `results/_archive/v16-engine-superseded/`
- Create: `results/_archive/README.md`

**Interfaces:** none. After Task 2, no committed file references `results/omnidocbench/v16/` (model cards repointed; how-it-works + repro rewritten).

- [ ] **Step 1: Confirm nothing live points at v16**

Run: `grep -rn "results/omnidocbench/v16" --include='*.json' --include='*.md' --include='*.yaml' . | grep -v '_archive' | grep -v 'docs/superpowers/'`
Expected: no matches (model_card, how-it-works, repro all updated). If any remain, fix them first.

- [ ] **Step 2: Move the directory + write the archive README**

```bash
mkdir -p results/_archive
git mv results/omnidocbench/v16 results/_archive/v16-engine-superseded
```

Create `results/_archive/README.md`:

```markdown
# Archived results — superseded

These are the **pre-rewrite `omnidocbench-amd` engine artefacts** (the `run_summary.json` /
`provenance.json` / `metric_result.json` / `run_stats.json` publish outputs, 95.56-era).

**Superseded by** `results/omnidocbench/v1.6/{pipeline,vlm-vllm}/` — the `mineru-rocm
predict|score` re-run (Overall **95.46 / 86.48**, the authoritative numbers).

Retained for provenance history only; **do not cite**. Host-specific paths/IPs in these
legacy JSONs have been redacted (see `docs/reproducibility.md`).
```

- [ ] **Step 3: Verify**

Run: `git status --short` → shows the rename + new README.
Run: `ls results/omnidocbench/` → shows `v1.6/` only (no `v16/`).

- [ ] **Step 4: Commit**

```bash
git add results/_archive
git commit -m "chore(results): archive superseded v16 engine artefacts under results/_archive/ + README"
```

---

## Task 7: Slim `v1.6/` predictions to a 10-page stratified sample + `.gitignore`

**Files:**
- Create: `results/omnidocbench/v1.6/{pipeline,vlm-vllm}/sample_predictions/` (10 `.md` + `manifest.json`)
- Delete: the ~1651 `page-*.md` (and other full-set prediction `.md`) per backend
- Modify: `.gitignore`

**Interfaces:** none.

- [ ] **Step 1: Write the deterministic sampler**

Create `scripts/sample_predictions.py`:

```python
#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
"""Pick a deterministic 10-page stratified sample from a prediction dir.

Selection: stable sha256 of the page stem (no randomness) → sorted → first 10.
Writes sample_predictions/<stem>.md (copied) + sample_predictions/manifest.json.
Usage: python scripts/sample_predictions.py <pred_dir>
"""
from __future__ import annotations
import hashlib, json, shutil, sys
from pathlib import Path

N = 10

def pick(pred_dir: Path) -> list[str]:
    stems = sorted(
        (p.stem for p in pred_dir.glob("page-*.md")),
        key=lambda s: hashlib.sha256(s.encode()).hexdigest(),
    )
    return stems[:N]

def main(pred_dir: Path) -> int:
    out = pred_dir / "sample_predictions"
    out.mkdir(exist_ok=True)
    chosen = pick(pred_dir)
    manifest = []
    for stem in chosen:
        src = pred_dir / f"{stem}.md"
        dst = out / f"{stem}.md"
        shutil.copyfile(src, dst)
        manifest.append({"stem": stem, "sha256": hashlib.sha256(src.read_bytes()).hexdigest()})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[OK] {out}: {len(manifest)} pages; manifest written")
    return 0

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: sample_predictions.py <pred_dir>", file=sys.stderr); sys.exit(2)
    sys.exit(main(Path(sys.argv[1])))
```

- [ ] **Step 2: Generate the samples**

```bash
python scripts/sample_predictions.py results/omnidocbench/v1.6/pipeline
python scripts/sample_predictions.py results/omnidocbench/v1.6/vlm-vllm
```
Expected: each prints `[OK] …: 10 pages; manifest written`.

- [ ] **Step 3: Remove the full-set predictions from git**

```bash
# keep run_manifest/metric_result/_errors/predict.log/.tail + the new sample_predictions/
git rm -r --quiet results/omnidocbench/v1.6/pipeline/page-*.md 2>/dev/null || true
git rm -r --quiet results/omnidocbench/v1.6/vlm-vllm/page-*.md 2>/dev/null || true
# also drop any non-page-*.md full-set predictions (PPT_*, *.pdf_*.md) committed earlier:
find results/omnidocbench/v1.6 -type f -name '*.md' ! -path '*/sample_predictions/*' \
  ! -name 'page-*.md' -print -delete
```

Then verify only the intended artefacts remain:

```bash
for d in results/omnidocbench/v1.6/pipeline results/omnidocbench/v1.6/vlm-vllm; do
  echo "== $d =="; ls "$d"; echo "sample_predictions:"; ls "$d/sample_predictions" | head
done
```
Expected (per dir): `.gitignore _errors.jsonl metric_result.json predict.log predict.log.tail run_manifest.json sample_predictions/`; sample_predictions has 10 `.md` + `manifest.json`.

- [ ] **Step 4: Add `.gitignore` rule**

Append to the repo `.gitignore`:

```
# Full-set predictions are regenerable; only the 10-page sample is committed.
results/omnidocbench/v1.6/*/page-*.md
results/omnidocbench/v1.6/*/*.pdf_*.md
```

- [ ] **Step 5: Verify**

Run: `git status --short | wc -l` → shows deletions + new sample files (a large but bounded change).
Run: `find results/omnidocbench/v1.6 -name 'page-*.md' | wc -l` → `0` (now gitignored).

- [ ] **Step 6: Commit**

```bash
git add scripts/sample_predictions.py results/omnidocbench/v1.6 .gitignore
git commit -m "chore(results): slim v1.6 predictions to 10-page deterministic sample/backend; gitignore full-set page-*.md"
```

---

## Task 8: OPSEC redaction + no-leak gate

**Files:**
- Modify: `results/_archive/v16-engine-superseded/**/*.json`, `results/omnidocbench/v1.6/*/metric_result.json`
- Modify: `docs/spike-vlm-vllm.md`, `docs/spike-mineru-api.md`, `docs/vlm-engine-sample.md`
- Modify: `scripts/check_repo.py` (add `check_no_internal_infra`), `tests/test_check_repo.py`

**Interfaces:** none new.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_check_repo.py`:

```python
def test_no_internal_infra_in_public_files():
    """No committed file under results/ or docs/ (excl. docs/superpowers/) leaks the
    internal HF mirror IP or host eval-root path."""
    import scripts.check_repo as cr
    findings = cr.check_no_internal_infra()
    assert findings == [], findings
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_check_repo.py::test_no_internal_infra_in_public_files -q`
Expected: FAIL (multiple files contain `134.199.133.77` / `/root/ocr-eval`).

- [ ] **Step 3: Add the gate to `scripts/check_repo.py`**

Add after `check_no_stale_overall`:

```python
_LEAK_PATTERNS = ("134.199.133.77", "/root/ocr-eval", "/opt/venv")
def check_no_internal_infra(repo=REPO) -> list[str]:
    """No public-facing file under results/ or docs/ (excluding docs/superpowers/
    design records), nor the root reproducibility.lock.yaml, contains internal infra
    (HF mirror IP, host eval root, host venv)."""
    errs = []
    targets = []
    for sub in ("results", "docs"):
        for p in (repo / sub).rglob("*"):
            if p.is_file() and p.suffix in (".json", ".md", ".yaml", ".yml", ".log") and "superpowers" not in p.parts:
                targets.append(p)
    lock = repo / "reproducibility.lock.yaml"          # public; linked from issue #5288
    if lock.is_file():
        targets.append(lock)
    for p in targets:
        txt = p.read_text(encoding="utf-8", errors="ignore")
        for pat in _LEAK_PATTERNS:
            if pat in txt:
                errs.append(f"{p.relative_to(repo)} leaks internal infra pattern {pat!r}")
    return errs
```

Wire into `main()` (after `check_no_stale_overall()`) and into `test_check_repo_clean_on_repo`:

```python
    findings += check_no_internal_infra()
```

- [ ] **Step 4: Write the redactor**

Create `scripts/redact_internal.py`:

```python
#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
"""Redact internal infrastructure from public artefacts in results/ and docs/.

Replaces the internal HF mirror IP, the host eval-root, and the host venv path with
neutral placeholders. Idempotent. Usage: python scripts/redact_internal.py
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REPLACEMENTS = {
    "http://134.199.133.77": "<hf-mirror>",
    "134.199.133.77": "<hf-mirror>",
    "/root/ocr-eval": "<eval-root>",
    "/opt/venv": "<host-venv>",
}
SUFFIXES = (".json", ".md", ".yaml", ".yml")

def main() -> int:
    changed = []
    for sub in ("results", "docs"):
        for p in (REPO / sub).rglob("*"):
            if not p.is_file() or p.suffix not in SUFFIXES or "superpowers" in p.parts:
                continue
            txt = p.read_text(encoding="utf-8")
            new = txt
            for old, repl in REPLACEMENTS.items():
                new = new.replace(old, repl)
            if new != txt:
                p.write_text(new, encoding="utf-8")
                changed.append(str(p.relative_to(REPO)))
    print(f"[OK] redacted {len(changed)} file(s):")
    for c in changed:
        print(f"  - {c}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run the redactor + spot-check docs that need manual wording**

```bash
python scripts/redact_internal.py
```

Then review `docs/spike-*.md` and `docs/vlm-engine-sample.md` — these are internal spike docs; after redaction the placeholders should read sensibly. If a line became nonsensical (e.g. a path used in a prose sentence), edit it to a clean generic form. **Do not** restore any literal IP/path.

- [ ] **Step 6: Run the gate to verify it passes**

Run: `python -m pytest tests/test_check_repo.py::test_no_internal_infra_in_public_files -q`
Expected: PASS.
Run: `python scripts/check_repo.py` → `check_repo: clean` (or only the slow install-smoke skipped).

- [ ] **Step 7: Commit**

```bash
git add scripts/redact_internal.py scripts/check_repo.py tests/test_check_repo.py results/ docs/spike-vlm-vllm.md docs/spike-mineru-api.md docs/vlm-engine-sample.md
git commit -m "fix(opsec): redact internal HF-mirror IP + host paths from results/ + docs/; add no-leak gate"
```

---

## Task 9: Stage the upstream PR content under `docs/upstream-pr/`

**Files:**
- Create: `docs/upstream-pr/README.md`, `docs/upstream-pr/AMD.md.section.zh.md`, `docs/upstream-pr/README.row.md`, `docs/upstream-pr/issue-5288-comment.md`

**Interfaces:** none. These are staging artefacts (not part of the MinerU-ROCm product); they will be ported into a PR against `opendatalab/MinerU` by the user after the §4.0 process gate.

- [ ] **Step 1: Create `docs/upstream-pr/README.md` (the PR landing page)**

```markdown
<!-- staging for upstream PR to opendatalab/MinerU — NOT part of the MinerU-ROCm product.
     Once merged upstream, this content becomes MinerU-Open-Source-Licensed. -->

# Upstream PR: add AMD ROCm (gfx1100/RDNA3) to MinerU docs

**PR title:** `docs: add AMD ROCm (gfx1100/RDNA3) — community-verified OmniDocBench v1.6, no code changes`

**Linked issue:** #5288

## Three changes (docs-only, one PR)
1. **`docs/zh/usage/acceleration_cards/AMD.md`** — append `AMD.md.section.zh.md` as a new top-level section **above** the existing community content (the existing perf-patch content is untouched).
2. **`README.md`** — extend the GPU-Acceleration **row only** per `README.row.md` (the Accuracy row is NOT touched).
3. **English mirror** — optional / maintainer-led (see the issue comment); the 12-page acceleration_cards family is currently zh-only.

## Honesty caveats (also in the PR body)
- "No patches needed" applies to **correctness only**. The VLM via vLLM runs correctly unpatched but slowly (~15–16 s/page); for speed, users should apply the existing community Triton patch already documented on the same page.
- Numbers are **community-verified** (AIwork4me/MinerU-ROCm), not official MinerU support — aligned with the README WARNING on non-mainline environments.
- 95.46 VLM is within tolerance of the official vlm-engine 95.30 (Δ +0.16 pp, vLLM non-determinism) — framed as parity, not superiority.

## Process gate
Do NOT open until a maintainer responds to the #5288 comment (`issue-5288-comment.md`). Match sign-off/DCO conventions from a recently-merged doc PR (no CONTRIBUTING.md exists upstream).
```

- [ ] **Step 2: Create `docs/upstream-pr/AMD.md.section.zh.md` (the new section)**

```markdown
<!-- Append this as a new top-level section ABOVE the existing community content in
     docs/zh/usage/acceleration_cards/AMD.md. Do not alter the existing content below. -->

## gfx1100（RDNA3）— Radeon PRO W7900 / ROCm 7.2：社区验证（非官方支持）

> 以下为社区验证结果（[AIwork4me/MinerU-ROCm](https://github.com/AIwork4me/MinerU-ROCm)），非 MinerU 官方支持。
> 上游 README 已声明"非主线环境不保证 100% 可用、欢迎社区反馈"——本节即此类反馈。

MinerU 3.4 流水线与 MinerU2.5-Pro VLM（经 vLLM）在 gfx1100 上经全量 OmniDocBench v1.6（1651 页）
验证可**正确**运行，**无需修改任何 MinerU 源码**（仅环境变量）。

### 环境
GPU：gfx1100（Radeon PRO W7900，48 GB）｜ROCm 7.2，bf16，torch 2.9.1+rocm7.2｜
mineru 3.4.4（pipeline）；mineru_vl_utils 1.0.5 + vLLM-on-ROCm 0.16.1（VLM）

### 关键配置：HSA_OVERRIDE_GFX_VERSION（gfx1100/1101/1102）
- **pipeline 后端**（进程内 PyTorch）：**无需** override —— PyTorch-ROCm 自动识别 RDNA3。
- **VLM 后端经 vLLM**：**必须** `export HSA_OVERRIDE_GFX_VERSION=11.0.0`（vLLM 预编译内核需要）。
- Windows 原生 ROCm 可能不识别此 override（windows-hip 未验证）。

### 性能：重要
- **pipeline**：无需补丁，~3–6 s/页，速度正常。
- **VLM（vLLM）**：**无需补丁即可正确运行，但未打补丁时 ~15–16 s/页（偏慢）**。
  原因同上文：vLLM 的 `qwen2_vl.py` 视觉编码器 `Conv3d(bf16)` 在 RDNA3 缺优化内核而回退。
  **追求速度请沿用上文社区 Triton/矩阵乘补丁**（可降至 ~1.3–1.8 s/it）。本节"无需补丁"仅指**正确性**。

### OmniDocBench v1.6 全量结果（1651 页）
| 模型 / 后端 | Overall | Text EditDist ↓ | Formula CDM ↑ | Table TEDS ↑ |
|---|---:|---:|---:|---:|
| MinerU 3.4 pipeline（ROCm） | 86.48 | 0.0566 | 83.07 | 82.04 |
| MinerU2.5-Pro VLM（vLLM-on-ROCm） | 95.46 | 0.0360 | 96.46 | 93.54 |

与上游 README 官方锚点对齐（容差内）：pipeline 86.47（Δ+0.01pp）、vlm-engine 95.30（Δ+0.16pp，vLLM 非确定性范围内）。
完整可复现锁定（代码 commit、权重 SHA256、评分器 commit、环境）见
[reproducibility.lock.yaml](https://github.com/AIwork4me/MinerU-ROCm/blob/main/reproducibility.lock.yaml)。
```

- [ ] **Step 3: Create `docs/upstream-pr/README.row.md` (the single README edit)**

```markdown
<!-- In opendatalab/MinerU README.md "Local Deployment" table, replace ONLY the
     "GPU Acceleration" row's first cell. Do NOT touch the Accuracy row. -->

Before:
| GPU Acceleration | Volta and later architecture GPUs or Apple Silicon | … |

After:
| GPU Acceleration | Volta+ / Apple Silicon / AMD ROCm (gfx1100/RDNA3; see [AMD guide](usage/acceleration_cards/AMD.md))¹ | … |

Footnote (add near the table footnotes):
¹ VLM/vLLM path requires `HSA_OVERRIDE_GFX_VERSION=11.0.0` on gfx1100/RDNA3; the pipeline backend does not. Community-verified (see AMD guide).
```

- [ ] **Step 4: Create `docs/upstream-pr/issue-5288-comment.md` (the process-gate comment)**

```markdown
<!-- Post this as a comment on issue #5288 BEFORE opening the PR. Wait for maintainer signal. -->

Thanks for the earlier discussion. We've prepared the docs-only contribution and would like a quick steer before opening the PR:

1. **`docs/zh/usage/acceleration_cards/AMD.md`** — add a new "gfx1100 (RDNA3) — community-verified" section (the existing perf-patch content stays untouched). It covers the `HSA_OVERRIDE` recipe and full-set OmniDocBench v1.6 numbers (pipeline 86.48, VLM 95.46), scoped honestly: "no patches needed" is about **correctness**; the unpatched VLM is ~15–16 s/page and we cross-reference the existing Triton patch for speed.
2. **`README.md`** — extend the GPU-Acceleration **row only** to mention AMD ROCm (gfx1100/RDNA3); the Accuracy row is unchanged.
3. **English mirror** — the acceleration_cards family is currently zh-only; would you like an `en/usage/acceleration_cards/AMD.md` mirror in the same PR, or keep it zh-only for consistency?

Two questions: (a) is this scope welcome as one docs-only PR? (b) any sign-off/DCO convention we should follow (we didn't find a CONTRIBUTING.md)? Full reproducibility lock: https://github.com/AIwork4me/MinerU-ROCm/blob/main/reproducibility.lock.yaml
```

- [ ] **Step 5: Verify**

Run: `ls docs/upstream-pr/` → the four files.
Run: `python scripts/check_repo.py` → clean (the `docs/upstream-pr/` files contain no literal internal IP/paths and no `95.56`, so they pass the gates; they are under `docs/` but not `docs/superpowers/`, which is correct — they must be clean).

- [ ] **Step 6: Commit**

```bash
git add docs/upstream-pr
git commit -m "feat(upstream-pr): stage docs-only MinerU ROCm contribution (zh AMD.md section + README row + process-gate comment)"
```

---

## Task 10: Update `CHANGELOG.md` + record deferred items in `docs/known-gaps.md`

**Files:**
- Modify: `CHANGELOG.md`
- Modify: `docs/known-gaps.md`

**Interfaces:** none.

- [ ] **Step 1: Add an `[Unreleased]` entry to `CHANGELOG.md`**

Replace the `## [Unreleased]` block (currently "No changes since v0.1.0.") with:

```markdown
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
- Slimmed `results/omnidocbench/v1.6/{pipeline,vlm-vllm}/` full-set predictions (~3300 `.md`) to a deterministic 10-page stratified sample per backend; `.gitignore` now excludes `page-*.md`.

### Security
- Redacted the internal HF-mirror IP and host eval-root/venv paths from all public artefacts under `results/` + `docs/`; added a `check_repo.py` no-leak gate.

### Added
- `scripts/check_repo.py` gates: modelcard↔lock tri-source agreement; no-stale-95.56; no-internal-infra leak (+ tests).
- `scripts/sample_predictions.py`, `scripts/redact_internal.py`.
- `docs/upstream-pr/` — staged docs-only contribution to `opendatalab/MinerU` (zh AMD.md section + README GPU row + #5288 process-gate comment).
```

- [ ] **Step 2: Add deferred items to `docs/known-gaps.md`**

Append a new section:

```markdown

## Deferred — upstream-PR-readiness backlog (2026-07-20)

Tracked here so they are not silently dropped (do not block the upstream PR):

- **Canary subset** not materialized — `reproducibility.lock.yaml` fields `canary_N.*`, `gt_json_canary_sha256`, `canary_manifest_sha256` are annotated `# (deferred → docs/known-gaps.md)`. Build via `mineru-rocm canary materialize` + a stratified manifest when picked up.
- **`pipeline_weights.table_sha256`** not recorded — table sub-models are pinned by the `PDF-Extract-Kit-1.0` `hf_revision ed6b654c`; record a representative file SHA when picked up.
- **v1.0.0 release** not cut — needs tag + wheel + `SHA256SUMS` + `release-artifact.md`/`release-checklist.md`.
- **`gpu-smoke.yml`** GPU-CI bridge not added (self-hosted gfx1100 runner topology TBD).
- **Docs**: `architecture.md`, `hardware-matrix.md`, `release-artifact.md`, `release-checklist.md` still missing (spec §8).
- **windows-hip** results still `community-wanted`.
```

- [ ] **Step 3: Verify**

Run: `python scripts/check_repo.py` → clean.
Run: `python -m pytest tests/test_check_repo.py -q` → PASS.

- [ ] **Step 4: Commit**

```bash
git add CHANGELOG.md docs/known-gaps.md
git commit -m "docs(changelog,known-gaps): record upstream-PR-readiness hardening + deferred backlog"
```

---

## Task 11: Final verification

**Files:** none (verification only).

- [ ] **Step 1: Full gate suite**

```bash
python -m pytest -q                       # all tests
python scripts/check_repo.py              # clean (incl. pip-install smoke)
```
Expected: all PASS; `check_repo: clean`.

- [ ] **Step 2: Targeted OPSEC + consistency grep**

```bash
grep -rn "134.199.133.77" results/ docs/ | grep -v 'docs/superports/' | grep -v 'docs/superpowers/'   # no matches
grep -rn "/root/ocr-eval" results/ docs/ | grep -v 'docs/superpowers/'                                # no matches
grep -rn "95\.56" docs/ results/ | grep -v 'docs/superpowers/'                                        # no matches
grep -rn "results/omnidocbench/v16" . | grep -v '_archive' | grep -v 'docs/superpowers/'              # no matches
```
Expected: no matches on each.

- [ ] **Step 3: Headline tri-source spot-check**

```bash
python -c "
import json, yaml
lock = yaml.safe_load(open('reproducibility.lock.yaml'))
print('lock   :', lock['benchmark']['full_1651']['vlm_vllm']['overall'], lock['benchmark']['full_1651']['pipeline']['overall'])
print('vlm    :', json.load(open('model_card.json'))['overall'])
print('pipeline:', json.load(open('model_card.pipeline.json'))['overall'])
"
```
Expected: `95.46` and `86.48` everywhere.

- [ ] **Step 4: Branch state**

```bash
git log --oneline main..HEAD             # the 11 task commits
git status --short                        # clean tree
```

- [ ] **Step 5: Report**

Summarise: gates green, OPSEC clean, numbers tri-source-consistent, upstream-PR staged under `docs/upstream-pr/`, deferred items recorded. The branch `feat/rocm-upstream-pr-readiness` is ready for PR (or merge to `main` per the user's call); opening the PR against `opendatalab/MinerU` remains gated on the #5288 maintainer signal (human action).

---

## Self-Review (run after writing — done)

**1. Spec coverage:** Bucket 3 (F6/F7) → Task 1. Bucket 2 (F1→T2, F3→T3, F5→T4, F2→T5). Bucket 4 (F4→T6, F12→T7, F13→T8). Bucket 1 → Task 9. R1 honesty → Task 9 §4.2 content + T5 recipe. R2 OPSEC → T8 + T4 Makefile. R3 i18n → T9 (en mirror optional). R4 accuracy-row → T9 README.row.md. R5 framing → T9 zh section. R6 process gate → T9 issue-comment. R7 CHANGELOG → T10. R8 deferred → T1 annotations + T10 known-gaps. R9 mkdocs build → called out in Task 9 Step 5 note + spec §6 (run by user against upstream tree, since the upstream mkdocs is not in this repo). R10 staging note → T9 README header. F10 HSA_OVERRIDE → T1 lock recipe + T5 repro. F11 cuda → T3. All covered.

**2. Placeholder scan:** Task 2 Step 3 has one value (`table_teds_structure_only_percent`) flagged for verification from the metric file with an explicit fallback (drop the key) — not a placeholder, a verified-or-removed instruction. No TBD/TODO. All code blocks complete.

**3. Type consistency:** `check_modelcard_lock_agreement`, `check_no_stale_overall`, `check_no_internal_infra` — names match between `check_repo.py` additions, `main()` wiring, and `test_check_repo.py`. Sampler writes `manifest.json` (list of `{stem, sha256}`) — consistent. `results/omnidocbench/v1.6/<backend>/sample_predictions/` path consistent across T2 (artefacts), T7 (creation), T9 (links).

> **Note on R9 (mkdocs build):** the upstream `mkdocs.yml` is not in this repo, so the build verification runs when the user ports `docs/upstream-pr/` into a checkout of `opendatalab/MinerU` (before opening the PR). The plan cannot run it here; this is called out rather than left implicit.
