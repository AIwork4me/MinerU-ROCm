# MinerU-ROCm — Upstream PR Readiness (harden the evidence base for issue #5288)

| | |
|---|---|
| **Date** | 2026-07-20 |
| **Status** | Approved (design); awaiting implementation plan |
| **Author** | Claude (AIwork4me) |
| **North star** | Get upstream `opendatalab/MinerU` to **quickly accept** the contribution that adds AMD ROCm to official MinerU (issue [#5288](https://github.com/opendatalab/MinerU/issues/5288): "Running MinerU on AMD ROCm" doc + README GPU table row + platform mention). |
| **Scope decision** | **A′ + 仓库清理** — acceptance-optimised subset of the standalone-port spec's P4, plus repo cleanup. |
| **Quality lens** | Reviewed for **open-source community quality** (2026-07-20): honesty of claims, OPSEC, i18n consistency, contribution process, framing. Findings R1–R10 folded in below. |
| **Relates to** | [`2026-07-19-mineru-rocm-standalone-port-design.md`](2026-07-19-mineru-rocm-standalone-port-design.md) (parent standalone-port design; this spec executes an acceptance-focused slice of its P4 + closes leftover Tier-1 consistency debt). |
| **Lives in** | this repo, `docs/superpowers/specs/` |

---

## 0. TL;DR

Upstream issue #5288 offers a **docs-only** contribution (MinerU code needs zero changes for *correctness*). The speed of maintainer acceptance is governed by **trust + honesty + respect for their conventions**. The linked evidence repo `AIwork4me/MinerU-ROCm` currently undercuts that trust three ways: it contradicts itself where a maintainer looks (stale 95.56 vs 95.46), it leaks internal infrastructure in public artefacts, and its "no patches needed" framing would mislead users about VLM performance.

This spec closes exactly those gaps + produces the upstream PR content, and nothing more:

1. **Bucket 1 — the upstream PR itself** (docs-only): add a **community-verified, honestly-scoped** gfx1100/RDNA3 section to the existing `docs/zh/usage/acceleration_cards/AMD.md` (no patches needed for *correctness*; VLM-via-vLLM is slow unpatched → cross-ref the existing perf-patch content), extend the upstream `README.md` GPU-Acceleration **row only** (never the accuracy row), and gate PR-opening on a maintainer signal in issue #5288. English mirror is **optional / ask-maintainer** (the 12-page acceleration_cards family is zh-only). Prepared under `docs/upstream-pr/` (staging, not a MinerU-ROCm product).
2. **Bucket 2 — Tier-1 consistency** (the 5 surfaces a maintainer clicks): `model_card.json`, `docs/reproducibility.md`, `docs/how-it-works.md`, `Makefile`, the README `Evaluation` section — all reconciled to 95.46/86.48 + the `mineru-rocm` path.
3. **Bucket 3 — falsifiability**: pin the `mineru` 3.4.4 and `mineru_vl_utils` 1.0.5 upstream commits; record the official anchors (now known: pipeline 86.47, vlm-engine 95.30).
4. **Bucket 4 — repo cleanup + OPSEC**: archive the superseded `results/omnidocbench/v16/`; slim the ~3300 committed predictions to a 10-page sample; **redact internal IPs and host-specific paths from all public artefacts** (results JSONs + docs).

Deferred (valuable internal hygiene, **not** on the PR's acceptance critical path — tracked in `known-gaps.md`): canary materialisation, `table_sha256`, v1.0.0 release, gpu-smoke CI, architecture/hardware-matrix docs.

---

## 1. Background — the acceptance problem

### 1.1 The upstream PR is docs-only and that is an advantage

Issue #5288 establishes the ideal shape: **MinerU code is unchanged for correctness; only an env var is needed.** A docs-only PR is the lowest-risk contribution a maintainer can merge. The bottleneck to a fast merge is therefore **not** engineering scope but **maintainer confidence that the claim is true, honest, and convention-respecting.**

### 1.2 The current evidence repo fails that test

The standalone-port rewrite (v0.1.0, 2026-07-19) re-ran both models via the new `mineru-rocm predict|score` CLI and recomputed **95.46 / 86.48**. But several artefacts were never reconciled. Audit findings (verified against the working tree 2026-07-20):

| # | Finding | Evidence |
|---|---|---|
| F1 | **`model_card.json` is stale + mis-pointed.** `"overall": 95.56`, points at old `v16/..._cdm_*` engine artefacts. | `model_card.json:9,24-28` |
| F2 | **`docs/reproducibility.md` describes the pre-rewrite engine workflow** (`omnidocbench-amd infer/score/publish`, machine-local paths, GPU 3, 2026-07-18) and quotes 95.56 + old submetrics. | `docs/reproducibility.md:112-209` |
| F3 | **`docs/how-it-works.md` still says 95.56**, treats the platform repo's `hub/registry.yaml` as the badge authority. | `docs/how-it-works.md:49-54` |
| F4 | **Dual `results/` layout, unreconciled** (`v16/` 95.56-era vs `v1.6/` 95.46-era). | `results/omnidocbench/{v16,v1.6}/` |
| F5 | **Makefile + README Evaluation are engine-based** (`make eval-linux` → `omnidocbench-amd run`; README says "once `_infer` is wired up"). | `Makefile:17-20`; `README.md:56-59` |
| F6 | **`mineru` / `mineru_vl_utils` upstream commits = `not_recorded`.** | `reproducibility.lock.yaml:21,26` |
| F7 | **`official_reference` entirely `not_verified`.** | `reproducibility.lock.yaml:101-106` |
| F10 | **`HSA_OVERRIDE` documented only for the VLM path;** issue #5288 calls it "mandatory" for gfx1100. | grep `HSA_OVERRIDE` → VLM-only files |
| F12 | **~3302 full-page `page-*.md` committed under `v1.6/`**, contradicting the "10-page sample" policy. | `find results/omnidocbench/v1.6 -name 'page-*.md'` ≈ 1651/backend |
| F13 | **Internal infra leaked in public artefacts (OPSEC).** `HF_ENDPOINT=http://134.199.133.77` + `/root/ocr-eval/...` in `results/**/provenance.json`, `results/omnidocbench/v1.6/*/metric_result.json` (the authoritative score file: `python_executable: /root/ocr-eval/OmniDocBench/.venv/bin/python`), and `docs/spike-*.md`. | grep `134\.199\.133\.77`, `/root/ocr-eval` |

F8 (canary), F9 (`table_sha256`), F11 (cuda-string), F14–F16 (release/CI/extra docs) are **deferred** — see §8.

---

## 2. Verified facts (resolved during design — no placeholders)

### 2.1 Upstream dependency commits (F6) — `git ls-remote`

```
mineru        3.4.4  →  tag mineru-3.4.4-released          @ 0dfc9460cd9ab693b9af60ae3fbffd7bc111b062
mineru_vl_utils 1.0.5 → tag mineru_vl_utils-1.0.5-released  @ cc467faaddb53d8b276cedf88f09302f540a7b83
```
Repos: `github.com/opendatalab/MinerU`, `github.com/opendatalab/mineru-vl-utils`. Both resolved authoritatively. Go into the lock as `# (verified)`.

### 2.2 Official anchors (F7) — from the upstream README "Local Deployment" table

| Upstream backend | Official OmniDocBench v1.6 Overall |
|---|---:|
| **pipeline** | **86.47** |
| vlm-engine | 95.30 |
| hybrid-engine | 95.39 (high) / 95.26 (medium) |

- Our pipeline **86.48** vs **86.47** → Δ +0.01 pp (parity).
- Our VLM-vLLM **95.46** vs **vlm-engine 95.30** → Δ +0.16 pp (parity within vLLM non-determinism; **frame as parity, not "beats official"**).
- Retires the "95.75 vs 95.69" confusion: the authoritative number was never 95.75 — it is **95.30 (vlm-engine)** per the upstream README.

`reproducibility.lock.yaml.benchmark.official_reference`:
```yaml
official_reference:
  source: verified                       # opendatalab/MinerU README.md "Local Deployment" table (2026-07-20)
  source_url: https://github.com/opendatalab/MinerU/blob/master/README.md
  pipeline_overall: 86.47
  vlm_overall: 95.30                     # vlm-engine row = closest match to our vlm-vllm path
  hybrid_engine_high: 95.39              # for reference
  inference_engine: vlm-engine
  provenance_note: "Official anchors are OmniDocBench v1.6 Overall from the upstream README table. The prior 'official 95.75' was unverified and is withdrawn."
```

### 2.3 `HSA_OVERRIDE_GFX_VERSION` scope (F10) — evidence + authoritative

**Truth: the in-process PyTorch pipeline does NOT need the override; the vLLM/VLM path DOES.**

- Pipeline `predict.log` ran on GPU ("GPU Memory: 48 GB") **without** `HSA_OVERRIDE` → deterministic 86.48. PyTorch-ROCm auto-detects gfx1100.
- Every VLM artefact sets `HSA_OVERRIDE_GFX_VERSION=11.0.0`.
- PyTorch ROCm JIT-compiles and tolerates auto-detected gfx1100; vLLM ships AoT-compiled CK/Flash-Attention kernels keyed to a specific gfx string, so on RDNA3 the override bridges the mismatch (vLLM #4514; vLLM ROCm install docs).

**Canonical statement for the upstream doc:**
> - **Pipeline backend** (in-process PyTorch): works on gfx1100 with **no override** — PyTorch-ROCm auto-detects RDNA3.
> - **VLM backend via vLLM**: requires `export HSA_OVERRIDE_GFX_VERSION=11.0.0` (gfx1100/1101/1102: W7900, 7900 XTX/XT/GRE, 7800 XT, 7700 XT, 7600).
> - **Windows caveat:** may not be honoured on native Windows ROCm (windows-hip path unverified).

### 2.4 Upstream doc structure + the existing AMD.md — defines the PR shape and an honesty constraint

- mkdocs + **i18n plugin** (`docs_structure: folder`, `fallback_to_default: true`, en default). The `acceleration_cards/` family is **12 pages, all zh, zero en** — so an English mirror is structurally inconsistent unless mirrored wholesale (see R3 → §4.3).
- **`docs/zh/usage/acceleration_cards/AMD.md` already exists** and is a community **performance-hack doc** (7900 XTX): it documents a **severe RDNA3 perf cliff** — vLLM's `qwen2_vl.py` vision-encoder `Conv3d(bfloat16)` has no optimised MIOpen kernel → **12 s per execution**; the doc's Triton/matmul patch cuts the VLM to ~1.3–1.8 s/it. It never mentions `HSA_OVERRIDE` and has no benchmark scores. **Our VLM is Qwen2-VL-family (MinerU2.5-Pro) — the same op applies.** Our lock/repro shows the unpatched VLM at **~15–16 s/page (≈7 h / 1651 pages)**; the pipeline is clean at **~3–6 s/page**.
  - → **R1 honesty constraint:** the upstream doc must scope "no patches needed" to *correctness*; explicitly state the unpatched VLM is slow; cross-reference the existing perf-patch content above it. Pipeline = clean + decent speed.
- The acceleration_cards pages are **orphaned from `mkdocs.yml` `nav:`**; the README row is the discovery path. Surgical PR only.
- Upstream `README.md` "GPU Acceleration" row reads **"Volta and later architecture GPUs or Apple Silicon"** — the exact cell to extend. **The Accuracy row (86.47 / 95.30) is NOT touched** (R4).
- Upstream Python **3.10–3.13**; our venvs (3.11.15, 3.12.3) in range. ✅

### 2.5 OPSEC evidence (F13) — what must be redacted

Committed public artefacts contain internal infrastructure (verified by grep):
- Internal HF mirror IP `http://134.199.133.77` in `results/omnidocbench/v16/.../provenance.json` (`adapter_command`) and `docs/spike-*.md`, `docs/reproducibility.md`.
- Host paths `/root/ocr-eval/...`, `/opt/venv` in `results/omnidocbench/v1.6/*/metric_result.json` (`python_executable`, `python_prefix`), `results/.../provenance.json`, and several docs.
- These are linked from issue #5288. Archiving v16 (§5.6) does **not** redact it; the authoritative v1.6 `metric_result.json` also carries it.

---

## 3. Scope

### 3.1 In scope (A′ + cleanup)

- **Bucket 1** — upstream PR content (§4), with R1 honesty, R3 i18n, R4 accuracy-row discipline, R5 community-verified framing, R6 process gate.
- **Bucket 2** — Tier-1 consistency: F1, F2, F3, F5 (F4 in Bucket 4). F10 fixed via the `reproducibility.md` rewrite + upstream doc.
- **Bucket 3** — provenance: F6 (pin commits), F7 (record anchors).
- **Bucket 4** — repo cleanup + OPSEC: F4 (archive v16), F12 (prediction sampling), **F13 (OPSEC redaction)**.

### 3.2 Out of scope (deferred — tracked in `known-gaps.md`, §8)

F8 canary · F9 `table_sha256` · remaining `canary_*` lock fields · Tier-4 release maturity (v1.0.0 tag/wheel/SHA256SUMS, gpu-smoke CI, architecture/hardware-matrix/release-artifact/release-checklist docs) · any new GPU run · windows-hip verification. F11 is folded lightly into §5.3 (one line).

---

## 4. Design — Bucket 1: the upstream PR content

A **docs-only** PR to `opendatalab/MinerU`. Authored in this repo under `docs/upstream-pr/` (clearly marked **staging — not part of the MinerU-ROCm product**; R10). The contribution becomes MinerU-Open-Source-Licensed once merged (note in the PR body).

### 4.0 Process gate (R6) — before opening the PR

- **Do not open the PR unilaterally.** Issue #5288 itself says *"Let us know if this would be welcome and we'll prepare the PR."* → **Comment on #5288 first**, summarising the proposed three changes + the honesty caveats, and wait for a maintainer go-ahead (or adjust to their preference).
- Upstream has **no `CONTRIBUTING.md`** (verified). Before opening: **inspect 2–3 recently-merged doc PRs** to learn the sign-off/DCO convention (e.g. `Signed-off-by:`), commit-message style, and whether a CLA bot enforces anything. Match it.
- Keep the PR **docs-only, one PR**, no code/behaviour changes.

### 4.1 `docs/upstream-pr/README.md` — PR landing page

- **Title:** `docs: add AMD ROCm (gfx1100/RDNA3) — community-verified OmniDocBench v1.6, no code changes`.
- **Body:** links #5288; states MinerU code is unchanged for *correctness*; the three changes; the honesty caveat (VLM unpatched is slow; cross-refs existing perf patch); cites evidence repo + lock; states the contribution complements (does not alter) the existing community `AMD.md`; notes it becomes MinerU-licensed.

### 4.2 Change A — extend `docs/zh/usage/acceleration_cards/AMD.md` (R1, R5)

Append a **new top-level section** above the existing community hack content (the existing content stays untouched). The section is **honest about correctness vs performance** and **framed as community-verified**:

```markdown
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
  **追求速度请沿用上文社区 Triton/矩阵乘补丁**（可降至 ~1.3–1.8 s/it）。本节的"无需补丁"仅指**正确性**。

### OmniDocBench v1.6 全量结果（1651 页）
| 模型 / 后端 | Overall | Text EditDist ↓ | Formula CDM ↑ | Table TEDS ↑ |
|---|---:|---:|---:|---:|
| MinerU 3.4 pipeline（ROCm） | 86.48 | 0.0566 | 83.07 | 82.04 |
| MinerU2.5-Pro VLM（vLLM-on-ROCm） | 95.46 | 0.0360 | 96.46 | 93.54 |

与上游 README 官方锚点对齐（容差内）：pipeline 86.47（Δ+0.01pp）、vlm-engine 95.30（Δ+0.16pp，vLLM 非确定性范围内）。
完整可复现锁定（代码 commit、权重 SHA256、评分器 commit、环境）见
[reproducibility.lock.yaml](https://github.com/AIwork4me/MinerU-ROCm/blob/main/reproducibility.lock.yaml)。
```

### 4.3 Change B — English mirror: **optional / ask maintainer** (R3)

The acceleration_cards family is 12 pages, **all zh, no en**. Adding a single `docs/en/usage/acceleration_cards/AMD.md` is structurally inconsistent (a maintainer may say "mirror all or none"). → **The primary PR is zh-only (Change A + Change C).** Raise the en mirror in the #5288 comment and let the maintainer decide; if they want it, add it as a faithful, idiomatic English counterpart of §4.2. Do **not** machine-translate; do **not** translate the existing community hack body (version-specific).

### 4.4 Change C — extend the upstream `README.md` GPU-Acceleration **row only** (R4)

**Principle: the official Accuracy row (86.47 / 95.30) is never touched.** ROCm numbers live **only** in the AMD.md doc. The single edit is the "GPU Acceleration" row:
```
| GPU Acceleration | Volta+ / Apple Silicon / AMD ROCm (gfx1100/RDNA3; see [AMD guide](usage/acceleration_cards/AMD.md))¹ | … |
```
Footnote `¹`: VLM/vLLM path needs `HSA_OVERRIDE_GFX_VERSION=11.0.0`; pipeline does not; see AMD.md for community-verified details. No other cells changed.

### 4.5 mkdocs nav wiring (verify-then-wire)

Acceleration_cards pages are not in `nav:`. During implementation: build the docs (or inspect `usage/index.md`) to confirm discoverability via the README link. If not, add **one** `Usage → Acceleration Cards → AMD` nav entry (en default nav; i18n `nav_translations` covers zh). Do not restructure beyond this.

---

## 5. Design — Buckets 2–4: the evidence-repo hardening

### 5.1 F1 — `model_card.json`

- `"overall": 95.56` → **`95.46`**; submetrics → Text 0.0360 / CDM 96.46 / TEDS 93.54 / reading-order 0.1236.
- Add `official_reference` → upstream README 95.30 (vlm-engine).
- **Repoint artefact fields** from engine `v16/` to authoritative **`results/omnidocbench/v1.6/vlm-vllm/`**. The v1.6 runner path does not produce `run_summary`/`provenance`/`run_stats` (engine concepts); it produces `run_manifest.json`, `metric_result.json`, `_errors.jsonl`, `predict.log`. Map:
  - `metric_result` → `v1.6/vlm-vllm/metric_result.json`.
  - `sample_predictions` → `v1.6/vlm-vllm/sample_predictions/` (§5.7).
  - `run_summary` + `provenance` + `run_stats` → collapse to a `run_manifest` field → `v1.6/vlm-vllm/run_manifest.json` (or keep keys pointing there + a one-line note — pick the smaller diff, document it).
- Same for `model_card.pipeline.json` (already 86.48): repoint to `v1.6/pipeline/{run_manifest,metric_result}.json` + `sample_predictions/`.

### 5.2 F2 — `docs/reproducibility.md` (full rewrite)

Replace the 218-line engine-workflow doc with a **standalone-CLI** guide:
- Lead with `mineru-rocm predict --backend {pipeline|vlm-vllm}` → `validate` → `score` → `manifest verify`.
- Quote **95.46 / 86.48** and new submetrics throughout.
- State `HSA_OVERRIDE` truth for **both** paths (§2.3).
- **No machine-local paths / IPs** — only env-var-derived paths (`$GT_JSON`, `$IMAGES_DIR`, `$PRED_DIR`, `$SCORER_VENV`). Remove `/root/ocr-eval/…`, `/opt/venv`, `134.199.133.77`, `GPU 3`, `2026-07-18`.
- Document the two-venv reality generically (mineru-venv for inference; OmniDocBench venv for scoring).
- Reference `reproducibility.lock.yaml` as the single source of truth.

### 5.3 F3 — `docs/how-it-works.md`

- `95.56` → **`95.46`** (line 49 + registry table); submetrics aligned.
- Reframe: `mineru-rocm` CLI is primary; `omnidocbench-amd` is optional `[platform]`.
- One-line F11 clarification in the Backends table: pipeline runs "in-process on `cuda` (PyTorch-ROCm exposes the HIP device as `cuda`)".

### 5.4 F5 — `Makefile` + README `Evaluation`

- `Makefile`: `eval-linux`/`eval-windows` drive **`mineru-rocm predict`** (+ `score`) instead of `omnidocbench-amd run`. Keep `smoke-test`, `publish`.
- `README.md` `Evaluation`: replace the "once `_infer` is wired up / `make eval-linux`" text with the actual `mineru-rocm predict | score` commands (match `benchmark-methodology.md`). Remove `_infer`.

### 5.5 F6 + F7 — `reproducibility.lock.yaml`

- `mineru.commit` → `0dfc9460cd9ab693b9af60ae3fbffd7bc111b062` `# (verified)`.
- `mineru_vl_utils.commit` → `cc467faaddb53d8b276cedf88f09302f540a7b83` `# (verified)`.
- `benchmark.official_reference` → the verified block in §2.2.
- Add a `rocm_recipe` block: HSA_OVERRIDE truth (pipeline none / VLM `11.0.0`) + the canonical env-var recipe (the machine-readable source for the upstream doc's claims).
- **R8:** convert the bare `not_recorded` comments on deferred fields (`canary_*`, `table_sha256`) to `# (deferred → docs/known-gaps.md)` so they read as deliberate, not oversight.

### 5.6 F4 — archive superseded `results/omnidocbench/v16/`

- Move `results/omnidocbench/v16/` → `results/_archive/v16-engine-superseded/`.
- Add `results/_archive/README.md`: "Pre-rewrite `omnidocbench-amd` engine artefacts (95.56-era). Superseded by `results/omnidocbench/v1.6/` (the `mineru-rocm predict|score` re-run, 95.46/86.48). Retained for provenance; do not cite."
- (The OPSEC redaction in §5.9 applies to these archived files too.)

### 5.7 F12 — prediction sampling (repo slimming)

- For each of `v1.6/pipeline/`, `v1.6/vlm-vllm/`: replace ~1651 `page-*.md` with a **deterministic 10-page stratified sample** under `sample_predictions/`. Selection: 10 GT pages spanning distinct OmniDocBench doc-types, chosen by `hashlib.sha256(page_id)` (deterministic — no `Date.now`/`Math.random`); chosen stems committed in `sample_predictions/manifest.json` (page id, doc-type, md sha256).
- Keep `run_manifest.json`, `metric_result.json`, `_errors.jsonl`, `predict.log`, `predict.log.tail`.
- Add `results/omnidocbench/v1.6/*/page-*.md` to `.gitignore`; `git rm` the tracked predictions (the score is regenerable via `mineru-rocm predict | score`).

### 5.8 Consistency gate — `scripts/check_repo.py` + test

- New assertion: **tri-source headline agreement** — VLM Overall and pipeline Overall in `README.md`, `model_card.json`(+`.pipeline.json`), and `reproducibility.lock.yaml` must be byte-identical (95.46 / 86.48). CI fails on drift (structural guard against F1/F3 recurrence).
- Regression test in `tests/test_check_repo.py`.

### 5.9 F13 — OPSEC redaction (R2)

Redact internal infrastructure from **all public-facing committed files** (the repo is linked from issue #5288):
- **`results/` JSONs** (v1.6 `metric_result.json`/`run_manifest.json`, archived v16 `provenance.json`/`run_summary.json`): replace host paths `/root/ocr-eval/...`, `/opt/venv/...` and the internal IP `http://134.199.133.77` with neutral placeholders (`<scorer-venv>/bin/python`, `<hf-mirror>`). These carry no information for a third party. Do this as a deterministic post-processing pass; record the redaction rule in `docs/reproducibility.md` so it is reproducible.
- **Docs**: `docs/spike-*.md`, `docs/vlm-engine-sample.md`, and any other doc carrying the IP/host paths → replace with placeholders or remove the host-specific lines (spike docs are internal-use; acceptable to genericise).
- **Sanity gate**: add a `check_repo.py` assertion that **no committed file under `results/` or `docs/` (excl. this spec + archived `_archive/README.md`) contains `134.199.133.77` or `/root/ocr-eval`** — prevents re-leakage. (Reproducibility is preserved: the lock pins commits + SHAs; host paths were never meaningful to anyone else.)

---

## 6. Testing / verification

- `make smoke-test` (pytest) green; new `test_check_repo.py` covers tri-source agreement + the OPSEC no-leak assertion.
- `scripts/check_repo.py` green (README↔lock cross-check + headline assertion + OPSEC scan + SPDX + pip-install smoke).
- `reuse lint` green (new files carry SPDX headers; `docs/upstream-pr/` drafts included).
- Manual: `results/omnidocbench/v1.6/{pipeline,vlm-vllm}/metric_result.json` Overall == README/model_card/lock (95.46 / 86.48) after cleanup, **and** `grep -r 134.199.133.77 results/ docs/` returns nothing (excl. permitted).
- **Upstream docs build (R9):** before the PR, run `mkdocs build` (or a markdown link-lint) on the upstream tree with our changes to confirm the page renders and the README link resolves.
- The upstream PR content is reviewed in `docs/upstream-pr/`; opened by the user only after the §4.0 process gate.

---

## 7. Risks & open questions

1. **Maintainer signal (R6).** The PR is opened only after a maintainer responds to the #5288 comment. If they request changes (e.g. zh-only, or a full en mirror, or different doc placement), adjust — the content is staged to be adaptable.
2. **mkdocs nav discoverability (§4.5).** Verify during implementation; wire one entry only if required.
3. **en mirror asymmetry (§4.3).** We add zh only; the en mirror is maintainer-led. Note in the PR body.
4. **VLM performance perception (R1).** Despite the cross-reference, some users may still expect unpatched vLLM to be fast. The doc states the speed honestly; the existing community patch is the documented remedy.
5. **`mineru-rocm score` venv assumption.** The repro rewrite keeps the scorer-venv requirement clear (`--venv-python` / `OMNIDOCBENCH_VENV`). No behaviour change.
6. **Push caveat (from the parent spec).** git-push from this env is historically flaky; the upstream PR is opened by the user.

---

## 8. Out of scope (deferred — recorded in `docs/known-gaps.md`, R8)

F8 canary · F9 `table_sha256` · remaining `canary_*` lock fields · F11 standalone (folded into §5.3) · v1.0.0 release + wheel/SHA256SUMS/tags · gpu-smoke CI · architecture/hardware-matrix/release-artifact/release-checklist docs. Each is added to `known-gaps.md` as a tracked item (and, where useful, a GitHub issue) so the backlog is not silently dropped. These do not gate the upstream PR.

---

## 9. Success criteria (definition of done)

- [ ] `docs/upstream-pr/` contains the zh AMD.md section (honest: correctness vs perf; community-verified) + the README row change + a PR landing README; the en mirror is staged-but-optional; **the §4.0 process-gate comment is drafted**. Ready for the user to open against `opendatalab/MinerU` after maintainer signal.
- [ ] `model_card.json`(+`.pipeline.json`) shows 95.46/86.48 and points at `results/omnidocbench/v1.6/…`.
- [ ] `docs/reproducibility.md` describes `mineru-rocm predict|score`, quotes 95.46/86.48, **no machine-local paths/IPs**, documents HSA_OVERRIDE for both paths.
- [ ] `docs/how-it-works.md` shows 95.46; `Makefile` + README `Evaluation` drive `mineru-rocm`.
- [ ] `reproducibility.lock.yaml` pins both upstream commits (verified) + records official anchors (verified); deferred fields annotated `→ known-gaps.md`.
- [ ] `results/omnidocbench/v16/` archived + superseded README; `v1.6/` predictions slimmed to 10-page sample/backend + `.gitignore`.
- [ ] **OPSEC: no `134.199.133.77` or `/root/ocr-eval` in `results/` or `docs/`** (gated by `check_repo.py`).
- [ ] `scripts/check_repo.py` enforces tri-source headline agreement + OPSEC no-leak; new tests green; `make smoke-test` + `check_repo.py` + `reuse lint` green.
- [ ] **R7: MinerU-ROCm `CHANGELOG.md` updated** with an `[Unreleased]` entry describing this hardening + the upstream-PR staging.
- [ ] **R8: `docs/known-gaps.md` records the deferred items.**
