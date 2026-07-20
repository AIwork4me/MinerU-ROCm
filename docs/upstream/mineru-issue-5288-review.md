# Review report — upstream issue #5288 optimization (2026-07-20)

Scope: audit + tighten the contribution to `opendatalab/MinerU` #5288 to a
top-tier-OSS professional level — evidence-first, accurate, no overclaims — and
align the `AIwork4me/MinerU-ROCm` repo to match. Every claim below is sourced from
the repo files/logs/lock, not from the issue's prior wording.

## 1. Files changed

| File | Change |
|---|---|
| `docs/upstream/mineru-issue-5288.md` | **Created** — optimized issue draft (Artifact 1) |
| `README.md` | "48 GB ×4" → "1 GPU per benchmark; host has 4× W7900" |
| `README.zh-CN.md` | same fix (zh) |
| `docs/benchmark-methodology.md` | same fix |
| `reproducibility.lock.yaml` | `rocm_recipe.gpu_arch` comment: removed untested-GPU list (7900 XTX/GRE/7800 XT/7700 XT/7600) → "only gfx1100 tested" |
| `scripts/check_repo.py` | added `check_version_consistency` (key-version agreement + ROCm-overclaim guardrail) |
| `tests/test_check_repo.py` | +2 tests (clean-on-repo, overclaim-flagged) |

## 2. Risk fixed per change

- **"48 GB ×4" (README ×2, benchmark-methodology):** read as 4-GPU inference. Risk: a maintainer/user assumes multi-GPU; overstates the tested config. Fix: states 1 GPU per benchmark + host has 4× (no tensor parallel) — matches the logs.
- **`rocm_recipe` GPU generalization (lock):** listed untested RDNA3 parts as covered. Risk: extrapolates one GPU to a product line. Fix: scoped to "only gfx1100 tested".
- **Version-consistency gate:** prevents future drift (ROCm 7.2 vs 7.2+, gfx1100 vs gfx1101) across README / issue draft / lock, and forbids assertion-form overclaims (`ROCm 7.2+`, `officially support`, `all RDNA3`, …). Low maintenance — pure string checks.
- **Issue draft (Artifact 1):** restructured, overclaims removed (see §5), single concrete maintainer question.

## 3. Facts confirmed from logs / lock / scorer output

| Fact | Source |
|---|---|
| **1 GPU per benchmark, no tensor parallel** | pipeline `predict.log`: "GPU Memory: 48 GB, Batch Ratio: 16"; `examples/serve_vlm_vllm.sh`: `HIP_VISIBLE_DEVICES=0`, no `--tensor-parallel-size` (default tp=1); host has 4× |
| mineru `3.4.4` / mineru_vl_utils `1.0.5` | lock + `git ls-remote` (commits `0dfc946`, `cc467fa`) |
| ROCm/HIP `7.2` | lock `environment.rocm_hip` 7.2.53211 (both venvs) |
| PyTorch `2.14.0.dev+rocm7.2` (pipeline) / `2.9.1+rocm7.2` (VLM) | lock + run_manifest `env` |
| vLLM `0.16.1.dev0` (VLM) | lock + run_manifest `env` |
| transformers `4.57.6`; Python 3.11.15 / 3.12.3; bf16 | lock |
| OS Linux `6.8.0-79-generic` | `metric_result.json` `uname` (hostname redacted, kernel kept) |
| Results pipeline **86.48** / VLM **95.46** + submetrics | lock + `metric_result.json` |
| Official anchors pipeline **86.47** / vlm-engine **95.30** | upstream MinerU README "Local Deployment" table |
| Empty pages pipeline 1 / VLM 2 (0.12%) | lock `full_1651.*.empty_pages` |
| HSA_OVERRIDE: pipeline none / VLM `11.0.0` | pipeline ran w/o it (`predict.log`); VLM requires it (`serve_vlm_vllm.sh`) |

## 4. Facts still NOT confirmable (handled honestly, not guessed)

- **Empty-page root cause:** "vLLM EOS-first-token behavior" is the hypothesis; we have no upstream issue/code link isolating it. The draft says "observed alongside vLLM's EOS-first-token behavior; we have not isolated a root cause and are not attributing it to MinerU."
- **Other GPUs / ROCm versions:** only gfx1100 + ROCm 7.2 were tested. The draft explicitly lists what is NOT covered (gfx1101/1102, RDNA4, MI-series, Radeon 7000/9000, Windows, other ROCm versions).
- **End-to-end third-party reproducibility:** not automated. The draft says "pinned for reproducibility … self-attested … has not been automated", and does NOT claim "fully reproducible".

## 5. Issue expressions tightened (old → new)

- Title: "AMD ROCm (gfx1100/RDNA3) **confirmed working** — platform table + docs contribution offer" → "Community-validated AMD ROCm (gfx1100) **compatibility** — OmniDocBench v1.6 results + docs contribution".
- "ROCm **7.2+**" (contribution offer) → "ROCm **7.2**".
- "Radeon PRO W7900, **48 GB ×4**" (Hardware) → "1× W7900 per benchmark; host has 4×; no tensor parallel".
- "torch 2.9.1+rocm7.2" (VLM-only, stated as the version) → both venv torch versions.
- Empty-output "**this is a vLLM issue, not MinerU**" (absolute) → "observed alongside … root cause not isolated".
- "**Let us know if this would be welcome**" (vague) → one concrete either/or question.
- Added explicit "community-validated, **not** official support" framing + "not requesting MinerU take on official ROCm support".

## 6. Recommend updating the live GitHub issue now?

**Yes — all six conditions are met:**

1. `gh auth status` OK (AIwork4me, token valid). ✓
2. Account can edit #5288 (it is the author). ✓
3. Optimized body passes the checks (`check_version_consistency` clean; tables well-formed; links resolve). ✓
4. All key numbers have evidence (§3). ✓
5. Git workspace changes are the intended artifacts (committed before the live update; no accidental edits). ✓
6. No unconfirmed critical hardware info — the GPU-usage question is resolved from logs (1 GPU/benchmark, no tensor parallel). ✓

## 7. Recommended final issue title

`Community-validated AMD ROCm (gfx1100) compatibility — OmniDocBench v1.6 results + docs contribution`

## Quality-gate results

- `pytest -q` → **127 passed** (+2 new).
- `scripts/check_repo.py` → clean (incl. the new `check_version_consistency`).
- `git diff --check` → no whitespace errors.
- Markdown tables in the draft → 3/3 well-formed (consistent column counts).
- Draft link `…/MinerU-ROCm/blob/main/reproducibility.lock.yaml` → HTTP 200.
