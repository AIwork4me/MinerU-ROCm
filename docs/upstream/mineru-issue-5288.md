# Issue draft — opendatalab/MinerU #5288 (optimized)

> Staging file for the live GitHub issue. Evidence-first, professional, scoped to what
> was actually tested. Every number/version below is sourced from
> `reproducibility.lock.yaml`, `results/omnidocbench/v1.6/**`, the run manifests, or
> the upstream README. Status: community-validated configuration — **not** an
> official-support claim, and the upstream comparison is **not** a controlled
> CUDA-vs-ROCm parity measurement.

---

**Title:** Community-validated AMD ROCm (gfx1100) compatibility — OmniDocBench v1.6 results + docs contribution

## Summary

We ran MinerU 3.4 (pipeline) and MinerU2.5-Pro (VLM served via vLLM) end-to-end on the full **OmniDocBench v1.6** benchmark (1651 pages) on **AMD gfx1100 (Radeon PRO W7900) / ROCm 7.2 / Linux**, with **no changes to MinerU source**. This issue shares the verified results + a reproducibility lock, and asks whether the maintainers would accept a **docs-only** contribution describing this as a *community-validated* ROCm configuration. We are **not** requesting that MinerU take on official ROCm support; that scope is the maintainers' call.

## Why this issue

- MinerU's mainline support table covers NVIDIA Volta+ and Apple Silicon; ROCm is not listed, and the README notes non-mainline environments are not guaranteed but community feedback is welcome.
- A community `docs/zh/usage/acceleration_cards/AMD.md` already exists, but it is a performance-patching guide for a different GPU (7900 XTX) and carries no benchmark scores or reproducible configuration.
- This contribution adds a full OmniDocBench v1.6 evaluation on gfx1100 with a pinned environment, so the claim is falsifiable.

## Verified configuration

All entries are recorded (with source comments) in [`reproducibility.lock.yaml`](https://github.com/AIwork4me/MinerU-ROCm/blob/main/reproducibility.lock.yaml).

| Component | Value (verified) |
|---|---|
| MinerU (pipeline) | `3.4.4` — `opendatalab/MinerU` @ `0dfc946` (tag `mineru-3.4.4-released`) |
| MinerU VLM utils | `1.0.5` — `opendatalab/mineru-vl-utils` @ `cc467fa` |
| VLM model | `MinerU2.5-Pro-2605-1.2B` (HF revision `bff20d4`; safetensors SHA256 in lock) |
| GPU | **1× Radeon PRO W7900 per benchmark** (gfx1100 / RDNA3, 48 GB); host has 4× W7900 |
| Parallelism | single-GPU per worker; **tensor-parallel size 1** (no TP) |
| ROCm / HIP | `7.2` (`7.2.53211`) |
| PyTorch | `2.14.0.dev20260717+rocm7.2` (pipeline) / `2.9.1+gitff65f5b` (VLM) — two venvs |
| vLLM | `0.16.1.dev0+g89a77b108.d20260317` (VLM http-client server — full build identifier) |
| transformers | `4.57.6` |
| dtype | `bfloat16` |
| OS | Linux `6.8.0-79-generic` |
| Python | `3.11.15` (pipeline) / `3.12.3` (VLM) |

> Note: **only ROCm 7.2 and only gfx1100 were tested.** Other ROCm versions and other AMD GPUs (including other RDNA3 variants such as gfx1101/1102, RDNA4, Radeon 7000/9000 consumer parts, and MI-series) are **not covered** by this evaluation.

## OmniDocBench v1.6 results (full 1651 pages)

| Model | Backend | Overall | Text EditDist ↓ | Formula CDM ↑ | Table TEDS ↑ |
|---|---|---:|---:|---:|---:|
| MinerU 3.4 pipeline | in-process PyTorch on ROCm | **86.48** | 0.0566 | 83.07 | 82.04 |
| MinerU2.5-Pro VLM | vLLM-on-ROCm (http-client) | **95.46** | 0.0360 | 96.46 | 93.54 |

For context, MinerU's own README "Local Deployment" table publishes Overall anchors of **86.47** (pipeline) and **95.30** (`vlm-engine`); our results differ by +0.01 pp and +0.16 pp respectively.

> **These upstream scores are contextual reference anchors, not a controlled CUDA-vs-ROCm parity comparison.** The upstream table does not pin the exact hardware, model revision, inference-framework build, decoding configuration, or preprocessing used for those scores. Our VLM result was produced through a vLLM HTTP-server/client path, while the upstream table labels its reference as `vlm-engine`. The deltas indicate consistency with the published range, **not** hardware superiority or strict numerical parity.

Overall formula (OmniDocBench v1.6): `((1 − text_EditDist) × 100 + formula_CDM × 100 + table_TEDS × 100) / 3`, per-page-then-averaged (`page.ALL`). Reading-order EditDist is reported separately and is not part of Overall.

## What worked without MinerU source changes

- `mineru.backend.pipeline.pipeline_analyze.ModelSingleton` — loads and infers on ROCm.
- `mineru_vl_utils.MinerUClient` (http-client) — connects to a vLLM-on-ROCm server normally.
- `mineru.cli.common.do_parse` — the standard pipeline CLI works.

The only ROCm-specific configuration is an environment variable, **and it is path-dependent (not a MinerU patch)**:
- **Pipeline backend** (in-process PyTorch): no override needed — PyTorch-ROCm detected gfx1100 without it.
- **VLM backend via vLLM**: the tested vLLM-on-ROCm server required `export HSA_OVERRIDE_GFX_VERSION=11.0.0`. This is an environment requirement **observed** with the tested vLLM build, not a MinerU source-code requirement. We have only tested this on gfx1100 and are not claiming it applies to other RDNA3 variants or other architectures.

## Reproduction

The package is installed from source (it is **not** published on PyPI):

```bash
git clone https://github.com/AIwork4me/MinerU-ROCm.git
cd MinerU-ROCm
git checkout v0.1.0            # = lock mineru_rocm.commit (dd591469); the results-producing tree
pip install -e .
mineru-rocm --help             # sanity: prints subcommands

mineru-rocm predict --backend pipeline \
  --gt-json OmniDocBench.json --images-dir images/ --pred-dir out/ --platform linux-rocm
mineru-rocm score --gt-json OmniDocBench.json --pred-dir out/ --label pipeline
# repeat with --backend vlm-vllm for the VLM (the vLLM-on-ROCm server is started separately)
```

The lock pins: the MinerU-ROCm results commit, the upstream `mineru` / `mineru_vl_utils` git commits, model weight + ground-truth SHA256s, the OmniDocBench scorer commit, both venvs' full environment (Python / ROCm / PyTorch / vLLM / transformers), the CLI recipe, and the metric formula. Inference and scoring run in separate venvs.

> **Reproducibility scope:** the environment is pinned and the headline was reproduced on our hardware (self-attested). Independent clean-environment reproduction has not been automated, so we do not claim "fully reproducible" beyond what the lock pins.

## Known limitations

*Correctness and performance are reported separately.*

- **Performance (not correctness):** the VLM via vLLM runs correctly without source patches, but is **~15–16 s/page unpatched** on gfx1100. The existing community `docs/zh/usage/acceleration_cards/AMD.md` documents an optional Triton Conv3d performance patch; its reported timing is on a different workload and is **not directly comparable** with our OmniDocBench per-page measurement, so evaluating that patch under the same setup is outside this issue. The pipeline backend needs no patches (~3–6 s/page).
- **Empty outputs:** 2/1651 VLM pages (0.12%) and 1/1651 pipeline pages produced empty output. The VLM empties were observed alongside vLLM's EOS-first-token behavior; we have not isolated a root cause and are not attributing it to MinerU. **Why the manifests say `status: failed`:** the run manifests use a strict completeness gate — a request may complete without a process crash but remain `pending` when its final prediction is empty, and the run is marked `failed` whenever any page is `pending`. So the committed manifests report `status: failed` (pipeline `pending: 1`, VLM `pending: 2`) — these are the same empty-output cases above, **not** additional process failures (`run_counts.failed: 0`, `interrupted: 0`).
- **transformers backend:** also runs on ROCm but ~100–150 s/page — impractical for full-set eval, so no full Overall is reported for it.
- **Determinism:** the pipeline is deterministic across runs (byte-identical predictions); the VLM (vLLM, bf16) shows run-to-run drift.
- **Scope not covered:** ROCm versions other than 7.2; AMD GPUs other than gfx1100/W7900 (including other RDNA3 variants, RDNA4, MI-series); Windows; multi-GPU / tensor-parallel configurations.

## Proposed contribution (docs-only, tiered)

1. A **"Running MinerU on AMD ROCm"** page (community-validated configuration): the HSA_OVERRIDE recipe, the tested hardware/versions, install steps, and these benchmark results.
2. A note/link near the GPU-acceleration table pointing to that page, labelled **community-validated** (not official support).
3. *(Maintainer's choice)* a scoped ROCm entry in the GPU-acceleration table — we'd defer to the maintainers on whether the mainline table is the right place, given only one GPU + one ROCm version was tested.

No MinerU source changes are proposed.

## Question for maintainers

Would you prefer this as a **community-validated deployment page first** (item 1 + 2), or should we also propose a scoped ROCm entry in the GPU-acceleration table (item 3)?
