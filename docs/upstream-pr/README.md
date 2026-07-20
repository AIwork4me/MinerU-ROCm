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
- 95.46 VLM is 0.16 pp from the vlm-engine 95.30 anchor — consistency with the published range, **not** a controlled CUDA-vs-ROCm comparison (the upstream table does not pin identical hardware, build, or decoding config). Framed as consistency, not parity or superiority.

## Process gate
Do NOT open until a maintainer responds to the #5288 comment (`issue-5288-comment.md`). Match sign-off/DCO conventions from a recently-merged doc PR (no CONTRIBUTING.md exists upstream).
