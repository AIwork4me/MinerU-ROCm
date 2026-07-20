<!-- Post this as a comment on issue #5288 BEFORE opening the PR. Wait for maintainer signal. -->

Thanks for the earlier discussion. We've prepared the docs-only contribution and would like a quick steer before opening the PR:

1. **`docs/zh/usage/acceleration_cards/AMD.md`** — add a new "gfx1100 (RDNA3) — community-verified" section (the existing perf-patch content stays untouched). It covers the `HSA_OVERRIDE` recipe and full-set OmniDocBench v1.6 numbers (pipeline 86.48, VLM 95.46), scoped honestly: "no patches needed" is about **correctness**; the unpatched VLM is ~15–16 s/page and we cross-reference the existing Triton patch for speed.
2. **`README.md`** — extend the GPU-Acceleration **row only** to mention AMD ROCm (gfx1100/RDNA3); the Accuracy row is unchanged.
3. **English mirror** — the acceleration_cards family is currently zh-only; would you like an `en/usage/acceleration_cards/AMD.md` mirror in the same PR, or keep it zh-only for consistency?

Two questions: (a) is this scope welcome as one docs-only PR? (b) any sign-off/DCO convention we should follow (we didn't find a CONTRIBUTING.md)? Full reproducibility lock: https://github.com/AIwork4me/MinerU-ROCm/blob/main/reproducibility.lock.yaml
