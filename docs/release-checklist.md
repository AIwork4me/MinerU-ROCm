# Release checklist

Use this checklist for MinerU-ROCm releases. Checked items are required before
publishing a release tag; GPU inference is reused only when its formal bundle
already passes integrity and provenance validation.

## Source and metadata

- [x] Package, import, citation, and changelog versions agree.
- [x] Release branch starts from a clean, synchronized `main`.
- [x] User-facing claims distinguish `community` from `verified`.
- [x] Known gaps retain all intentionally deferred work.

## Evidence

- [x] Dataset revision and GT SHA-256 are committed.
- [x] Model cards point to self-contained committed bundles.
- [x] Prediction counts, failures, and fallback policy are recorded.
- [x] Windows DirectML activity and the explicit CPU override are auditable.
- [x] Platform bundle validation and repository conformance pass.

## Quality and packaging

- [x] Full CPU test suite passes.
- [x] Ruff, REUSE, repository gates, and `pip check` pass.
- [x] Wheel and source archive build in an isolated build environment.
- [x] Built wheel metadata reports version 1.0.0 and the expected Python range.
- [x] `SHA256SUMS` is generated from the final release assets.

## Publication

- [x] Changes are reviewed through a green GitHub pull request.
- [x] An annotated `v1.0.0` tag targets the merged release commit.
- [x] GitHub Release contains wheel, source archive, checksum file, and release
  manifest.
- [ ] Independent maintainer reproduction completed; until then badges remain
  `community` rather than `verified`.
