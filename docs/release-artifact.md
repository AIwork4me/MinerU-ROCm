# Release artifact — MinerU-ROCm v1.0.0

This release freezes the first evidence-complete community snapshot of the
Linux ROCm VLM and pipeline results plus the Windows-HIP Phase 1 pipeline.

## Source and packages

- Source tag: `v1.0.0` (annotated tag on `main`).
- Python distribution: `mineru-rocm==1.0.0`.
- Release assets: universal wheel, source archive, `SHA256SUMS`, and a generated
  release manifest containing the release commit and annotated-tag object SHA.
- Optional platform engine: OmniDocBench-ROCm 0.3.2 release wheel, pinned by
  SHA-256 in `pyproject.toml`.

Verify downloaded assets with:

```powershell
Get-FileHash -Algorithm SHA256 mineru_rocm-1.0.0-py3-none-any.whl
Get-FileHash -Algorithm SHA256 mineru_rocm-1.0.0.tar.gz
Get-Content SHA256SUMS
```

## Formal result bundles

| Model/backend | Platform | Pages | Overall | Bundle |
|---|---|---:|---:|---|
| MinerU2.5-Pro VLM | Linux ROCm | 1651 | 95.56 | `results/omnidocbench/v16/linux-rocm/` |
| MinerU 3.4.4 pipeline | Linux ROCm | 1651 | 86.48 | `results/omnidocbench/v16/linux-rocm/` |
| MinerU 3.4.4 pipeline | Windows-HIP | 1651 | 86.59 | `results/omnidocbench/v16/windows-hip/` |

The Windows VLM is not part of v1.0.0.

## Repository verification

```powershell
python -m pytest -q
python -m ruff check .
python scripts/check_repo.py
python scripts/validate_platform_artifacts.py
omnidocbench-rocm conformance .
omnidocbench-rocm validate-bundle results\omnidocbench\v16\windows-hip `
  --model-card model_card.pipeline.windows-hip.json
python -m build
python -m pip check
```

The release is a `community` result. Independent reproduction is a separate
badge-promotion gate and does not change the committed score evidence.
