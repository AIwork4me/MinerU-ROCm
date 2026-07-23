# Hardware matrix

These entries describe completed, evidence-backed runs. They are not general
support claims for untested AMD architectures.

| Platform | Hardware | Backend | Software path | OmniDocBench v1.6 | Status |
|---|---|---|---|---:|---|
| `linux-rocm` | Radeon PRO W7900, gfx1100, 48 GB | MinerU2.5-Pro VLM | vLLM ROCm, bf16, one GPU | 95.56 | `community` |
| `linux-rocm` | Radeon PRO W7900, gfx1100, 48 GB | MinerU 3.4.4 pipeline | PyTorch ROCm | 86.48 | `community` |
| `windows-hip` | Ryzen AI MAX+ 395 / Radeon 8060S, shared memory | MinerU 3.4.4 pipeline | Windows ROCm PyTorch + DirectML | 86.59 | `community` |
| `windows-hip` | Not yet established | MinerU2.5-Pro VLM | Serving runtime undecided | — | `community-wanted` |

## Windows execution detail

- PyTorch 2.9.1+rocm7.2.1 reported HIP 7.2.53211 and the Radeon 8060S GPU.
- DirectML executed 3336 `model.onnx`, 486 table-classifier, and 634 `unet`
  runs with zero runtime fallback.
- `slanet-plus.onnx` executed 511 times through the documented CPU override
  because its ONNX control-flow graph is incompatible with DirectML 1.24.4.

See the platform model cards and the bundle README files under
`results/omnidocbench/v16/` for exact submetrics and artifact paths.
