# Getting help

| Question type | Where |
|---|---|
| Bug or unexpected result in THIS repo | [GitHub Issues](https://github.com/AIwork4me/MinerU-ROCm/issues) — include the OmniDocBench page id, the backend, and the `reproducibility.lock.yaml` environment block. |
| ROCm / gfx1100 compatibility | Open an issue with `rocm-smi --showproductname` output; mark it `rocm-compat`. |
| Upstream MinerU behavior (model quality, pipeline options) | [opendatalab/MinerU](https://github.com/opendatalab/MinerU) — this repo wraps upstream, it does not change the models. |
| OmniDocBench scoring / metrics | [opendatalab/OmniDocBench](https://github.com/opendatalab/OmniDocBench). |
| Security | See [SECURITY.md](SECURITY.md) — do not open a public issue. |

Before filing, please run `python scripts/check_deps.py` (P0) / the future
`mineru-rocm doctor` (P1) and include its output.
