# Known gaps

Track the open items for `MinerU-ROCm` here. A `verified` badge requires these to be resolved or explicitly scoped.

- **`smoke` backend is a placeholder.** `_infer` in `adapter/run_adapter.py` raises `NotImplementedError`. Wire up real model inference before any evaluation.
- **No `verified` results.** `model_card.json` badges both platforms `community-wanted`; no `run_summary.json` / `provenance.json` are committed under `results/` yet.
- **Provisioning is stubbed.** `adapter/setup/00-install-deps.{sh,ps1}` only echo; implement weights download + runtime (ROCm EP / DirectML EP) setup.
- **No VLM server config.** `adapter/setup/.env.local.example` has empty `SERVER_URL` / `WEIGHTS_DIR`; fill these for server-based backends.
- **Windows/HIP path untested.** The `windows-hip` results dir exists but has no artifacts; validate the DirectML/ONNX path on a Windows machine before claiming it.

Remove each bullet as it is resolved.
