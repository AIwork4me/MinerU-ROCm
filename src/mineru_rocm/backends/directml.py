# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AIwork4me
"""DirectML policy for MinerU's native-Windows ONNX sub-models."""
from __future__ import annotations

from pathlib import Path
from typing import Any

DML_PROVIDER = "DmlExecutionProvider"
CPU_PROVIDER = "CPUExecutionProvider"
CPU_ONLY_MODELS = frozenset({"slanet-plus.onnx"})

_active_providers: list[str] = []
_fallback_count = 0
_fallback_reasons: list[str] = []
_fallback_events: list[dict[str, object]] = []
_run_counts_by_model: dict[str, int] = {}
_cpu_override_models_active: set[str] = set()
_cpu_override_run_counts_by_model: dict[str, int] = {}


def _model_label(args: tuple, kwargs: dict) -> str:
    """Return a stable, non-sensitive label for an ORT model source."""
    source = args[0] if args else kwargs.get("path_or_bytes", "<unknown>")
    if isinstance(source, (str, Path)):
        return Path(source).name
    if isinstance(source, (bytes, bytearray)):
        return f"<model-bytes:{len(source)}>"
    return f"<{type(source).__name__}>"


def _input_signature(args: tuple, kwargs: dict) -> dict[str, object]:
    """Describe an ORT input feed without recording tensor contents."""
    feed = args[1] if len(args) > 1 else kwargs.get("input_feed")
    if not isinstance(feed, dict):
        return {}
    signature: dict[str, object] = {}
    for name, value in feed.items():
        shape = getattr(value, "shape", None)
        signature[str(name)] = {
            "shape": list(shape) if shape is not None else None,
            "dtype": str(getattr(value, "dtype", type(value).__name__)),
        }
    return signature


def _exception_metadata(exc: Exception) -> dict[str, object]:
    """Preserve native ORT error bytes when pybind decodes them incorrectly."""
    metadata: dict[str, object] = {
        "exception_type": type(exc).__name__,
        "exception": str(exc),
    }
    if isinstance(exc, UnicodeDecodeError):
        raw = bytes(exc.object)
        metadata["exception_bytes_hex"] = raw.hex()
        try:
            metadata["exception_decoded_gbk"] = raw.decode("gbk")
        except UnicodeDecodeError:
            pass
    return metadata


def configure_onnxruntime_directml(
    ort_module: Any | None = None, *, device_id: int = 0
) -> dict[str, object]:
    """Prefer DirectML for compatible ORT sessions, with audited overrides.

    MinerU 3.4 creates its table ONNX sessions through multiple code paths,
    including one that does not pass providers at all. Patching the shared
    ``InferenceSession`` initializer before MinerU imports those paths applies
    one fail-closed Windows policy consistently.
    """
    if ort_module is None:
        import onnxruntime as ort_module

    available = list(ort_module.get_available_providers())
    if DML_PROVIDER not in available:
        raise RuntimeError(
            "windows-hip requires ONNX Runtime DirectML, but "
            "DmlExecutionProvider is unavailable. Install "
            "onnxruntime-directml after mineru[pipeline]."
        )

    session_cls = ort_module.InferenceSession
    if not getattr(session_cls, "_mineru_rocm_directml_patched", False):
        original_init = session_cls.__init__
        original_run = session_cls.run

        def directml_init(self, *args, **kwargs):
            model_label = _model_label(args, kwargs)
            force_cpu = kwargs.pop("_mineru_rocm_force_cpu", False)
            policy_cpu = (
                not force_cpu and model_label.lower() in CPU_ONLY_MODELS
            )
            session_options = kwargs.get("sess_options")
            if session_options is None:
                session_options = ort_module.SessionOptions()
                kwargs["sess_options"] = session_options
            session_options.enable_mem_pattern = False
            session_options.execution_mode = ort_module.ExecutionMode.ORT_SEQUENTIAL

            kwargs.pop("provider_options", None)
            if force_cpu or policy_cpu:
                kwargs["providers"] = [CPU_PROVIDER]
                original_init(self, *args, **kwargs)
                self._mineru_rocm_is_cpu_fallback = force_cpu
                self._mineru_rocm_is_cpu_override = policy_cpu
                self._mineru_rocm_model_label = model_label
                if policy_cpu:
                    _cpu_override_models_active.add(model_label)
                return

            kwargs["providers"] = [
                (DML_PROVIDER, {"device_id": str(device_id)}),
                CPU_PROVIDER,
            ]
            original_init(self, *args, **kwargs)
            self._mineru_rocm_model_label = model_label

            active = list(self.get_providers())
            if not active or active[0] != DML_PROVIDER:
                raise RuntimeError(
                    "DmlExecutionProvider failed to activate as the first "
                    f"provider; active providers: {active}"
                )
            _active_providers[:] = active

            cpu_kwargs = dict(kwargs)
            cpu_kwargs["_mineru_rocm_force_cpu"] = True
            self._mineru_rocm_cpu_fallback_session = session_cls(
                *args, **cpu_kwargs
            )

        def directml_run(self, *args, **kwargs):
            global _fallback_count
            if getattr(self, "_mineru_rocm_is_cpu_fallback", False):
                return original_run(self, *args, **kwargs)
            model_label = getattr(
                self, "_mineru_rocm_model_label", "<unknown>"
            )
            if getattr(self, "_mineru_rocm_is_cpu_override", False):
                _cpu_override_run_counts_by_model[model_label] = (
                    _cpu_override_run_counts_by_model.get(model_label, 0) + 1
                )
                return original_run(self, *args, **kwargs)
            _run_counts_by_model[model_label] = (
                _run_counts_by_model.get(model_label, 0) + 1
            )
            try:
                return original_run(self, *args, **kwargs)
            except Exception as exc:
                cpu_session = getattr(
                    self, "_mineru_rocm_cpu_fallback_session", None
                )
                if cpu_session is None:
                    raise
                _fallback_count += 1
                reason = f"{type(exc).__name__}: {exc}"
                if reason not in _fallback_reasons:
                    _fallback_reasons.append(reason)
                event = {
                    "model": model_label,
                    "inputs": _input_signature(args, kwargs),
                    "providers": list(self.get_providers()),
                }
                event.update(_exception_metadata(exc))
                _fallback_events.append(event)
                return original_run(cpu_session, *args, **kwargs)

        session_cls.__init__ = directml_init
        session_cls.run = directml_run
        session_cls._mineru_rocm_directml_patched = True

    return {
        "onnxruntime_provider_requested": "directml",
        "onnxruntime_providers_available": available,
        "onnxruntime_cpu_fallback_enabled": True,
        "onnxruntime_cpu_overrides_configured": sorted(CPU_ONLY_MODELS),
    }


def directml_runtime_metadata() -> dict[str, object]:
    """Return provider evidence collected from initialized ORT sessions."""
    return {
        "onnxruntime_providers_active": list(_active_providers),
        "onnxruntime_directml_fallback_count": _fallback_count,
        "onnxruntime_directml_fallback_reasons": list(_fallback_reasons),
        "onnxruntime_directml_fallback_events": list(_fallback_events),
        "onnxruntime_directml_run_counts_by_model": dict(
            _run_counts_by_model
        ),
        "onnxruntime_cpu_overrides_active": sorted(
            _cpu_override_models_active
        ),
        "onnxruntime_cpu_override_run_counts_by_model": dict(
            _cpu_override_run_counts_by_model
        ),
    }
