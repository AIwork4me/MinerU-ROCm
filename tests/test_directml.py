"""Windows DirectML provider policy tests (no real ORT or GPU required)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from mineru_rocm.backends.directml import (
    configure_onnxruntime_directml,
    directml_runtime_metadata,
)


def _fake_ort(available):
    class SessionOptions:
        enable_mem_pattern = True
        execution_mode = "parallel"

    class InferenceSession:
        def __init__(self, model, sess_options=None, providers=None, **kwargs):
            self.options = sess_options
            self.requested = providers

        def get_providers(self):
            return [p[0] if isinstance(p, tuple) else p for p in self.requested]

        def run(self, *args, **kwargs):
            if self.get_providers()[0] == "DmlExecutionProvider":
                raw = b"HRESULT 80070057: \xb2\xce\xca\xfd\xb4\xed\xce\xf3\xa1\xa3"
                start = raw.index(b"\xb2")
                raise UnicodeDecodeError(
                    "utf-8", raw, start, start + 1, "bad"
                )
            return ["cpu-result"]

    return SimpleNamespace(
        get_available_providers=lambda: list(available),
        InferenceSession=InferenceSession,
        SessionOptions=SessionOptions,
        ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL="sequential"),
    )


def test_directml_is_required():
    ort = _fake_ort(["CPUExecutionProvider"])
    with pytest.raises(RuntimeError, match="onnxruntime-directml"):
        configure_onnxruntime_directml(ort)


def test_directml_is_first_for_all_sessions():
    ort = _fake_ort(["DmlExecutionProvider", "CPUExecutionProvider"])
    metadata = configure_onnxruntime_directml(ort, device_id=0)

    session = ort.InferenceSession(
        "table.onnx", providers=["CPUExecutionProvider"]
    )

    assert metadata["onnxruntime_provider_requested"] == "directml"
    assert session.get_providers() == [
        "DmlExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert session.options.enable_mem_pattern is False
    assert session.options.execution_mode == "sequential"
    assert directml_runtime_metadata()["onnxruntime_providers_active"][0] == (
        "DmlExecutionProvider"
    )


def test_slanet_uses_explicit_cpu_override_without_runtime_fallback():
    ort = _fake_ort(["DmlExecutionProvider", "CPUExecutionProvider"])
    configured = configure_onnxruntime_directml(ort)
    session = ort.InferenceSession("SLANet-Plus.onnx")

    assert session.get_providers() == ["CPUExecutionProvider"]
    assert session.run(None, {"x": 1}) == ["cpu-result"]
    assert configured["onnxruntime_cpu_overrides_configured"] == [
        "slanet-plus.onnx"
    ]

    metadata = directml_runtime_metadata()
    assert "SLANet-Plus.onnx" in metadata[
        "onnxruntime_cpu_overrides_active"
    ]
    assert metadata["onnxruntime_cpu_override_run_counts_by_model"][
        "SLANet-Plus.onnx"
    ] >= 1


def test_directml_runtime_failure_is_counted_and_retried_on_cpu():
    ort = _fake_ort(["DmlExecutionProvider", "CPUExecutionProvider"])
    configure_onnxruntime_directml(ort)
    session = ort.InferenceSession("table.onnx")

    assert session.run(
        None, {"x": np.zeros((1, 3, 224, 224), dtype=np.float32)}
    ) == ["cpu-result"]
    metadata = directml_runtime_metadata()
    assert metadata["onnxruntime_directml_fallback_count"] >= 1
    assert "UnicodeDecodeError" in metadata[
        "onnxruntime_directml_fallback_reasons"
    ][-1]
    event = metadata["onnxruntime_directml_fallback_events"][-1]
    assert event["model"] == "table.onnx"
    assert event["inputs"]["x"] == {
        "shape": [1, 3, 224, 224],
        "dtype": "float32",
    }
    assert event["providers"][0] == "DmlExecutionProvider"
    assert "80070057" in event["exception_decoded_gbk"]
    assert event["exception_decoded_gbk"].endswith("参数错误。")
    assert metadata["onnxruntime_directml_run_counts_by_model"][
        "table.onnx"
    ] >= 1
