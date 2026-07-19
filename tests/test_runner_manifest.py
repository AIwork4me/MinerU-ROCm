# tests/test_runner_manifest.py
"""Manifest section tests (run-manifest + validate_manifest)."""

import json
import pytest

from mineru_rocm import runner


def test_safe_argv_redacts_secrets():
    argv = ["--gt-json", "x.json", "--hf-token", "SECRET123", "--api-key=TOPSECRET", "--ports", "8000"]
    redacted = runner.safe_argv(argv)
    assert "SECRET123" not in redacted
    assert "TOPSECRET" not in redacted
    assert "--gt-json" in redacted and "x.json" in redacted and "8000" in redacted


def test_safe_argv_no_false_positive_on_monkey():
    # 'monkey' contains 'key' substring but is not a secret flag
    redacted = runner.safe_argv(["--monkey", "tail"])
    assert redacted == ["--monkey", "tail"]


def test_write_run_manifest_structure_and_no_secret(tmp_path):
    p = runner.write_run_manifest(
        tmp_path,
        backend="vllm",
        model="HYVL",
        run_counts={"attempted": 3, "succeeded": 2, "failed": 1, "skipped": 0},
        final_state={"expected": 3, "complete": 2, "failed": 1, "pending": 0},
        ports=[8000, 8001],
        max_pixels=0,
        max_tokens=32768,
        status="failed",
    )
    m = json.loads(p.read_text("utf-8"))
    assert m["schema_version"] == runner.MANIFEST_SCHEMA_VERSION
    assert m["backend"] == "vllm" and m["status"] == "failed"
    assert m["run_counts"] == {"attempted": 3, "succeeded": 2, "failed": 1, "skipped": 0, "interrupted": 0}
    assert m["final_state"] == {"expected": 3, "complete": 2, "failed": 1, "pending": 0}
    assert m["ports"] == [8000, 8001]
    assert "timestamp_iso" in m and m["timestamp_iso"].endswith("+00:00")
    assert m["extensions"] == {}  # extra is namespaced, never top-level
    # env is best-effort and OPTIONAL — present only when the dep is installed.
    # Never assert a specific package exists (env-independent test).
    assert isinstance(m["env"], dict)
    assert isinstance(m["platform"], dict) and "python" in m["platform"]
    # no secrets: a token-bearing flag value must be redacted
    assert all("TOPSECRET" != str(v) for v in m["command"])


def test_write_run_manifest_extra_is_namespaced_and_collision_rejected(tmp_path):
    # extra lands under 'extensions', not at the top level
    p = runner.write_run_manifest(
        tmp_path,
        backend="llamacpp",
        model="HYVL",
        run_counts={"attempted": 1, "succeeded": 1, "failed": 0, "skipped": 0},
        final_state={"expected": 1, "complete": 1, "failed": 0, "pending": 0},
        extra={"endpoints": [{"alias": "p1", "state": "closed"}]},
    )
    m = json.loads(p.read_text("utf-8"))
    assert "endpoints" not in m  # not top-level
    assert m["extensions"]["endpoints"] == [{"alias": "p1", "state": "closed"}]
    assert runner.validate_manifest(m) == []
    # a reserved core key in extra must be rejected, not silently overwrite it
    with pytest.raises(ValueError, match="reserved core field"):
        runner.write_run_manifest(
            tmp_path / "x",
            backend="llamacpp",
            model="HYVL",
            run_counts={"attempted": 1, "succeeded": 1, "failed": 0, "skipped": 0},
            final_state={"expected": 1, "complete": 1, "failed": 0, "pending": 0},
            extra={"status": "ok"},  # would clobber the real status -> rejected
        )


def test_manifest_invariants_hold(tmp_path):
    # Conservation laws: attempted == succeeded+failed+interrupted;
    # expected == attempted+skipped; expected == complete+failed+pending.
    for rc, fs in [
        (
            {"attempted": 5, "succeeded": 5, "failed": 0, "skipped": 0},
            {"expected": 5, "complete": 5, "failed": 0, "pending": 0},
        ),
        (
            {"attempted": 1, "succeeded": 1, "failed": 0, "skipped": 2},
            {"expected": 3, "complete": 3, "failed": 0, "pending": 0},
        ),  # partial resume
        (
            {"attempted": 2, "succeeded": 1, "failed": 1, "skipped": 0},
            {"expected": 2, "complete": 1, "failed": 1, "pending": 0},
        ),  # retry-failed (one page failed -> status is honestly 'failed')
    ]:
        # status must match the counts: 'ok' only when nothing failed or is pending.
        status = "ok" if (fs["failed"] == 0 and fs["pending"] == 0) else "failed"
        runner.write_run_manifest(tmp_path, backend="vllm", model="m", run_counts=rc, final_state=fs, status=status)
        m = json.loads((tmp_path / "run_manifest.json").read_text("utf-8"))
        assert runner.validate_manifest(m) == [], (rc, fs)


def test_manifest_invariants_violated(tmp_path):
    # The broken pre-fix shape: expected=3,succeeded=3,skipped=2 must be rejected.
    runner.write_run_manifest(
        tmp_path,
        backend="vllm",
        model="m",
        run_counts={"attempted": 3, "succeeded": 3, "failed": 0, "skipped": 2},
        final_state={"expected": 3, "complete": 3, "failed": 0, "pending": 0},
    )
    m = json.loads((tmp_path / "run_manifest.json").read_text("utf-8"))
    errs = runner.validate_manifest(m)
    assert errs and any("expected" in e and "attempted" in e for e in errs)


def test_manifest_works_without_torch(tmp_path):
    # Generating a manifest must not require torch/transformers/vllm. This test
    # runs in both the GPU env (where env may list them) and the no-torch CI venv
    # (where env is empty); either way manifest generation + validation must work.
    runner.write_run_manifest(
        tmp_path,
        backend="llamacpp",
        model="HYVL",
        run_counts={"attempted": 1, "succeeded": 1, "failed": 0, "skipped": 0},
        final_state={"expected": 1, "complete": 1, "failed": 0, "pending": 0},
    )
    m = json.loads((tmp_path / "run_manifest.json").read_text("utf-8"))
    assert m["backend"] == "llamacpp"
    assert runner.validate_manifest(m) == []


def _valid_manifest(**overrides):
    m = {
        "schema_version": 2,
        "repo_commit": "abc123",
        "backend": "llamacpp",
        "model": "HYVL",
        "timestamp_iso": "2026-07-17T12:00:00+00:00",
        "status": "ok",
        "run_counts": {"attempted": 2, "succeeded": 2, "failed": 0, "skipped": 0, "interrupted": 0},
        "final_state": {"expected": 2, "complete": 2, "failed": 0, "pending": 0},
    }
    m.update(overrides)
    return m


def test_validate_manifest_accepts_valid():
    assert runner.validate_manifest(_valid_manifest()) == []


def test_validate_manifest_rejects_non_object():
    assert runner.validate_manifest("not a dict")  # non-empty errors list
    assert runner.validate_manifest([1, 2, 3])


def test_validate_manifest_unknown_schema_version():
    errs = runner.validate_manifest(_valid_manifest(schema_version=99))
    assert any("schema_version" in e for e in errs)


def test_validate_manifest_v1_read_compat():
    # A legacy v1 manifest (no interrupted key, extra formerly top-level) must
    # still validate on read. interrupted defaults to 0.
    m = _valid_manifest()
    m["schema_version"] = 1
    del m["run_counts"]["interrupted"]
    assert runner.validate_manifest(m) == []


def test_validate_manifest_missing_run_counts():
    m = _valid_manifest()
    del m["run_counts"]
    errs = runner.validate_manifest(m)
    assert any("run_counts" in e for e in errs)


def test_validate_manifest_missing_single_count():
    m = _valid_manifest()
    del m["run_counts"]["failed"]
    errs = runner.validate_manifest(m)
    assert any("run_counts.failed is missing" in e for e in errs)


def test_validate_manifest_rejects_string_count():
    m = _valid_manifest()
    m["run_counts"]["attempted"] = "3"
    errs = runner.validate_manifest(m)
    assert any("run_counts.attempted" in e and "non-negative integer" in e for e in errs)


def test_validate_manifest_rejects_float_count():
    m = _valid_manifest()
    m["final_state"]["complete"] = 2.0
    errs = runner.validate_manifest(m)
    assert any("final_state.complete" in e and "non-negative integer" in e for e in errs)


def test_validate_manifest_rejects_bool_count():
    # booleans are a subclass of int in Python; they must NOT count as integers.
    m = _valid_manifest()
    m["run_counts"]["succeeded"] = True
    errs = runner.validate_manifest(m)
    assert any("run_counts.succeeded" in e and "non-negative integer" in e for e in errs)


def test_validate_manifest_rejects_empty_backend_or_model():
    for bad in ("backend", "model"):
        m = _valid_manifest(**{bad: ""})
        errs = runner.validate_manifest(m)
        assert any(bad in e and "non-empty string" in e for e in errs), bad


def test_validate_manifest_rejects_unparseable_timestamp():
    m = _valid_manifest(timestamp_iso="not-a-date")
    errs = runner.validate_manifest(m)
    assert any("timestamp_iso" in e for e in errs)


def test_validate_manifest_ok_with_failed_is_invalid():
    # status == ok must imply final_state.failed == 0 and pending == 0.
    m = _valid_manifest(status="ok")
    m["final_state"]["failed"] = 1
    m["run_counts"]["failed"] = 1  # keep run_counts arithmetic honest in isolation
    errs = runner.validate_manifest(m)
    assert any("status is 'ok' but final_state.failed" in e for e in errs)


def test_validate_manifest_conservation_with_interrupted():
    # A crashed run: dispatched 5, resolved 3 (2 ok, 1 failed), 2 interrupted.
    m = _valid_manifest(
        status="crashed",
        run_counts={"attempted": 5, "succeeded": 2, "failed": 1, "skipped": 0, "interrupted": 2},
        final_state={"expected": 5, "complete": 2, "failed": 1, "pending": 2},
    )
    assert runner.validate_manifest(m) == []
    # ...but dropping interrupted makes attempted(5) != succeeded(2)+failed(1)
    m2 = _valid_manifest(
        status="crashed",
        run_counts={"attempted": 5, "succeeded": 2, "failed": 1, "skipped": 0, "interrupted": 0},
        final_state={"expected": 5, "complete": 2, "failed": 1, "pending": 2},
    )
    errs = runner.validate_manifest(m2)
    assert errs and any("attempted" in e and "succeeded" in e and "failed" in e for e in errs)
