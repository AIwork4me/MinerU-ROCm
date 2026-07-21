import pytest

# This validator requires the omnidocbench-rocm engine; skip in core-only envs.
pytest.importorskip("omnidocbench_rocm")


def test_validate_platform_artifacts_conformant():
    """The committed v16/linux-rocm bundles validate against model_card.json."""
    import scripts.validate_platform_artifacts as v
    rc = v.main()
    assert rc == 0
