# tests/test_runner_lock.py
import json
import pytest
from mineru_rocm import runner
from mineru_rocm.runner import RunLock, RunLockHeld, acquire_run_lock


def test_run_lock_acquire_and_release(tmp_path):
    lock = RunLock(tmp_path)
    lock.acquire()
    assert (tmp_path / ".run.lock").is_file()
    # the lock file records holder metadata as JSON
    info = json.loads((tmp_path / ".run.lock").read_text("utf-8"))
    assert {"pid", "host", "started_iso", "command"} <= set(info)
    lock.release()
    assert not (tmp_path / ".run.lock").exists()


def test_run_lock_context_manager(tmp_path):
    with acquire_run_lock(tmp_path, command=["predict", "--x", "1"]) as lock:
        assert (tmp_path / ".run.lock").is_file()
    assert not (tmp_path / ".run.lock").exists()  # released on exit


def test_run_lock_second_acquire_raises(tmp_path):
    with acquire_run_lock(tmp_path):
        second = RunLock(tmp_path)
        with pytest.raises(RunLockHeld):
            second.acquire()


def test_run_lock_reacquire_after_release(tmp_path):
    with acquire_run_lock(tmp_path):
        pass
    # after release, a fresh acquire must succeed (flock auto-released)
    with acquire_run_lock(tmp_path):
        assert (tmp_path / ".run.lock").is_file()
