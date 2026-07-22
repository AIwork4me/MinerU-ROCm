# tests/test_runner_lock.py
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
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
    # after release, a fresh acquire must succeed (the OS lock was released)
    with acquire_run_lock(tmp_path):
        assert (tmp_path / ".run.lock").is_file()


def test_run_lock_cross_process_crash_auto_releases(tmp_path):
    child_code = """
import sys
import time
from mineru_rocm.runner import RunLock

RunLock(sys.argv[1], command="cross-process-holder").acquire()
print("READY", flush=True)
time.sleep(60)
"""
    env = os.environ.copy()
    src = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (src, env.get("PYTHONPATH"))))
    child = subprocess.Popen(
        [sys.executable, "-c", child_code, str(tmp_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        assert child.stdout is not None
        ready = child.stdout.readline().strip()
        if ready != "READY":
            assert child.stderr is not None
            pytest.fail(f"lock-holder child failed before ready: {child.stderr.read()}")
        with pytest.raises(RunLockHeld):
            RunLock(tmp_path, command="contender").acquire()

        # Simulate an ungraceful crash: no RunLock.release() and stale metadata
        # remains. The kernel-managed lock must nevertheless become acquirable.
        child.kill()
        child.wait(timeout=10)
        with acquire_run_lock(tmp_path, command="after-crash"):
            assert (tmp_path / ".run.lock").is_file()
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)
