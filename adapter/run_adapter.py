"""omnidocbench-rocm platform shim — the engine invokes this as a subprocess.

Thin entry that delegates to mineru_rocm.dispatcher, preserving the engine's
adapter contract (runnable script, same CLI). The real logic lives in the
mineru_rocm package; this shim exists only so `python adapter/run_adapter.py …`
keeps working for the optional [platform] integration.
"""
import sys
from pathlib import Path

# Allow running as a bare script (no parent package): put src/ on sys.path.
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mineru_rocm.dispatcher import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
