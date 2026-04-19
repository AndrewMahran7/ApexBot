"""Backward compatibility shim - canonical code is in scripts/run_live.py."""
from scripts.run_live import *  # noqa: F401,F403

if __name__ == "__main__":
    from scripts.run_live import main as _main
    _main()