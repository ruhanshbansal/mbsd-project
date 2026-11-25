"""Compatibility entry point. Prefer python -m branch_prediction_sim."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from branch_prediction_sim.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
