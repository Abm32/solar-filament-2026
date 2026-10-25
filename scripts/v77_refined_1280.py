"""V77: p2first_1280 proposals through the r3 refiner (V75 harness, single member). CPU only.
Pre-registration: docs/v77_refined_1280_preregistration_2026-10-11.md
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v75_ensemble_refined as V  # noqa: E402

V.MEMBERS = ["p2first_1280"]
V.OUT = V.E.ROOT / "outputs/v77"

if __name__ == "__main__":
    V.main("run")
