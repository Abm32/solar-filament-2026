"""V54 step 4: re-run the V53 fusion grid with the gate-RECALIBRATED V52 scores.

Thin wrapper over v53_fusion: identical pipeline, identical control (bit-identical to
V17), but EXTRA_SETS points at the gate-rescored tune dumps (scores = P(TP) from the
FIT-only v54 gate instead of mean-probability). The hypothesis under test: a better
per-proposal score makes the recall path (beta>0) net-positive.

Compares, in one grid: base32_t040 (original mean-prob score, = V53 control arm) vs
base32_gate (recalibrated). Same output dir semantics, written to outputs/v54_fusion.

  CUDA_VISIBLE_DEVICES=N python scripts/v54_fusion.py eval  --shard N
  python scripts/v54_fusion.py merge
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import v53_fusion as v53  # noqa: E402

# Recalibrated set + the original (mean-prob) set as an in-grid baseline.
v53.EXTRA_SETS = ("base32_t040", "base32_gate")
v53.OUTPUT = v53.ROOT / "outputs/v54_fusion"

if __name__ == "__main__":
    v53.main()
