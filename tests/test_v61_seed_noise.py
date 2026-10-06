"""Tests for the seed-noise analysis helpers (per-file PQ, delta summary, split-half)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
pytest.importorskip("pycocotools")
spec = importlib.util.spec_from_file_location("analyze_v61_seed_noise",
                                              SCRIPTS / "analyze_v61_seed_noise.py")
N = importlib.util.module_from_spec(spec)
spec.loader.exec_module(N)


def test_per_file_pq_and_nan_for_empty():
    counts = np.array([[3.0, 4, 2, 2],      # 3/(4+2) = 0.5
                       [0.0, 0, 0, 0],      # nothing: NaN
                       [2.0, 2, 0, 0]])     # 1.0
    pq = N.per_file_pq(counts)
    assert abs(pq[0] - 0.5) < 1e-12 and np.isnan(pq[1]) and abs(pq[2] - 1.0) < 1e-12


def test_summarize_delta_ignores_nan_and_counts_large_moves():
    d = np.array([0.0, 0.05, -0.3, np.nan, 0.15, 0.0])
    s = N.summarize_delta(d)
    assert s["n_files"] == 5 and s["abs_gt_0.10"] == 2 and s["abs_gt_0.20"] == 1
    assert s["share_equal"] == 0.4 and abs(s["share_aprime_better"] - 0.4) < 1e-12
