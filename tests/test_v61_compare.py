"""Tests for the V61 fold comparison logic (categories, bootstrap, pre-registered verdict)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
pytest.importorskip("pycocotools")
spec = importlib.util.spec_from_file_location("eval_v61_compare", SCRIPTS / "eval_v61_compare.py")
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


def test_category_priority_matches_diagnose_v59_0():
    cut = 100.0
    assert C.categorize(5000, 6.0, 50, cut) == "thin"       # thin beats small and faint
    assert C.categorize(500, 1.5, 50, cut) == "small"       # small beats faint
    assert C.categorize(5000, 1.5, 90, cut) == "faint"
    assert C.categorize(5000, 1.5, 150, cut) == "normal"
    assert C.categorize(1000, 1.5, 150, cut) == "normal"    # area==1000 is NOT small (<1000)
    assert C.categorize(5000, 4.0, 150, cut) == "thin"      # elong==4 IS thin (>=4)


def test_gt_category_uses_512_scale_for_area():
    m = np.zeros((512, 512), bool)
    m[10:17, 10:17] = True                                   # 49 px at 512 -> 784 px full-res
    g = np.full((512, 512), 200, np.uint8)
    assert C.gt_category(m, g, 100.0) == "small"
    m2 = np.zeros((512, 512), bool)
    m2[10:30, 10:30] = True                                  # 400 px -> 6400 full-res
    assert C.gt_category(m2, g, 100.0) == "normal"


def test_faint_cut_ignores_dark_background():
    g = np.zeros((512, 512), np.uint8)
    g[100:200, 100:200] = np.arange(10000).reshape(100, 100) % 200 + 30
    cut = C.faint_cut_of(g)
    sel = g[g > 20]
    assert abs(cut - np.percentile(sel, 25)) < 1e-6 and cut > 30


def test_paired_bootstrap_detects_consistent_gain_and_null():
    rng = np.random.default_rng(0)
    base = np.stack([rng.uniform(4, 6, 80), rng.integers(8, 12, 80), rng.integers(3, 6, 80),
                     rng.integers(3, 6, 80)], 1).astype(float)
    better = base.copy()
    better[:, 1] += 2                                        # +2 TP every file
    better[:, 3] -= 1
    better[:, 0] += 1.4
    gain = C.paired_bootstrap(base, better)
    assert gain["delta"] > 0 and gain["ci95"][0] > 0 and gain["p_gt_0"] > 0.99
    null = C.paired_bootstrap(base, base)
    assert abs(null["delta"]) < 1e-12 and null["ci95"][0] <= 0 <= null["ci95"][1]


def test_verdict_rules_match_preregistration():
    v = C.verdict
    # kill: small gain, or not above noise
    assert v(0.004, 0.001, 0.99, 0.004, 0.05, 0.0) == "kill"
    assert v(0.012, 0.014, 0.99, 0.012, 0.05, 0.0) == "kill"             # d < n
    # continue needs everything
    assert v(0.015, 0.005, 0.95, 0.010, 0.05, 0.01) == "continue"
    assert v(0.015, 0.005, 0.85, 0.010, 0.05, 0.01) == "inconclusive"    # P(d>0) < 0.90
    assert v(0.015, 0.009, 0.95, 0.010, 0.05, 0.01) == "inconclusive"    # d < 2n
    assert v(0.015, 0.005, 0.95, -0.001, 0.05, 0.01) == "inconclusive"   # threshold-shift artefact
    assert v(0.015, 0.005, 0.95, 0.010, 0.00, 0.03) == "inconclusive"    # fine < normal
    assert v(0.0075, 0.001, 0.99, 0.0075, 0.05, 0.0) == "inconclusive"   # between 0.005 and 0.010


def test_summarize_pools_counts_and_recall():
    def pf(counts, hits, tot):
        return {0.3: {"counts": np.asarray(counts, float), "hits": hits, "tot": tot}}
    z = dict.fromkeys(C.CATS, 0)
    a = pf([3.0, 4, 2, 1], {**z, "thin": 1, "normal": 3}, {**z, "thin": 2, "normal": 3})
    b = pf([1.0, 2, 0, 2], {**z, "thin": 1, "normal": 1}, {**z, "thin": 2, "normal": 2})
    s = C.summarize([a, b], 0.3)
    assert s["tp"] == 6 and s["fp"] == 2 and s["fn"] == 3
    assert s["recall"]["thin"] == 0.5 and s["recall"]["normal"] == 0.8
    assert s["recall"]["small"] is None
    assert abs(s["mean_matched_iou"] - 4.0 / 6) < 1e-9
