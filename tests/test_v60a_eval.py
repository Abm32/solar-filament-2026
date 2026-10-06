"""Tests for V60A refiner eval logic (gating, AUC, verdict, selective PQ) - CPU, no torch model."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load():
    spec = importlib.util.spec_from_file_location("eval_v60a_refiner",
                                                  SCRIPTS / "eval_v60a_refiner.py")
    m = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(m)
    except Exception as e:                       # pragma: no cover
        pytest.skip(f"deps unavailable: {e}")
    return m


E = _load()
cv2 = pytest.importorskip("cv2")
F = 2048


def _box(x0, y0, x1, y1):
    m = np.zeros((F, F), np.uint8)
    m[y0:y1, x0:x1] = 1
    return m


def _per(v, rf, g, corr, conf):
    gid = "1"
    gsmall = cv2.resize(g, (E.SMALL, E.SMALL), interpolation=cv2.INTER_NEAREST).astype(bool)
    return [{"v17": [v], "refined": [rf], "corr": [corr], "conf": [conf],
             "gt_full": [(gid, g)], "gt_sets": {gid: [gsmall]}}]


def test_iou_basic():
    a = _box(0, 0, 100, 100).astype(bool)
    assert E._iou(a, a) == 1.0
    assert abs(E._iou(a, _box(0, 0, 100, 50).astype(bool)) - 0.5) < 1e-6


def test_auc_perfect_and_chance():
    assert E.auc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert E.auc([0.1, 0.2], [0.9, 0.8]) == 0.0
    assert E.auc([0.5], [0.5]) == 0.5
    assert E.auc([], [0.5]) is None


def test_candidate_table_populations():
    g = _box(100, 100, 300, 300)                      # area 40000
    v_res = _box(100, 100, 300, 190)                  # 18000/40000 = 0.45 -> reservoir
    v_good = _box(100, 100, 300, 260)                 # 32000/40000 = 0.8 -> good
    rows = E.candidate_table(_per(v_res, g, g, 0.9, 0.9) + _per(v_good, g, g, 0.1, 0.9))
    pops = sorted(r["pop"] for r in rows)
    assert pops == ["good", "reservoir"]


def test_gate_blocks_damage_and_keeps_push():
    g = _box(100, 100, 300, 300)
    v_res = _box(100, 100, 300, 190)                  # reservoir, refined fixes it
    v_good = _box(100, 100, 300, 260)                 # good, refined wrecks it
    wreck = _box(0, 0, 2000, 2000)
    per = _per(v_res, g, g, 0.95, 0.95) + _per(v_good, wreck, g, 0.05, 0.95)
    rows = E.candidate_table(per)
    ungated = E.gate_metrics(rows)
    gated = E.gate_metrics(rows, 0.5, 0.0)
    assert ungated["pushed_to_0.5"] == 1 and ungated["good_damaged"] == 1
    assert gated["pushed_to_0.5"] == 1 and gated["good_damaged"] == 0   # head protects good


def test_separability_auc_reflects_head():
    g = _box(100, 100, 300, 300)
    v_res = _box(100, 100, 300, 190)
    v_good = _box(100, 100, 300, 260)
    per = _per(v_res, g, g, 0.9, 0.9) + _per(v_good, g, g, 0.1, 0.9)
    sep = E.separability(E.candidate_table(per))
    assert sep["auc_reservoir_vs_good"] == 1.0
    assert sep["mean_corr_reservoir"] > sep["mean_corr_good"]


def test_selective_pq_uses_model_outputs_not_gt():
    g = _box(100, 100, 300, 300)
    v = _box(100, 100, 300, 190)                      # under-seg, misses @0.5
    per = _per(v, g, g, 0.9, 0.9)
    base = E.score_selective(per, cv2, None, baseline=True)
    on = E.score_selective(per, cv2, lambda v_, rf, c, f: c >= 0.5)
    off = E.score_selective(per, cv2, lambda v_, rf, c, f: c >= 0.99)
    assert base["tp"] == 0 and on["tp"] == 1 and off["tp"] == 0


def test_verdict_rules():
    sep_ok = {"auc_reservoir_vs_good": 0.9}
    sep_bad = {"auc_reservoir_vs_good": 0.5}
    good_gate = {"replace_all": {"good_damaged": 172, "pushed_to_0.5": 17},
                 "g": {"good_damaged": 5, "pushed_to_0.5": 15}}
    assert E.verdict(sep_ok, good_gate)["verdict"] == "continue"
    still_bad = {"replace_all": {"good_damaged": 172, "pushed_to_0.5": 17},
                 "g": {"good_damaged": 90, "pushed_to_0.5": 15}}
    assert E.verdict(sep_ok, still_bad)["verdict"] == "kill_still_damages_good"
    assert E.verdict(sep_bad, still_bad)["verdict"] == "kill_head_cannot_separate"
