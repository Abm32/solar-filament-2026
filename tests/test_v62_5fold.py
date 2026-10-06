"""Tests for the V62 5-fold pre-registered verdict (pure function)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
pytest.importorskip("pycocotools")
spec = importlib.util.spec_from_file_location("eval_v62_5fold", SCRIPTS / "eval_v62_5fold.py")
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)

GOOD = dict(d=0.015, n_noise=0.004, p_gt0=0.97, ci_low=0.003, folds_pos=5, small_gain=0.03,
            d_own=0.016, prop_ratio=1.02, fp_ratio=0.95)


def v(**kw):
    args = dict(GOOD)
    args.update(kw)
    return E.verdict(**args)


def test_all_conditions_met_continues():
    assert v() == "continue"


def test_kill_rules():
    assert v(d=0.004) == "kill"
    assert v(folds_pos=2) == "kill"
    assert v(d=-0.01, folds_pos=1) == "kill"


def test_each_continue_condition_is_necessary():
    assert v(d=0.0099, n_noise=0.001) == "inconclusive"          # below +0.010 floor
    assert v(d=0.012, n_noise=0.007) == "inconclusive"           # d < 2n
    assert v(p_gt0=0.85) == "inconclusive"
    assert v(ci_low=-0.001) == "inconclusive"
    assert v(folds_pos=3) == "inconclusive"                      # needs >= 4
    assert v(small_gain=0.0) == "inconclusive"
    assert v(d_own=-0.001) == "inconclusive"
    assert v(prop_ratio=1.31) == "inconclusive"
    assert v(fp_ratio=1.31) == "inconclusive"


def test_boundary_values():
    assert v(d=0.010, n_noise=0.004) == "continue"               # d == floor passes
    assert v(folds_pos=4) == "continue"
    assert v(prop_ratio=1.30, fp_ratio=1.30) == "continue"
    assert v(d=0.005, folds_pos=5, n_noise=0.001) == "inconclusive"   # 0.005 is not < 0.005
