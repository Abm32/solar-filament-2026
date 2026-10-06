"""Tests for V54 gate math (pure numpy; no GPU/caches)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load():
    spec = importlib.util.spec_from_file_location("fit_v54_gate", SCRIPTS / "fit_v54_gate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate = _load()


def test_sigmoid_bounds_and_center():
    assert gate.sigmoid(0.0) == pytest.approx(0.5)
    assert 0.0 < gate.sigmoid(-50) < 1e-10
    assert 1.0 - gate.sigmoid(50) < 1e-10


def test_auc_perfect_and_random():
    y = np.array([0, 0, 1, 1])
    assert gate.auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert gate.auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == pytest.approx(0.0)
    # ties -> 0.5
    assert gate.auc(y, np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.5)


def test_logistic_separates_linearly_separable_data():
    rng = np.random.default_rng(0)
    n = 400
    x_pos = rng.normal(2.0, 0.5, (n // 2, 2))
    x_neg = rng.normal(-2.0, 0.5, (n // 2, 2))
    x = np.vstack([x_pos, x_neg])
    y = np.concatenate([np.ones(n // 2), np.zeros(n // 2)])
    mean, std = x.mean(0), x.std(0)
    z = (x - mean) / std
    w, b = gate.fit_logistic(z, y, iters=2000)
    scores = gate.sigmoid(z @ w + b)
    assert gate.auc(y, scores) > 0.99


def test_auc_handles_single_class():
    y = np.array([1, 1, 1])
    assert np.isnan(gate.auc(y, np.array([0.1, 0.2, 0.3])))
