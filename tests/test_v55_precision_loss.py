"""Tests for V55 precision-focused loss (CPU torch; no GPU/caches)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load():
    spec = importlib.util.spec_from_file_location(
        "train_v55_precision_semantic", SCRIPTS / "train_v55_precision_semantic.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


torch = pytest.importorskip("torch")
v55 = _load()


def test_tversky_perfect_prediction_is_zero():
    target = torch.zeros(1, 1, 8, 8)
    target[..., 2:5, 2:5] = 1.0
    prob = target.clone()
    loss = v55.tversky_loss(torch, prob, target, 0.7, 0.3)
    assert float(loss) < 1e-3


def test_tversky_alpha_gt_beta_penalises_fp_harder_than_fn():
    target = torch.zeros(1, 1, 8, 8)
    target[..., 0:4, :] = 1.0  # top half positive

    # Case A: a false-positive blob (predict where target=0)
    fp_pred = target.clone()
    fp_pred[..., 5:7, 0:4] = 1.0
    # Case B: a false-negative hole (miss where target=1), same pixel count
    fn_pred = target.clone()
    fn_pred[..., 0:2, 0:4] = 0.0

    alpha_fp, beta_fn = 0.8, 0.2
    loss_fp = float(v55.tversky_loss(torch, fp_pred, target, alpha_fp, beta_fn))
    loss_fn = float(v55.tversky_loss(torch, fn_pred, target, alpha_fp, beta_fn))
    # with alpha>beta, the FP error must cost strictly more than an equal-size FN error
    assert loss_fp > loss_fn


def test_focal_gamma_down_weights_easy_examples():
    import torch.nn.functional as functional

    # confident-correct logits -> easy; gamma>0 should shrink the loss vs gamma=0
    logits = torch.full((1, 1, 8, 8), 6.0)  # very confident positive
    target = torch.ones(1, 1, 8, 8)
    easy_g0 = float(v55.focal_bce(torch, functional, logits, target, gamma=0.0))
    easy_g2 = float(v55.focal_bce(torch, functional, logits, target, gamma=2.0))
    assert easy_g2 < easy_g0


def test_precision_loss_returns_parts():
    import torch.nn.functional as functional

    logits = torch.randn(1, 1, 16, 16)
    target = torch.zeros(1, 1, 16, 16)
    target[..., 4:12, 4:12] = 1.0
    total, parts = v55.precision_loss(torch, functional, logits, target,
                                      cldice_w=1.0, alpha_fp=0.7, beta_fn=0.3, gamma=2.0)
    assert set(parts) == {"focal_bce", "tversky", "cldice"}
    assert float(total) > 0
    assert all(v >= 0 for v in parts.values())
