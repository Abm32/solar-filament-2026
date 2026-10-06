"""Tests for V60A refiner pure logic (spine raster, loss, seed mixture, correction label)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load():
    spec = importlib.util.spec_from_file_location("train_v60a_refiner",
                                                  SCRIPTS / "train_v60a_refiner.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


torch = pytest.importorskip("torch")
cv2 = pytest.importorskip("cv2")
v60a = _load()


def _blob(h=64, w=64):
    m = np.zeros((h, w), np.uint8)
    m[20:28, 8:56] = 1                       # thin horizontal filament, area 384
    return m


def test_rasterize_spine_in_crop_frame():
    net, crop = 256, 1024
    spine = [100, 200, 400, 200]
    hm = v60a.rasterize_spine(np, cv2, spine, 100, 100, net / crop, net)
    assert hm.shape == (net, net)
    assert hm.max() <= 1.0 + 1e-6
    assert hm.sum() > 0
    assert hm[25].sum() > hm[200].sum()


def test_losses_perfect_prediction_low():
    import torch.nn.functional as functional
    mt = torch.zeros(1, 1, 32, 32)
    mt[..., 8:24, 8:24] = 1.0
    st = torch.zeros(1, 1, 32, 32)
    mlogit = (mt * 20 - 10)
    total, dice, _focal = v60a.losses(torch, functional, mlogit, mlogit, mt, st, spine_w=0.0)
    assert dice < 0.05
    assert float(total) < 0.2


def test_corr_term_adds_loss_only_when_weighted():
    import torch.nn.functional as functional
    mt = torch.zeros(1, 1, 16, 16)
    mt[..., 4:12, 4:12] = 1.0
    ml = torch.randn(1, 1, 16, 16)
    st = torch.zeros(1, 1, 16, 16)
    cl = torch.tensor([2.0])
    ct = torch.tensor([0.0])
    t0, _, _ = v60a.losses(torch, functional, ml, ml, mt, st, 0.0,
                           corr_logit=cl, corr_tar=ct, corr_w=0.0)
    t1, _, _ = v60a.losses(torch, functional, ml, ml, mt, st, 0.0,
                           corr_logit=cl, corr_tar=ct, corr_w=0.5)
    assert float(t1) > float(t0)


def test_identity_seed_equals_mask_and_label_zero():
    rng = np.random.default_rng(0)
    m = _blob()
    kinds = {}
    for _ in range(400):
        seed, kind, label = v60a.sample_seed(np, cv2, m, rng)
        kinds.setdefault(kind, []).append((seed, label))
    assert set(kinds) == {"identity", "mild", "regime", "severe"}
    for seed, label in kinds["identity"]:
        assert (seed == m).all() and label == 0.0           # good mask -> preserve
    frac_id = len(kinds["identity"]) / 400
    assert 0.40 < frac_id < 0.60                              # ~50% identity


def test_seed_is_subset_and_label_matches_iou():
    rng = np.random.default_rng(1)
    m = _blob()
    for _ in range(200):
        seed, _kind, label = v60a.sample_seed(np, cv2, m, rng)
        assert seed.sum() >= 1
        assert (seed.astype(bool) & ~m.astype(bool)).sum() == 0   # strict subset of GT
        iou = v60a.mask_iou(np, seed, m)
        assert label == (1.0 if iou < v60a.CORR_IOU_THR else 0.0)


def test_regime_seed_area_near_069_and_needs_correction():
    rng = np.random.default_rng(2)
    m = _blob()
    ratios, labels = [], []
    for _ in range(400):
        seed, kind, label = v60a.sample_seed(np, cv2, m, rng)
        if kind == "regime":
            ratios.append(seed.sum() / m.sum())
            labels.append(label)
    assert 0.6 < float(np.median(ratios)) < 0.8
    assert np.mean(labels) > 0.9              # the measured V17 regime needs correction


def test_model_has_three_outputs_and_corr_shape():
    model = v60a.build_unet(torch, corr_head=True)
    x = torch.randn(2, 2, 64, 64)
    ml, sl, cl = model(x)
    assert ml.shape == (2, 1, 64, 64) and sl.shape == (2, 1, 64, 64)
    assert cl.shape == (2,)
