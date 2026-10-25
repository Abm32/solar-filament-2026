"""Tests for V60A refiner pure logic (spine rasterization, loss) — CPU."""
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
v60a = _load()


def test_rasterize_spine_in_crop_frame():
    import cv2
    # spine is a horizontal line in full coords; crop at ox=100,oy=100, scale net/crop
    net, crop = 256, 1024
    spine = [100, 200, 400, 200]  # (100,200)->(400,200) in full coords
    hm = v60a.rasterize_spine(np, cv2, spine, 100, 100, net / crop, net)
    assert hm.shape == (net, net)
    assert hm.max() <= 1.0 + 1e-6
    assert hm.sum() > 0                      # line rendered
    # y=200 full -> (200-100)*256/1024 = 25 in crop; row ~25 should be hot
    assert hm[25].sum() > hm[200].sum()


def test_losses_perfect_prediction_low():
    import torch.nn.functional as functional
    mt = torch.zeros(1, 1, 32, 32)
    mt[..., 8:24, 8:24] = 1.0
    st = torch.zeros(1, 1, 32, 32)
    mlogit = (mt * 20 - 10)                  # confident-correct logits
    total, dice, focal = v60a.losses(torch, functional, mlogit, mlogit, mt, st, spine_w=0.0)
    assert dice < 0.05
    assert float(total) < 0.2


def test_spine_weight_adds_term():
    import torch.nn.functional as functional
    mt = torch.zeros(1, 1, 16, 16)
    mt[..., 4:12, 4:12] = 1.0
    st = torch.zeros(1, 1, 16, 16)
    st[..., 8, 4:12] = 1.0
    ml = torch.randn(1, 1, 16, 16)
    sl = torch.randn(1, 1, 16, 16)
    t0, _, _ = v60a.losses(torch, functional, ml, sl, mt, st, spine_w=0.0)
    t2, _, _ = v60a.losses(torch, functional, ml, sl, mt, st, spine_w=0.3)
    assert float(t2) != float(t0)            # spine term changes the loss when weighted
