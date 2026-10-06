"""Tests for the V61 Focal+Dice mask loss. Needs torch + ultralytics (skipped otherwise)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
torch = pytest.importorskip("torch")
pytest.importorskip("ultralytics")
import torch.nn.functional as functional  # noqa: E402
from ultralytics.utils import loss as U  # noqa: E402
from ultralytics.utils.ops import crop_mask  # noqa: E402

spec = importlib.util.spec_from_file_location("v61_seg_loss", SCRIPTS / "v61_seg_loss.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

UPSTREAM = U.v8SegmentationLoss.__dict__["single_mask_loss"].__func__


def _batch(n=5, h=40, w=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    pred = torch.randn(n, 32, generator=g)
    proto = torch.randn(32, h, w, generator=g)
    gt = (torch.rand(n, h, w, generator=g) > 0.7).float()
    # xyxy in MASK pixel coordinates (as calculate_segmentation_loss passes it)
    xyxy = torch.tensor([[2, 3, 30, 25], [0, 0, 39, 39], [10, 10, 20, 35],
                         [5, 5, 12, 12], [20, 1, 38, 30]], dtype=torch.float32)[:n]
    area = torch.tensor([0.30, 1.0, 0.15, 0.03, 0.33], dtype=torch.float32)[:n]
    return gt, pred, proto, xyxy, area


def _build(w_focal, w_dice):
    return S.make_single_mask_loss(torch, functional, crop_mask, UPSTREAM, w_focal, w_dice)


def test_zero_weights_equal_upstream_bitwise():
    gt, pred, proto, xyxy, area = _batch()
    ref = UPSTREAM(gt, pred, proto, xyxy, area)
    mine = _build(0.0, 0.0)(gt, pred, proto, xyxy, area)
    assert torch.equal(ref, mine)                       # exact, not allclose


def test_zero_weights_equal_upstream_many_seeds_and_grads():
    for seed in range(5):
        gt, pred, proto, xyxy, area = _batch(seed=seed)
        p1 = pred.clone().requires_grad_(True)
        p2 = pred.clone().requires_grad_(True)
        UPSTREAM(gt, p1, proto, xyxy, area).backward()
        _build(0.0, 0.0)(gt, p2, proto, xyxy, area).backward()
        assert torch.equal(p1.grad, p2.grad)


def test_terms_are_additive_and_isolated():
    gt, pred, proto, xyxy, area = _batch()
    base = _build(0.0, 0.0)(gt, pred, proto, xyxy, area)
    f = _build(1.0, 0.0)(gt, pred, proto, xyxy, area)
    d = _build(0.0, 1.0)(gt, pred, proto, xyxy, area)
    both = _build(1.0, 1.0)(gt, pred, proto, xyxy, area)
    assert f > base and d > base                        # each adds a positive term
    assert torch.allclose(both, f + d - base, rtol=1e-5)


def test_weights_scale_linearly():
    gt, pred, proto, xyxy, area = _batch()
    base = _build(0.0, 0.0)(gt, pred, proto, xyxy, area)
    d1 = _build(0.0, 1.0)(gt, pred, proto, xyxy, area) - base
    d3 = _build(0.0, 3.0)(gt, pred, proto, xyxy, area) - base
    assert torch.allclose(d3, 3 * d1, rtol=1e-4)


def test_dice_zero_for_perfect_prediction_inside_box():
    h = w = 32
    gt = torch.zeros(1, h, w)
    gt[0, 8:24, 8:24] = 1.0
    proto = torch.zeros(32, h, w)
    proto[0] = gt[0] * 40 - 20                           # logits: +20 inside, -20 outside
    pred = torch.zeros(1, 32)
    pred[0, 0] = 1.0
    xyxy = torch.tensor([[4.0, 4.0, 28.0, 28.0]])
    area = torch.tensor([0.3])
    base = _build(0.0, 0.0)(gt, pred, proto, xyxy, area)
    with_d = _build(0.0, 1.0)(gt, pred, proto, xyxy, area)
    assert float(with_d - base) < 1e-3                   # Dice ~ 0 when prediction is exact


def test_dice_penalises_wrong_overlap_more_than_perfect():
    h = w = 32
    gt = torch.zeros(1, h, w)
    gt[0, 8:24, 8:24] = 1.0
    proto_bad = torch.zeros(32, h, w)
    bad = torch.zeros(h, w)
    bad[8:24, 20:32] = 1.0                               # mostly misses the GT
    proto_bad[0] = bad * 40 - 20
    proto_ok = torch.zeros(32, h, w)
    proto_ok[0] = gt[0] * 40 - 20
    pred = torch.zeros(1, 32)
    pred[0, 0] = 1.0
    xyxy = torch.tensor([[0.0, 0.0, 32.0, 32.0]])
    area = torch.tensor([1.0])
    fn = _build(0.0, 1.0)
    assert fn(gt, pred, proto_bad, xyxy, area) > fn(gt, pred, proto_ok, xyxy, area)


def test_install_patches_class_and_zero_restores_upstream_value():
    gt, pred, proto, xyxy, area = _batch()
    ref = UPSTREAM(gt, pred, proto, xyxy, area)
    try:
        info = S.install(1.0, 1.0)
        assert info["w_focal"] == 1.0 and info["w_dice"] == 1.0
        patched = U.v8SegmentationLoss.single_mask_loss(gt, pred, proto, xyxy, area)
        assert patched > ref
        S.install(0.0, 0.0)
        assert torch.equal(U.v8SegmentationLoss.single_mask_loss(gt, pred, proto, xyxy, area),
                           ref)
    finally:
        U.v8SegmentationLoss.single_mask_loss = staticmethod(UPSTREAM)


def test_gradients_are_finite_with_extreme_logits():
    gt, pred, proto, xyxy, area = _batch()
    pred = (pred * 50).requires_grad_(True)
    out = _build(1.0, 1.0)(gt, pred, proto, xyxy, area)
    out.backward()
    assert torch.isfinite(out) and torch.isfinite(pred.grad).all()


def _build_chunk(w_focal, w_dice, chunk):
    return S.make_single_mask_loss(torch, functional, crop_mask, UPSTREAM, w_focal, w_dice,
                                   chunk=chunk)


def test_chunked_equals_unchunked_value_and_gradient():
    gt, pred, proto, xyxy, area = _batch()
    for chunk in (1, 2, 3, 16):
        p1 = pred.clone().requires_grad_(True)
        p2 = pred.clone().requires_grad_(True)
        a = _build(1.0, 1.0)(gt.clone(), p1, proto, xyxy, area)
        b = _build_chunk(1.0, 1.0, chunk)(gt.clone(), p2, proto, xyxy, area)
        a.backward()
        b.backward()
        assert torch.allclose(a, b, rtol=1e-5, atol=1e-6), chunk
        assert torch.allclose(p1.grad, p2.grad, rtol=1e-4, atol=1e-6), chunk


def test_chunked_zero_weights_still_bitwise_upstream():
    gt, pred, proto, xyxy, area = _batch()
    ref = UPSTREAM(gt, pred, proto, xyxy, area)
    assert torch.equal(ref, _build_chunk(0.0, 0.0, 2)(gt, pred, proto, xyxy, area))


def test_chunked_single_terms_match():
    gt, pred, proto, xyxy, area = _batch()
    for wf, wd in ((1.0, 0.0), (0.0, 1.0), (0.5, 2.0)):
        a = _build(wf, wd)(gt.clone(), pred, proto, xyxy, area)
        b = _build_chunk(wf, wd, 2)(gt.clone(), pred, proto, xyxy, area)
        assert torch.allclose(a, b, rtol=1e-5, atol=1e-6), (wf, wd)


def test_install_default_is_chunked():
    try:
        info = S.install(1.0, 1.0)
        assert info["chunk"] == S.DEFAULT_CHUNK and S.DEFAULT_CHUNK >= 1
    finally:
        U.v8SegmentationLoss.single_mask_loss = staticmethod(UPSTREAM)
