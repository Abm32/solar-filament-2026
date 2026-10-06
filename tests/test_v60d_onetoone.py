"""Tests for the V60D one-to-one assignment (synthetic boxes; torch CPU, no pretrained weights)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
torch = pytest.importorskip("torch")
tv = pytest.importorskip("torchvision")


def _load():
    spec = importlib.util.spec_from_file_location("train_v60d_onetoone",
                                                  SCRIPTS / "train_v60d_onetoone.py")
    m = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(m)
    except Exception as e:                       # pragma: no cover
        pytest.skip(f"deps unavailable: {e}")
    return m


D = _load()


def _roi_heads():
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
    from torchvision.models.detection.roi_heads import RoIHeads
    from torchvision.ops import MultiScaleRoIAlign
    pool = MultiScaleRoIAlign(["0"], 7, 2)
    head = TwoMLPHead(256 * 49, 64)
    return RoIHeads(pool, head, FastRCNNPredictor(64, 2), 0.5, 0.5, 64, 0.5, None,
                    0.05, 0.5, 100)


def _fixture():
    gt = torch.tensor([[100., 100., 300., 200.], [500., 500., 560., 540.]])
    # 3 fg proposals near GT0 (IoU 1.0 / ~0.86 / ~0.74), 1 near GT1, 2 background
    props = torch.tensor([[100., 100., 300., 200.], [110., 100., 300., 200.],
                          [120., 100., 300., 200.], [500., 500., 560., 540.],
                          [900., 900., 950., 950.], [0., 0., 30., 30.]])
    targets = [{"boxes": gt, "labels": torch.ones(2, dtype=torch.int64)}]
    return gt, props, targets


def test_duplicate_mask_marks_non_canonical_only():
    from torchvision.ops import box_iou
    gt, props, _ = _fixture()
    rh = _roi_heads()
    allp = rh.add_gt_proposals([props], [gt])[0]            # props + 2 appended GT boxes
    matched, labels = rh.assign_targets_to_proposals([allp], [gt],
                                                     [torch.ones(2, dtype=torch.int64)])
    dup = D.duplicate_mask(torch, box_iou, gt, allp, matched[0], labels[0])
    n_rpn = props.shape[0]
    # proposal 0 (IoU 1.0) canonical; 1,2 duplicates; proposal 3 canonical for GT1
    assert dup[:n_rpn].tolist() == [False, True, True, False, False, False]
    assert not dup[n_rpn:].any()                             # appended GT boxes never duplicates


def test_adjacent_gt_not_treated_as_duplicate():
    from torchvision.ops import box_iou
    gt = torch.tensor([[100., 100., 200., 200.], [205., 100., 305., 200.]])   # neighbours
    props = torch.tensor([[100., 100., 200., 200.], [205., 100., 305., 200.]])
    rh = _roi_heads()
    allp = rh.add_gt_proposals([props], [gt])[0]
    matched, labels = rh.assign_targets_to_proposals([allp], [gt],
                                                     [torch.ones(2, dtype=torch.int64)])
    dup = D.duplicate_mask(torch, box_iou, gt, allp, matched[0], labels[0])
    assert not dup.any()


def test_select_training_samples_matches_torchvision_when_neutral():
    """Single proposal per GT -> no duplicates -> my override == torchvision exactly."""
    from torchvision.ops import box_iou
    gt, props, targets = _fixture()
    props = props[[0, 3, 4, 5]]                               # drop the duplicate proposals
    rh = _roi_heads()
    torch.manual_seed(0)
    ref = rh.select_training_samples([props.clone()], targets)
    fn = D.make_select_training_samples(torch, box_iou, dup_w=1.0, margin=0.0)
    torch.manual_seed(0)
    mine = fn(rh, [props.clone()], targets)
    for a, b in zip(ref, mine, strict=True):
        assert torch.equal(a[0], b[0]) if not isinstance(a, list) else True
    assert torch.equal(ref[0][0], mine[0][0])
    assert torch.equal(ref[1][0], mine[1][0])
    assert torch.equal(ref[2][0], mine[2][0])
    assert torch.allclose(ref[3][0], mine[3][0])
    assert int((D._SIDE["cls"][0] != mine[2][0]).sum()) == 0   # cls targets == labels


def test_duplicates_get_background_cls_target_and_weight():
    from torchvision.ops import box_iou
    gt, props, targets = _fixture()
    rh = _roi_heads()
    rh.fg_bg_sampler.batch_size_per_image = 64
    for w in (1.0, 0.3, 0.0):
        fn = D.make_select_training_samples(torch, box_iou, dup_w=w, margin=0.0)
        torch.manual_seed(0)
        _p, _m, labels, _r = fn(rh, [props.clone()], targets)
        cls, wt = D._SIDE["cls"][0], D._SIDE["w"][0]
        changed = (cls != labels[0])
        assert int(changed.sum()) == 2                        # the two duplicates
        assert (cls[changed] == 0).all() and (labels[0][changed] == 1).all()
        assert torch.allclose(wt[changed], torch.tensor(w))
        assert torch.allclose(wt[~changed], torch.tensor(1.0))


def test_margin_keeps_near_best_proposals_canonical():
    from torchvision.ops import box_iou
    gt, props, _ = _fixture()
    rh = _roi_heads()
    allp = rh.add_gt_proposals([props], [gt])[0]
    matched, labels = rh.assign_targets_to_proposals([allp], [gt],
                                                     [torch.ones(2, dtype=torch.int64)])
    dup = D.duplicate_mask(torch, box_iou, gt, allp, matched[0], labels[0], margin=0.2)
    assert dup[:props.shape[0]].tolist() == [False, False, False, False, False, False]


def test_patched_loss_equals_original_without_duplicates():
    import torch.nn.functional as functional
    from torchvision.models.detection import roi_heads as rh
    orig = rh.fastrcnn_loss
    patched = D.make_patched_fastrcnn_loss(torch, functional, orig)
    torch.manual_seed(0)
    logits = torch.randn(10, 2)
    reg = torch.randn(10, 8)
    labels = [torch.tensor([1, 1, 0, 0, 0, 0, 1, 0, 0, 0])]
    rt = [torch.randn(10, 4)]
    D._SIDE["cls"] = [labels[0].clone()]
    D._SIDE["w"] = [torch.ones(10)]
    a = orig(logits, reg, labels, rt)
    b = patched(logits, reg, labels, rt)
    assert torch.allclose(a[0], b[0]) and torch.allclose(a[1], b[1])


def test_dup_weight_reduces_cls_loss_contribution():
    import torch.nn.functional as functional
    from torchvision.models.detection import roi_heads as rh
    patched = D.make_patched_fastrcnn_loss(torch, functional, rh.fastrcnn_loss)
    logits = torch.tensor([[-3., 3.], [-3., 3.]])             # both confidently "filament"
    reg = torch.zeros(2, 8)
    labels = [torch.tensor([1, 1])]
    rt = [torch.zeros(2, 4)]
    D._SIDE["cls"] = [torch.tensor([1, 0])]                   # second is a duplicate
    out = {}
    for w in (1.0, 0.3, 0.0):
        D._SIDE["w"] = [torch.tensor([1.0, w])]
        out[w] = float(patched(logits, reg, labels, rt)[0])
    assert out[1.0] > out[0.3] > out[0.0]                     # softer penalty -> lower cls loss
