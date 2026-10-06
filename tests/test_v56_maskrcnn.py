"""Tests for V56 Mask R-CNN geometry + PQ logic (CPU; no torchvision/GPU)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from pycocotools import mask as mask_utils

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load():
    spec = importlib.util.spec_from_file_location("train_v56_maskrcnn",
                                                  SCRIPTS / "train_v56_maskrcnn.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v56 = _load()


def _mask(r0, r1, c0, c1, size=2048):
    m = np.zeros((size, size), np.uint8)
    m[r0:r1, c0:c1] = 1
    return m


def test_masks_to_boxes_xyxy():
    m = _mask(10, 50, 20, 80)
    (x0, y0, x1, y1), = v56.masks_to_boxes(np, [m])
    assert (x0, y0, x1, y1) == (20.0, 10.0, 80.0, 50.0)


def test_pq_perfect_match():
    m = _mask(100, 200, 100, 200)
    rle = mask_utils.encode(np.asfortranarray(m))
    s, tp, fp, fn = v56.pq_counts_for_image(np, mask_utils, [rle], [rle])
    assert (tp, fp, fn) == (1, 0, 0)
    assert s == pytest.approx(1.0)


def test_pq_fp_and_fn():
    gt = mask_utils.encode(np.asfortranarray(_mask(0, 100, 0, 100)))
    pred_far = mask_utils.encode(np.asfortranarray(_mask(500, 600, 500, 600)))
    s, tp, fp, fn = v56.pq_counts_for_image(np, mask_utils, [pred_far], [gt])
    assert (tp, fp, fn) == (0, 1, 1)


def test_disjoint_rles_enforces_ownership():
    # two overlapping masks; higher score keeps the overlap, lower loses it
    a = _mask(0, 100, 0, 100).astype(np.uint8)       # score 0.9
    b = _mask(50, 150, 50, 150).astype(np.uint8)     # score 0.5, overlaps a
    import cv2
    rles = v56.disjoint_rles(np, cv2, mask_utils, np.stack([a, b]),
                             np.array([0.9, 0.5]), imgsz=2048)
    # decode and verify no pixel owned twice
    decoded = [mask_utils.decode(r) for r in rles]
    assert len(decoded) == 2
    overlap = (decoded[0] & decoded[1]).sum()
    assert overlap == 0


def test_disjoint_rles_drops_tiny_after_subtraction():
    big = _mask(0, 200, 0, 200).astype(np.uint8)
    tiny_inside = _mask(10, 15, 10, 15).astype(np.uint8)  # inside big; vanishes after subtract
    import cv2
    rles = v56.disjoint_rles(np, cv2, mask_utils, np.stack([big, tiny_inside]),
                             np.array([0.9, 0.8]), imgsz=2048)
    assert len(rles) == 1  # tiny fully subtracted -> dropped


def _ann_from_mask(m):
    """Build a minimal COCO-style annotation (polygon segmentation) from a mask bbox."""
    ys, xs = np.nonzero(m)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    poly = [x0, y0, x1, y0, x1, y1, x0, y1]
    return {"segmentation": [poly]}


def test_union_rles_merges_annotators_and_dedups():
    # annotator A marks filaments 1,2; annotator B marks filaments 2,3 (2 overlaps)
    f1 = _mask(0, 100, 0, 100)
    f2 = _mask(0, 100, 500, 600)
    f3 = _mask(0, 100, 1000, 1100)
    images = {"a": {"height": 2048, "width": 2048, "file_name": "x.jpeg"},
              "b": {"height": 2048, "width": 2048, "file_name": "x.jpeg"}}
    by_file = {"x.jpeg": ["a", "b"]}
    annotations = {"a": [_ann_from_mask(f1), _ann_from_mask(f2)],
                   "b": [_ann_from_mask(f2), _ann_from_mask(f3)]}
    union = v56.union_rles(mask_utils, np, images, by_file, annotations, "x.jpeg")
    # union of {1,2} and {2,3} with dedup = {1,2,3} = 3 distinct instances
    assert len(union) == 3


def test_union_rles_single_annotator_unchanged():
    f1 = _mask(0, 100, 0, 100)
    f2 = _mask(0, 100, 500, 600)
    images = {"a": {"height": 2048, "width": 2048, "file_name": "x.jpeg"}}
    by_file = {"x.jpeg": ["a"]}
    annotations = {"a": [_ann_from_mask(f1), _ann_from_mask(f2)]}
    union = v56.union_rles(mask_utils, np, images, by_file, annotations, "x.jpeg")
    assert len(union) == 2


def test_union_rles_consensus_min_agree2_keeps_only_agreed():
    # A marks {1,2}, B marks {2,3}. Only filament 2 has >=2-annotator agreement.
    f1 = _mask(0, 100, 0, 100)
    f2 = _mask(0, 100, 500, 600)
    f3 = _mask(0, 100, 1000, 1100)
    images = {"a": {"height": 2048, "width": 2048, "file_name": "x.jpeg"},
              "b": {"height": 2048, "width": 2048, "file_name": "x.jpeg"}}
    by_file = {"x.jpeg": ["a", "b"]}
    annotations = {"a": [_ann_from_mask(f1), _ann_from_mask(f2)],
                   "b": [_ann_from_mask(f2), _ann_from_mask(f3)]}
    full = v56.union_rles(mask_utils, np, images, by_file, annotations, "x.jpeg", min_agree=1)
    consensus = v56.union_rles(mask_utils, np, images, by_file, annotations, "x.jpeg", min_agree=2)
    assert len(full) == 3        # union
    assert len(consensus) == 1   # only filament 2 (both annotators)


def test_target_min_agree_mapping():
    assert v56._target_min_agree("union") == 1
    assert v56._target_min_agree("consensus2") == 2
    assert v56._target_min_agree("single") == 1
