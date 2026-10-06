"""Focused tests for V52 clDice source helpers that do not require torch.

Validates: frozen canonical split parity with the project convention, semantic->
instance conversion (connected components + min-area + disjointness), and the
greedy one-to-one IoU>0.5 PQ counting against hand-built cases.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from pycocotools import mask as mask_utils
from skimage import measure

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_v52_cldice_semantic.py"
spec = importlib.util.spec_from_file_location("v52", SCRIPT)
v52 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v52)


def _doc():
    # two physical files, one with two annotator records (max-instance wins)
    return {
        "images": [
            {"id": 1, "file_name": "a.jpeg", "height": 2048, "width": 2048},
            {"id": 2, "file_name": "a.jpeg", "height": 2048, "width": 2048},
            {"id": 3, "file_name": "b.jpeg", "height": 2048, "width": 2048},
        ],
        "annotations": [
            {"image_id": 1, "area": 10, "segmentation": {"size": [2048, 2048], "counts": "a"}},
            {"image_id": 2, "area": 10, "segmentation": {"size": [2048, 2048], "counts": "b"}},
            {"image_id": 2, "area": 10, "segmentation": {"size": [2048, 2048], "counts": "c"}},
            {"image_id": 3, "area": 10, "segmentation": {"size": [2048, 2048], "counts": "d"}},
        ],
    }


def test_canonical_split_matches_convention():
    import random
    from collections import defaultdict

    doc = {"images": [{"id": i, "file_name": f"f{i:03d}.jpeg",
                       "height": 2048, "width": 2048} for i in range(707)],
           "annotations": []}
    fit, tune, by_file, images = v52.canonical_split(doc)
    # reference computation
    bf = defaultdict(list)
    for r in doc["images"]:
        bf[r["file_name"]].append(str(r["id"]))
    files = sorted(bf)
    random.Random(42).shuffle(files)
    assert tune == sorted(files[:141])
    assert fit == sorted(files[141:])
    assert len(tune) == 141 and len(fit) == 566
    assert set(tune).isdisjoint(set(fit))


def test_records_for_picks_max_instance_record():
    doc = _doc()
    annotations = v52.load_annotations(doc)
    images = {str(r["id"]): r for r in doc["images"]}
    by_file = v52.document_by_file(doc)
    recs = dict(v52.records_for(["a.jpeg", "b.jpeg"], by_file, images, annotations))
    # a.jpeg: id 2 has two anns -> wins over id 1
    assert recs["a.jpeg"] == "2"
    assert recs["b.jpeg"] == "3"


def test_mask_to_instances_disjoint_and_min_area():
    prob = np.zeros((64, 64), dtype=np.float32)
    prob[5:15, 5:15] = 0.9      # area 100 -> kept
    prob[40:43, 40:42] = 0.9    # area 6 < MIN_AREA(20) -> dropped
    masks, scores = v52.mask_to_instances(np, None, measure, prob, 0.5)
    assert len(masks) == 1
    assert masks[0].sum() == 100
    assert scores[0] == pytest.approx(0.9, abs=1e-6)
    # connected components are disjoint by construction
    overlap = np.zeros((64, 64), dtype=np.int32)
    for m in masks:
        overlap += m
    assert overlap.max() <= 1


def _rle(binary):
    return mask_utils.encode(np.asfortranarray(binary.astype(np.uint8)))


def test_pq_counts_one_to_one_matching():
    gt = np.zeros((64, 64), dtype=np.uint8)
    gt[5:25, 5:25] = 1
    gt_rles = [_rle(gt)]
    # strong overlap pred (IoU>0.5) + a spurious FP
    good = np.zeros((64, 64), dtype=np.uint8)
    good[6:25, 6:25] = 1
    fp = np.zeros((64, 64), dtype=np.uint8)
    fp[40:50, 40:50] = 1
    iou_sum, tp, fp_c, fn = v52.pq_counts_for_image(
        np, mask_utils, [good, fp], [0.9, 0.8], gt_rles
    )
    assert tp == 1 and fp_c == 1 and fn == 0
    assert iou_sum > 0.5


def test_pq_counts_empty_cases():
    assert v52.pq_counts_for_image(np, mask_utils, [], [], []) == (0.0, 0, 0, 0)
    g = [_rle(np.ones((8, 8), dtype=np.uint8))]
    assert v52.pq_counts_for_image(np, mask_utils, [], [], g) == (0.0, 0, 0, 1)
    m = [np.ones((8, 8), dtype=np.uint8)]
    assert v52.pq_counts_for_image(np, mask_utils, m, [0.5], []) == (0.0, 0, 1, 0)
