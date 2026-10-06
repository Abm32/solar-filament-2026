"""Focused tests for V43 pure helpers (no GPU)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "train_v43_mask_refine_overfit.py"
_spec = importlib.util.spec_from_file_location("v43", _PATH)
v43 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v43)


def _rle(mask):
    return mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))


def test_gate_and_cohort_identity_match_v41_v42():
    assert v43.TARGET_PQ == 0.85
    assert v43.TARGET_RECALL == 0.90
    assert v43.TARGET_MEDIAN_IOU == 0.80
    assert v43.MAX_FP == 40
    assert v43.EXPECTED_ANNOTATIONS == 308
    assert v43.SELECTION_ID == "741d8124fd834c465743cb4ddbee265298354b4e59b5ab9650379d162673ca2c"
    assert v43.ANNOTATION_SHA256 == (
        "5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1"
    )


def test_pq_perfect():
    m = np.zeros((64, 64), dtype=np.uint8)
    m[10:30, 10:30] = 1
    metric, matched = v43.pq_metric(np, mask_utils, [_rle(m)], [_rle(m)])
    assert metric["pq"] == 1.0 and metric["tp"] == 1 and matched == [1.0]


def test_merge_overlapping_views_collapse_to_one():
    # Two near-identical soft masks (two TTA views of same object) -> one prediction.
    a = np.zeros((64, 64), dtype=np.float32)
    a[10:30, 10:30] = 0.9
    b = np.zeros((64, 64), dtype=np.float32)
    b[11:31, 10:30] = 0.9  # heavy overlap
    preds = v43._merge_dets(np, mask_utils, [(a, 0.9), (b, 0.8)])
    assert len(preds) == 1


def test_merge_disjoint_views_stay_separate():
    a = np.zeros((64, 64), dtype=np.float32)
    a[5:15, 5:15] = 0.9
    b = np.zeros((64, 64), dtype=np.float32)
    b[40:55, 40:55] = 0.9  # no overlap
    preds = v43._merge_dets(np, mask_utils, [(a, 0.9), (b, 0.8)])
    assert len(preds) == 2


def test_merge_drops_tiny_masks():
    a = np.zeros((64, 64), dtype=np.float32)
    a[0:3, 0:3] = 0.9  # area 9 < MIN_AREA(20)
    preds = v43._merge_dets(np, mask_utils, [(a, 0.9)])
    assert preds == []


def test_merge_empty_returns_empty():
    assert v43._merge_dets(np, mask_utils, []) == []
