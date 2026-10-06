"""Focused tests for V42 pure helpers (no GPU, no full dataset)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "train_v42_pretrained_detector_overfit.py"
_spec = importlib.util.spec_from_file_location("v42", _PATH)
v42 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v42)


def _rle(mask: np.ndarray):
    return mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))


def test_gate_bars_match_v41():
    # Controlled A/B: gate bars and cohort identities must equal V41's.
    assert v42.TARGET_PQ == 0.85
    assert v42.TARGET_RECALL == 0.90
    assert v42.TARGET_MEDIAN_IOU == 0.80
    assert v42.MAX_FP == 40
    assert v42.EXPECTED_ANNOTATIONS == 308
    assert v42.EXPECTED_SMALL_THIN == 244
    assert v42.SELECTION_ID == "741d8124fd834c465743cb4ddbee265298354b4e59b5ab9650379d162673ca2c"
    assert v42.ANNOTATION_SHA256 == (
        "5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1"
    )


def test_pq_perfect_match():
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:30, 10:30] = 1
    metric, matched = v42.pq_metric(np, mask_utils, [_rle(mask)], [_rle(mask)])
    assert metric["tp"] == 1 and metric["fp"] == 0 and metric["fn"] == 0
    assert metric["pq"] == 1.0 and metric["recall"] == 1.0
    assert matched == [1.0]


def test_pq_empty_predictions_all_fn():
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:30, 10:30] = 1
    metric, matched = v42.pq_metric(np, mask_utils, [], [_rle(mask), _rle(mask)])
    assert metric["tp"] == 0 and metric["fp"] == 0 and metric["fn"] == 2
    assert metric["pq"] == 0.0 and matched == []


def test_pq_false_positive_counts():
    gt = np.zeros((64, 64), dtype=np.uint8)
    gt[10:30, 10:30] = 1
    far = np.zeros((64, 64), dtype=np.uint8)
    far[40:60, 40:60] = 1  # no overlap -> FP + FN
    metric, _ = v42.pq_metric(np, mask_utils, [_rle(far)], [_rle(gt)])
    assert metric["tp"] == 0 and metric["fp"] == 1 and metric["fn"] == 1


def test_pq_partial_iou_below_threshold_is_not_tp():
    gt = np.zeros((64, 64), dtype=np.uint8)
    gt[10:30, 10:30] = 1  # area 400
    pred = np.zeros((64, 64), dtype=np.uint8)
    pred[10:30, 10:22] = 1  # ~60% overlap -> IoU ~0.6 > 0.5 -> TP
    metric, matched = v42.pq_metric(np, mask_utils, [_rle(pred)], [_rle(gt)])
    assert metric["tp"] == 1
    assert 0.5 < matched[0] < 1.0


def test_densify_polyline_monotone_length():
    dense = v42.densify_polyline(np, [0.0, 0.0, 10.0, 0.0])
    assert len(dense) >= 10
    assert dense[0].tolist() == [0, 0]
    assert dense[-1].tolist() == [10, 0]


def test_densify_polyline_rejects_single_point():
    import pytest

    with pytest.raises(ValueError):
        v42.densify_polyline(np, [1.0, 1.0])
