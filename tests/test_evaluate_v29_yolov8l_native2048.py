import ast

import numpy as np
from pycocotools import mask as mask_utils

from scripts import evaluate_v29_yolov8l_native2048 as v29


def rle(mask):
    return mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))


def test_public_operating_point_and_bounded_grid_are_frozen():
    assert 0.30 in v29.CONFIDENCES
    assert v29.NMS_IOUS == (0.0, 0.3, 0.7)
    assert v29.ORDER_POLICIES == ("public_class_id", "confidence")
    assert v29.candidate_key(0.0, "public_class_id", 0.30) == "iou0.0/public_class_id/c0.30"


def test_public_class_id_policy_reverses_equal_class_order():
    first = np.zeros((8, 8), np.uint8); first[1:6, 1:6] = 1
    second = np.zeros((8, 8), np.uint8); second[3:8, 3:8] = 1
    public, _ = v29.clean_masks([first, second], np.asarray([0.9, 0.5]), "public_class_id")
    confidence, _ = v29.clean_masks([first, second], np.asarray([0.9, 0.5]), "confidence")
    assert not np.array_equal(mask_utils.decode(public[0]), mask_utils.decode(confidence[0]))


def test_pq_is_strict_and_global_count_compatible():
    gt = np.zeros((8, 8), np.uint8); gt[1:5, 1:5] = 1
    perfect = v29.pq_counts([rle(gt)], [rle(gt)])
    assert perfect.tolist() == [1.0, 1.0, 0.0, 0.0]
    half = gt.copy(); half[1:3] = 0
    assert v29.pq_counts([rle(half)], [rle(gt)]).tolist() == [0.0, 0.0, 1.0, 1.0]


def test_no_audit_test_or_export_paths():
    source = open("scripts/evaluate_v29_yolov8l_native2048.py").read()
    tree = ast.parse(source)
    calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name))
    }
    assert "test/test_images" not in source
    assert "submission.csv" not in source
    assert "submit" not in calls and "to_csv" not in calls


def test_promotion_bar_is_materially_above_v17():
    assert v29.PROMOTION_PQ == 0.50
    assert v29.PROMOTION_PQ - v29.V17_TUNE_PQ > 0.05
