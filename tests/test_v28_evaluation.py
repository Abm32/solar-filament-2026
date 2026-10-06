import ast
import json
from pathlib import Path

import numpy as np
import pytest
from pycocotools import mask as mask_utils

from scripts import evaluate_v28_utility as v28

ROOT = Path(__file__).resolve().parents[1]


def encoded(mask):
    return mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))


def test_oof_authorization_requires_all_frozen_deltas_and_unanimous_threshold(tmp_path, monkeypatch):
    report = {
        "learnable": True,
        "metrics": {
            "confidence": {"auc": 0.80, "utility_correlation": 0.50,
                           "cross_fitted_pq": {"mean_fold_pq": 0.40}},
            "oof_product": {"auc": 0.82, "utility_correlation": 0.52,
                            "cross_fitted_pq": {
                                "mean_fold_pq": 0.41,
                                "folds": [{"threshold": 0.10}] * 4,
                            }},
        },
        "learnability_gates": {"auc_delta": 0.01, "correlation_delta": 0.01,
                               "cross_fitted_pq_delta": 0.005},
    }
    path = tmp_path / "oof_report.json"
    path.write_text(json.dumps(report))
    monkeypatch.setattr(v28, "OOF_REPORT", path)
    authorization = v28.require_oof_authorization()
    assert authorization["observed_deltas"]["cross_fitted_pq"] == pytest.approx(0.01)
    report["metrics"]["oof_product"]["cross_fitted_pq"]["folds"][0]["threshold"] = 0.125
    path.write_text(json.dumps(report))
    with pytest.raises(RuntimeError, match="unanimously"):
        v28.require_oof_authorization()


def test_disjoint_refined_uses_product_order_and_minimum_area():
    first = np.zeros((12, 12), dtype=np.uint8)
    first[1:9, 1:9] = 1
    second = np.zeros((12, 12), dtype=np.uint8)
    second[4:11, 4:11] = 1
    small = np.zeros((12, 12), dtype=np.uint8)
    small[0:2, 0:2] = 1
    rles, scores = v28.disjoint_refined(
        [encoded(first), encoded(second), encoded(small)], [0.2, 0.3, 0.9], threshold=0.1,
    )
    assert scores.tolist() == [0.3, 0.2]
    decoded = [mask_utils.decode(rle).astype(bool) for rle in rles]
    assert not (decoded[0] & decoded[1]).any()
    assert decoded[0].sum() == second.sum()
    assert decoded[1].sum() == first.sum() - (first.astype(bool) & second.astype(bool)).sum()


def test_product_threshold_filters_after_ownership_without_blocking_lower_scores():
    high_but_rejected = np.zeros((12, 12), dtype=np.uint8)
    high_but_rejected[1:9, 1:9] = 1
    accepted = np.zeros((12, 12), dtype=np.uint8)
    accepted[4:11, 4:11] = 1
    rles, scores = v28.disjoint_refined(
        [encoded(high_but_rejected), encoded(accepted)], [0.09, 0.20], threshold=0.10,
    )
    assert scores.tolist() == [0.20]
    assert mask_utils.area(rles[0]) == accepted.sum()


def test_tune_gates_require_pq_recall_and_fp_control():
    metrics = {
        "control": {"pq": 0.448703, "tp": 704, "fp": 306},
        "hybrid_confidence": {"pq": 0.432869, "tp": 678, "fp": 339},
        "v28_product": {"pq": 0.460, "tp": 705, "fp": 350},
    }
    decision = v28.evaluate_gates(metrics)
    assert decision["audit_authorized"]
    metrics["v28_product"]["tp"] = 703
    assert not v28.evaluate_gates(metrics)["audit_authorized"]


def test_online_crop_lists_normalize_to_model_tensor_shape():
    crops = [np.zeros((4, 64, 64), dtype=np.uint8) for _ in range(3)]
    values = v28.normalized_crops(crops)
    assert values.shape == (3, 4, 64, 64)
    assert values.dtype == np.uint8


def test_regression_guard_checks_exact_counts_and_iou():
    v28.assert_regression(v28.EXPECTED_CONTROL.copy(), v28.EXPECTED_CONTROL, "control")
    changed = v28.EXPECTED_CONTROL.copy()
    changed[2] += 1
    with pytest.raises(RuntimeError, match="counts changed"):
        v28.assert_regression(changed, v28.EXPECTED_CONTROL, "control")


def test_frozen_constants_match_successful_oof_and_v26_geometry():
    assert v28.EPOCHS == 6
    assert v28.PRODUCT_THRESHOLD == 0.10
    assert v28.REFINE_THRESHOLD == 0.45
    assert v28.CONTROL_CONFIDENCE == 0.30
    assert v28.HYBRID_CONFIDENCE == 0.45
    assert v28.cohort.CELL["name"] == "iou030"
    assert v28.ABSOLUTE_TUNE_REQUIRED == pytest.approx(0.45870271727943207)


def test_no_test_csv_or_submission_access():
    source = (ROOT / "scripts/evaluate_v28_utility.py").read_text()
    tree = ast.parse(source)
    calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name))
    }
    assert "test/test_images" not in source
    assert "submission.csv" not in source
    assert "competitions_submit" not in calls and "submit" not in calls
    assert "to_csv" in calls  # diagnostics only
