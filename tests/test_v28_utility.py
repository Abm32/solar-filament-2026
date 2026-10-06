import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import analyze_v28_utility as v28

ROOT = Path(__file__).resolve().parents[1]


def test_expected_utility_is_annotator_aware_and_thresholded():
    best = np.asarray([
        [0.8, 0.4, 0.7, 0.0],
        [0.6, 0.9, 0.2, 0.0],
    ])
    utility, fraction, mean_iou = v28.expected_utility(best)
    np.testing.assert_allclose(utility, [0.7, 0.45, 0.35, 0.0])
    np.testing.assert_allclose(fraction, [1.0, 0.5, 0.5, 0.0])
    np.testing.assert_allclose(mean_iou, [0.7, 0.65, 0.45, 0.0])


def test_file_folds_are_deterministic_and_filename_grouped():
    assert v28.file_fold("a.jpeg") == v28.file_fold("a.jpeg")
    assert 0 <= v28.file_fold("b.jpeg") < v28.FOLDS
    names = [f"{index}.jpeg" for index in range(100)]
    folds = {name: v28.file_fold(name) for name in names}
    assert len(set(folds.values())) == v28.FOLDS


def test_ranking_auc_handles_ties_and_perfect_order():
    assert v28.ranking_auc([1, 1, 1, 1], [0, 1, 0, 1]) == pytest.approx(0.5)
    assert v28.ranking_auc([0.1, 0.9, 0.2, 0.8], [0, 1, 0, 1]) == pytest.approx(1.0)


def sample_frame():
    rows = []
    for fold in range(4):
        for file_index in range(2):
            name = f"f{fold}_{file_index}.jpeg"
            for confidence, utility, fraction in ((0.9, 0.8, 1.0), (0.2, 0.0, 0.0)):
                rows.append({"file_name": name, "fold": fold, "confidence": confidence,
                             "utility": utility, "positive_fraction": fraction,
                             "annotator_records": 1, "gt_instances": 1,
                             "model": confidence})
    return pd.DataFrame(rows)


def test_expected_pq_prices_selected_fp_and_gt_fn():
    frame = sample_frame()
    high_only = v28.utility_pq(frame, "confidence", 0.5)
    all_rows = v28.utility_pq(frame, "confidence", 0.0)
    assert high_only > all_rows
    assert high_only == pytest.approx(0.8)


def test_cross_fitted_thresholds_use_other_folds_and_return_all_folds():
    frame = sample_frame()
    report = v28.cross_fitted_utility_pq(frame, "model")
    assert len(report["folds"]) == v28.FOLDS
    assert report["mean_fold_pq"] == pytest.approx(0.8)
    assert all(row["threshold"] >= 0.05 for row in report["folds"])


def test_serving_crop_channels_are_fixed():
    assert v28.CROP_SIZE == 64
    source = (ROOT / "scripts/analyze_v28_utility.py").read_text()
    assert "image_crop" in source
    assert "proposal_crop" in source
    assert "probability_native" in source
    assert "crop.astype(np.uint8)" in source


def test_oof_model_body_is_reachable_and_not_nested_after_return():
    source = (ROOT / "scripts/analyze_v28_utility.py").read_text()
    tree = ast.parse(source)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert {"utility_pq", "cross_fitted_utility_pq", "train_oof"} <= set(functions)
    train_source = ast.get_source_segment(source, functions["train_oof"])
    assert "UtilityNet" in train_source
    assert "cross_fitted_pq" in train_source
    assert "learnability_gates" in train_source


def test_no_tune_audit_test_or_submission_paths():
    source = (ROOT / "scripts/analyze_v28_utility.py").read_text()
    tree = ast.parse(source)
    calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name))
    }
    assert "test/test_images" not in source
    assert "submission.csv" not in source
    assert "audit_opened.json" not in source
    assert "competitions_submit" not in calls and "submit" not in calls
