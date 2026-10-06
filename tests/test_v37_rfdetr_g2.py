import ast
from pathlib import Path

import numpy as np

from scripts import train_v37_rfdetr_g2 as g2

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "scripts/train_v37_rfdetr_g2.py").read_text()


def test_g2_contract_matches_frozen_design():
    assert g2.EPOCHS == 125
    assert g2.MAX_STEPS == 2000
    assert g2.CHECKPOINT_INTERVAL == 25
    assert g2.CONFIDENCE_GRID == (0.05, 0.10, 0.20, 0.30, 0.50)
    assert g2.PQ_GATE == 0.85
    assert g2.RECALL_GATE == 0.90
    assert g2.MEDIAN_IOU_GATE == 0.80
    assert g2.RESOLUTION == 768
    assert g2.MASK_DOWNSAMPLE_RATIO == 2


def test_strict_match_uses_one_to_one_and_strict_half_iou():
    target_a = np.zeros((8, 8), bool); target_a[1:5, 1:5] = True
    target_b = np.zeros((8, 8), bool); target_b[3:7, 3:7] = True
    prediction = target_a.copy()
    matched, tp, fp, fn = g2.strict_match([prediction], [target_a, target_b])
    assert matched == [1.0]
    assert (tp, fp, fn) == (1, 0, 1)

    half_target = np.zeros((2, 4), bool); half_target[:, :2] = True
    half_prediction = np.zeros((2, 4), bool); half_prediction[0, :] = True
    # Intersection 2, union 6 -> below half and therefore not matched.
    assert g2.strict_match([half_prediction], [half_target])[1:] == (0, 1, 1)


def test_checkpoint_epoch_parses_lightning_zero_based_names():
    assert g2.checkpoint_epoch(Path("checkpoint_24.ckpt")) == 25
    assert g2.checkpoint_epoch(Path("checkpoint_124.ckpt")) == 125


def test_g2_runner_has_no_tune_audit_test_export_or_submission_path():
    tree = ast.parse(SOURCE)
    calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name))
    }
    assert "submission.csv" not in SOURCE
    assert "to_csv" not in calls and "submit" not in calls and "export" not in calls
    assert '"reads_tune": False' in SOURCE
    assert '"reads_audit": False' in SOURCE
    assert '"reads_test": False' in SOURCE
    assert "run_test=False" in SOURCE
    assert "multi_scale=False" in SOURCE
    assert "scale_jitter=False" in SOURCE
    assert "epochs=EPOCHS" in SOURCE


def test_g2_source_resolution_and_selection_are_pinned():
    assert "masks.shape[1:] != (2048, 2048)" in SOURCE
    assert "linear_sum_assignment(-ious)" in SOURCE
    assert "ious[row, column] > 0.5" in SOURCE
    assert "max(candidates" in SOURCE
