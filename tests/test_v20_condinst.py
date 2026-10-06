"""Tests for V20 CondInst fast RLE evaluation and frozen result."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fast = load("v20_fast_eval_under_test", ROOT / "scripts/evaluate_v20_condinst_fast.py")
RESULT = ROOT / "outputs/v20_condinst_azure/v20_evaluation.json"
requires_result = pytest.mark.skipif(not RESULT.exists(), reason="V20 result not pulled")


def test_rle_scorer_exactly_matches_mask_scorer():
    rng = np.random.default_rng(20260911)
    for n_pred, n_gt in ((0, 0), (0, 3), (4, 0), (1, 1), (5, 7)):
        predictions = [rng.random((64, 64)) > 0.91 for _ in range(n_pred)]
        targets = [rng.random((64, 64)) > 0.93 for _ in range(n_gt)]
        assert fast.pq_counts_rles(
            fast.encode_predictions(predictions), fast.encode_predictions(targets)
        ) == fast.v20.pq_counts(predictions, targets)


def test_strict_iou_threshold_is_greater_than_half():
    target = np.zeros((2, 3), dtype=bool)
    target[0, :2] = True
    prediction = np.zeros((2, 3), dtype=bool)
    prediction[0, 0] = True  # IoU exactly 0.5: not a hit.
    result = fast.pq_counts_rles(
        fast.encode_predictions([prediction]), fast.encode_predictions([target])
    )
    assert result == [0.0, 0, 1, 1]


@requires_result
def test_v20_result_failed_both_frozen_export_bars():
    result = json.loads(RESULT.read_text())
    assert result["selected_checkpoint"] == "model_0002349.pth"
    assert result["selected_confidence"] == 0.5
    assert result["selected_tune"]["pq"] == pytest.approx(0.3581905995201856)
    assert result["selected_audit"]["pq"] == pytest.approx(0.397239502051274)
    assert result["selected_tune"]["pq"] < result["submit_bar_tune"]
    assert result["selected_audit"]["pq"] < result["submit_bar_audit"]
    assert result["export_authorized"] is False
    assert result["action"] == "no_export"


@requires_result
def test_v20_result_has_exact_canonical_populations_and_audit_once():
    result = json.loads(RESULT.read_text())
    assert result["populations"] == {
        "train": {"files": 566, "records": 937, "annotations": 6606},
        "tune": {"files": 94, "records": 150, "annotations": 1134},
        "audit": {"files": 47, "records": 67, "annotations": 459},
    }
    assert len(result["tune_runs"]) == 4
    assert result["audit_selected_run"]["thresholds"] == [0.5]
    assert result["audit_selected_run"]["checkpoint"] == result["selected_checkpoint"]
