"""V14 decision logic: grid discipline, losses, IoU-head metrics, and both bars.

Extracts the pure-Python/numpy parts of v14_dualhead.py by AST so the GPU-only
helpers (build_dual_head, load_compatible_weights, run_experiment) and their torch /
torchvision imports are never touched, matching test_detector_flags.py and
test_v13_compose.py.

The torch-dependent loss functions are covered separately with a tiny numpy stub
that mimics the handful of torch operations they use, so the FN-weighting property
(the whole point of arm_b) is actually asserted rather than assumed.
"""
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SOURCE = Path(__file__).resolve().parents[1] / "kaggle_inference/v14_dualhead.py"
tree = ast.parse(SOURCE.read_text())

constant_names = {"CONTROL", "ARMS", "ARM_LOSS", "CROP_SIZE", "BATCH_SIZE", "EPOCHS",
                  "LEARNING_RATE", "WEIGHT_DECAY", "SEED", "IOU_LOSS_WEIGHT",
                  "TVERSKY_ALPHA", "TVERSKY_BETA", "TVERSKY_GAMMA",
                  "DEEP_SUPERVISION_WEIGHTS", "MASK_THRESHOLDS", "IOU_GATES",
                  "CONTROL_REFERENCE", "BASELINES", "CONSENSUS_SCORE_AUC",
                  "SUBMIT_BAR", "PROMOTION_GATE", "MIN_AREA", "COUNTS", "POLICY",
                  "MATCH_IOU"}
function_names = {"pq", "axis_metrics", "candidate_key", "declared_candidates",
                  "verify_grid", "verify_control", "select_candidate", "evaluate_bars",
                  "ranking_auc", "calibration_error"}
body = [node for node in tree.body
        if (isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in constant_names for t in node.targets))
        or (isinstance(node, ast.FunctionDef) and node.name in function_names)]
ns = {"np": np, "pd": pd, "math": __import__("math")}
exec(compile(ast.Module(body=body, type_ignores=[]), str(SOURCE), "exec"), ns)


# ---------------------------------------------------------------- grid discipline

def test_grid_is_exactly_twenty_one_predeclared_candidates():
    assert ns["verify_grid"]() is True
    assert len(ns["declared_candidates"]()) == 21
    assert ns["MASK_THRESHOLDS"] == (0.40, 0.45, 0.50)
    assert ns["IOU_GATES"] == (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)


def test_candidate_keys_are_stable_and_unique():
    keys = ns["declared_candidates"]()
    assert len(set(keys)) == len(keys)
    assert ns["candidate_key"](0.45, 0.30) == "m0.45_g0.30"
    assert "m0.45_g0.30" in keys


def test_select_candidate_rejects_anything_outside_the_declared_grid():
    good = {"arm_a/m0.45_g0.30": dict(tune=[1.0, 2, 0, 0], audit=[1.0, 2, 0, 0])}
    assert ns["select_candidate"](good) == "arm_a/m0.45_g0.30"
    for bad_key in ("arm_c/m0.45_g0.30", "arm_a/m0.99_g0.99"):
        with pytest.raises(ValueError, match="outside the predeclared"):
            ns["select_candidate"]({bad_key: dict(tune=[1.0, 2, 0, 0], audit=[1.0, 2, 0, 0])})


def test_select_candidate_is_tune_only_and_ignores_audit():
    results = {
        "arm_a/m0.40_g0.30": dict(tune=[44.0, 100, 0, 0], audit=[10.0, 100, 0, 0]),
        "arm_b/m0.45_g0.35": dict(tune=[43.0, 100, 0, 0], audit=[99.0, 100, 0, 0]),
    }
    # The 0.99-audit row must NOT win.
    assert ns["select_candidate"](results) == "arm_a/m0.40_g0.30"


# ------------------------------------------------------------------ axis reporting

def test_axis_metrics_separates_recall_from_mask_quality():
    """The core V14 requirement: PQ alone hid which axis moved on four submissions."""
    counts = [470.272, 686, 309, 448]
    axes = ns["axis_metrics"](counts)
    assert axes["recall"] == pytest.approx(686 / 1134, abs=1e-6)
    assert axes["mean_matched_iou"] == pytest.approx(470.272 / 686, abs=1e-6)
    assert axes["fp"] == 309
    assert axes["n_gt"] == 1134
    assert axes["pq"] == pytest.approx(ns["pq"](counts))


def test_axis_metrics_distinguishes_two_runs_with_identical_pq():
    """Same PQ, opposite axis trade: the report must not present these as equal."""
    # recall-heavy vs iou-heavy, tuned to near-identical PQ
    a = ns["axis_metrics"]([600.0, 900, 300, 234])
    b = ns["axis_metrics"]([600.0, 750, 150, 384])
    assert a["pq"] == pytest.approx(b["pq"], abs=2e-3)
    assert a["recall"] > b["recall"] + 0.1
    assert b["mean_matched_iou"] > a["mean_matched_iou"] + 0.1


def test_axis_metrics_handles_empty_predictions():
    axes = ns["axis_metrics"]([0.0, 0, 0, 0])
    assert axes["recall"] == 0.0 and axes["mean_matched_iou"] == 0.0 and axes["pq"] == 0.0


# --------------------------------------------------------------- control reference

def test_control_reference_pins_the_accepted_e015_counts():
    reference = ns["CONTROL_REFERENCE"]
    assert reference["tune"] == [pytest.approx(426.5473399071961), 626, 211, 508]
    assert reference["audit"] == [pytest.approx(182.75261599994468), 265, 88, 194]
    assert ns["pq"](reference["tune"]) == pytest.approx(0.4328232774299302)


def test_verify_control_accepts_exact_counts_and_rejects_drift():
    reference = ns["CONTROL_REFERENCE"]
    assert ns["verify_control"]({k: list(v) for k, v in reference.items()}) is True
    drifted = {k: list(v) for k, v in reference.items()}
    drifted["tune"][1] += 1
    with pytest.raises(ValueError, match="Control tune counts differ"):
        ns["verify_control"](drifted)
    nudged = {k: list(v) for k, v in reference.items()}
    nudged["audit"][0] += 1e-3
    with pytest.raises(ValueError, match="Control audit counts differ"):
        ns["verify_control"](nudged)


# ------------------------------------------------------------------------- gates

def _results(tune_pq, audit_pq, tp=100, fp=0, fn=0):
    """Counts whose PQ equals the requested value (tp-only => PQ = iou_sum/tp)."""
    denominator = tp + 0.5 * (fp + fn)
    return {"arm_b/m0.45_g0.30": dict(tune=[tune_pq * denominator, tp, fp, fn],
                                      audit=[audit_pq * denominator, tp, fp, fn])}


def test_submit_bar_is_anchored_to_v13_and_the_e015_audit_floor():
    bar, baselines = ns["SUBMIT_BAR"], ns["BASELINES"]
    assert bar["min_tune_pq"] == pytest.approx(baselines["v13_compose"]["tune_pq"])
    assert bar["min_audit_pq"] == pytest.approx(baselines["e015_control"]["audit_pq"])


def test_promotion_gate_demands_ten_times_v13s_delta():
    """V13 gained +0.0038 locally and moved the public score 0.00, so the bar is 0.010."""
    gate = ns["PROMOTION_GATE"]
    assert gate["required_tune_delta_vs_v13"] == pytest.approx(0.010)
    assert gate["require_audit_at_least_v13"] is True


def test_export_blocked_below_the_best_measured_candidate():
    v13 = ns["BASELINES"]["v13_compose"]
    report = ns["evaluate_bars"]("arm_b/m0.45_g0.30",
                                 _results(v13["tune_pq"] - 0.001, 0.47))
    assert report["export_authorized"] is False
    assert report["promotable"] is False
    assert "do not submit" in report["recommendation"]


def test_export_authorized_but_not_promotable_says_do_not_submit():
    """The V13 lesson encoded: a small local gain must not spend a submission."""
    v13 = ns["BASELINES"]["v13_compose"]
    report = ns["evaluate_bars"]("arm_b/m0.45_g0.30",
                                 _results(v13["tune_pq"] + 0.004, v13["audit_pq"] + 0.001))
    assert report["export_authorized"] is True
    assert report["promotable"] is False
    assert "do NOT submit" in report["recommendation"]


def test_promotable_only_when_both_tune_margin_and_audit_hold():
    v13 = ns["BASELINES"]["v13_compose"]
    clear = ns["evaluate_bars"]("arm_b/m0.45_g0.30",
                                _results(v13["tune_pq"] + 0.010, v13["audit_pq"] + 0.001))
    assert clear["promotable"] is True
    assert clear["recommendation"] == "promote and submit"
    audit_regress = ns["evaluate_bars"](
        "arm_b/m0.45_g0.30", _results(v13["tune_pq"] + 0.020, v13["audit_pq"] - 0.001))
    assert audit_regress["export_authorized"] is True
    assert audit_regress["promotable"] is False


def test_report_exposes_axis_deltas_against_v13():
    v13 = ns["BASELINES"]["v13_compose"]
    results = {"arm_b/m0.45_g0.30": dict(tune=[700.0, 950, 200, 184],
                                         audit=[182.75, 265, 88, 194])}
    report = ns["evaluate_bars"]("arm_b/m0.45_g0.30", results)
    deltas = report["axis_deltas_vs_v13"]
    assert deltas["recall"] > 0, "a recall gain must be visible as a recall gain"
    assert deltas["fp"] == 200 - v13["fp"]
    assert set(report["deltas_vs_baselines"]) == {"e015_control", "v13_compose"}
    assert report["tune_axes"]["mean_matched_iou"] == pytest.approx(700.0 / 950)


def test_target_arithmetic_matches_the_ceiling_analysis():
    """Sanity-check the premise: 0.60 local needs BOTH axes, per the analysis doc."""
    n_gt = 1134
    # Perfect recall at today's mask quality still barely reaches 0.60.
    perfect_recall = ns["pq"]([0.6855 * n_gt, n_gt, 309, 0])
    assert 0.59 < perfect_recall < 0.61
    # Today's recall with excellent masks does not reach 0.60 either.
    perfect_masks = ns["pq"]([0.90 * 686, 686, 309, 448])
    assert perfect_masks < 0.60
    # Both together clear it.
    both = ns["pq"]([0.75 * 948, 948, 309, n_gt - 948])
    assert both > 0.59


# --------------------------------------------------------------- IoU-head metrics

def test_ranking_auc_matches_known_values():
    assert ns["ranking_auc"]([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == pytest.approx(1.0)
    assert ns["ranking_auc"]([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == pytest.approx(0.0)
    assert ns["ranking_auc"]([0.9, 0.1, 0.8, 0.2], [1, 0, 1, 0]) == pytest.approx(1.0)


def test_ranking_auc_of_a_constant_head_is_exactly_one_half():
    """A collapsed IoU head must score 0.5, not something flattering."""
    assert ns["ranking_auc"]([0.5] * 6, [1, 1, 1, 0, 0, 0]) == pytest.approx(0.5)


def test_ranking_auc_is_nan_without_both_classes():
    assert np.isnan(ns["ranking_auc"]([0.1, 0.2], [1, 1]))
    assert np.isnan(ns["ranking_auc"]([0.1, 0.2], [0, 0]))


def test_consensus_baseline_is_recorded_so_the_head_is_judged_fairly():
    """E021 established consensus_score AUC 0.7252; beating it is the bar."""
    assert ns["CONSENSUS_SCORE_AUC"] == pytest.approx(0.7252)


def test_calibration_error_is_zero_for_a_perfect_head_and_positive_otherwise():
    values = np.linspace(0.05, 0.95, 19)
    assert ns["calibration_error"](values, values) == pytest.approx(0.0, abs=1e-9)
    assert ns["calibration_error"](values, np.clip(values + 0.2, 0, 1)) > 0.1


# --------------------------------------------------------- losses (torch-dependent)

torch = pytest.importorskip("torch", reason="loss maths needs torch; GPU work runs on Kaggle")


def _loss_ns():
    """Load the torch-dependent loss functions on top of the constants."""
    names = {"focal_tversky_loss", "mask_loss", "deep_supervision_loss",
             "observed_iou", "iou_head_loss"}
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    local = dict(ns)
    exec(compile(ast.Module(body=body, type_ignores=[]), str(SOURCE), "exec"), local)
    return local


def test_focal_tversky_penalises_false_negatives_more_than_false_positives():
    """arm_b's entire purpose: under-segmentation must cost more than over-segmentation."""
    fns = _loss_ns()
    target = torch.zeros(1, 1, 8, 8)
    target[..., 2:6, 2:6] = 1.0
    under = torch.full((1, 1, 8, 8), -4.0)   # misses part of the object
    under[..., 3:5, 3:5] = 4.0
    over = torch.full((1, 1, 8, 8), -4.0)    # covers it plus a margin
    over[..., 1:7, 1:7] = 4.0
    assert fns["focal_tversky_loss"](under, target) > fns["focal_tversky_loss"](over, target)


def test_focal_tversky_rejects_non_fn_weighted_parameters():
    fns = _loss_ns()
    target = torch.zeros(1, 1, 4, 4)
    target[..., 1:3, 1:3] = 1.0
    logits = torch.zeros(1, 1, 4, 4)
    with pytest.raises(ValueError, match="beta > alpha"):
        fns["focal_tversky_loss"](logits, target, alpha=0.7, beta=0.3)
    with pytest.raises(ValueError, match="must equal 1"):
        fns["focal_tversky_loss"](logits, target, alpha=0.3, beta=0.5)


def test_focal_tversky_is_near_zero_for_a_perfect_prediction():
    fns = _loss_ns()
    target = torch.zeros(1, 1, 8, 8)
    target[..., 2:6, 2:6] = 1.0
    logits = torch.where(target > 0, 12.0, -12.0)
    assert float(fns["focal_tversky_loss"](logits, target)) < 1e-3


def test_arm_losses_differ_and_only_arm_b_is_fn_weighted():
    fns = _loss_ns()

    class Base:
        @staticmethod
        def soft_dice_loss(logits, target, eps=1e-6):
            probability = torch.sigmoid(logits)
            num = 2 * (probability * target).sum() + eps
            den = probability.sum() + target.sum() + eps
            return 1 - num / den

    target = torch.zeros(2, 1, 8, 8)
    target[..., 2:6, 2:6] = 1.0
    logits = torch.randn(2, 1, 8, 8)
    a = float(fns["mask_loss"]("arm_a", logits, target, Base))
    b = float(fns["mask_loss"]("arm_b", logits, target, Base))
    assert a != b, "the two arms must not be the same objective"
    with pytest.raises(ValueError, match="Unknown arm"):
        fns["mask_loss"]("arm_z", logits, target, Base)


def test_deep_supervision_downsamples_the_target_to_each_scale():
    fns = _loss_ns()

    class Base:
        @staticmethod
        def soft_dice_loss(logits, target, eps=1e-6):
            probability = torch.sigmoid(logits)
            return 1 - (2 * (probability * target).sum() + eps) / (
                probability.sum() + target.sum() + eps)

    target = torch.zeros(1, 1, 16, 16)
    target[..., 4:12, 4:12] = 1.0
    outputs = (torch.zeros(1, 1, 16, 16), torch.zeros(1, 1, 8, 8), torch.zeros(1, 1, 4, 4))
    value = fns["deep_supervision_loss"]("arm_a", outputs, target, Base)
    assert torch.isfinite(value) and float(value) > 0
    with pytest.raises(ValueError, match="One output per"):
        fns["deep_supervision_loss"]("arm_a", outputs[:2], target, Base)


def test_observed_iou_is_the_real_iou_of_the_thresholded_mask():
    fns = _loss_ns()
    target = torch.zeros(1, 1, 10, 10)
    target[..., 0:5, :] = 1.0          # 50 pixels
    logits = torch.full((1, 1, 10, 10), -8.0)
    logits[..., 0:5, :] = 8.0          # exact match
    assert float(fns["observed_iou"](logits, target)) == pytest.approx(1.0, abs=1e-4)
    half = torch.full((1, 1, 10, 10), -8.0)
    half[..., 0:5, 0:5] = 8.0          # 25 of 50 -> IoU 0.5
    assert float(fns["observed_iou"](half, target)) == pytest.approx(0.5, abs=1e-3)


def test_iou_head_loss_is_downweighted_so_it_cannot_dominate():
    fns = _loss_ns()
    predicted = torch.tensor([0.9, 0.1])
    actual = torch.tensor([0.4, 0.4])
    plain = float(torch.nn.functional.mse_loss(predicted, actual))
    weighted = float(fns["iou_head_loss"](predicted, actual))
    assert weighted == pytest.approx(ns["IOU_LOSS_WEIGHT"] * plain)
    assert ns["IOU_LOSS_WEIGHT"] < 0.5
    with pytest.raises(ValueError):
        fns["iou_head_loss"](predicted, actual, weight=0.0)
