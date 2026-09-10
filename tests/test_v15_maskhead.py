"""V15 logic: the three V14 defect fixes, grid discipline, selection, and bars.

Extracts the pure-Python parts of v15_maskhead.py by AST so the GPU-only helpers
(build_mask_head, losses, run_experiment) and their torch imports are never touched,
matching the pattern in test_v13_compose.py and test_v12_resolution.py.

The tests that matter most here are the ones that would have caught V14:
  * test_scoring_cohort_* -- rejects a positives-only cohort (defect 2)
  * test_head_alignment_* -- catches a misaligned predicted/actual diagnostic (defect 3)
  * test_constant_head_*  -- proves a constant head is exposed, not flattered
  * test_build_mask_head_refuses_random_init -- defect 1, checked by AST
"""
import ast
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1] / "kaggle_inference/v15_maskhead.py"
tree = ast.parse(SOURCE.read_text())

constant_names = {"CONTROL", "ARM", "V11_RUN_IDENTITY", "V11_SOURCE_RUN", "CELL",
                  "CONTROL_CELL", "DETECTOR_RESOLUTION", "DETECTOR_CONFIDENCE_FLOOR",
                  "POLICY", "MATCH_IOU", "MODE", "MIN_AREA", "COUNTS", "THRESHOLDS",
                  "CONFIDENCES", "CROP_SIZE", "BATCH_SIZE", "EPOCHS", "LEARNING_RATE",
                  "WEIGHT_DECAY", "SEED", "IOU_LOSS_WEIGHT",
                  "DEEP_SUPERVISION_WEIGHTS", "MIN_NEGATIVE_FRACTION",
                  "NEGATIVE_IOU_CEILING", "KEEP_FRACTIONS", "CONTROL_REFERENCE",
                  "ACCEPTED_CONTROL_CSV_SHA256", "BASELINES", "SUBMIT_BAR",
                  "PROMOTION_GATE", "CONSENSUS_SCORE_AUC"}
function_names = {"sha256", "digest_json", "pq", "axis_metrics", "candidate_key",
                  "declared_candidates", "verify_grid",
                  "verify_grid_matches_frozen_constants", "cache_id",
                  "focal_tversky_disabled_note", "ranking_auc", "calibration_error",
                  "verify_head_alignment", "verify_scoring_cohort",
                  "keep_fraction_frontier", "verify_control", "select_candidate",
                  "evaluate_bars"}
body = [node for node in tree.body
        if (isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in constant_names for t in node.targets))
        or (isinstance(node, ast.FunctionDef) and node.name in function_names)]
module = ast.Module(body=body, type_ignores=[])
ns = {"np": np, "Path": Path, "hashlib": hashlib, "json": json, "math": math}
exec(compile(module, str(SOURCE), "exec"), ns)


class FrozenTTA:
    """Stand-in for detector_tta's frozen tuning constants."""
    LOCAL_THRESHOLDS = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60)
    LOCAL_CONFIDENCES = (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55)


def counts(iou_sum, tp, fp, fn):
    return [iou_sum, tp, fp, fn]


def results_at(tune, audit):
    return {"t0.45_c0.30": {"tune": tune, "audit": audit}}


# --- grid discipline -------------------------------------------------------------

def test_grid_is_exactly_v13s_42_candidates():
    assert ns["verify_grid"]() is True
    assert len(ns["declared_candidates"]()) == 42
    assert len(set(ns["declared_candidates"]())) == 42


def test_grid_matches_the_frozen_sweep_constants():
    assert ns["verify_grid_matches_frozen_constants"](FrozenTTA) is True


def test_grid_rejects_drifted_thresholds():
    class Drifted(FrozenTTA):
        LOCAL_THRESHOLDS = (0.35, 0.40)
    with pytest.raises(ValueError, match="Thresholds drifted"):
        ns["verify_grid_matches_frozen_constants"](Drifted)


def test_grid_rejects_drifted_confidences():
    class Drifted(FrozenTTA):
        LOCAL_CONFIDENCES = (0.30,)
    with pytest.raises(ValueError, match="Confidences drifted"):
        ns["verify_grid_matches_frozen_constants"](Drifted)


def test_selection_grid_is_unchanged_from_v13():
    """The whole V15 claim is that only the refiner changed."""
    assert ns["THRESHOLDS"] == (0.35, 0.40, 0.45, 0.50, 0.55, 0.60)
    assert ns["CONFIDENCES"] == (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55)
    assert ns["MODE"] == "refined_constrained"
    assert ns["CELL"] == dict(name="ea_iou030", retina_masks=True, agnostic_nms=True,
                              iou=0.3)


def test_control_candidate_is_inside_the_declared_grid():
    assert ns["CONTROL_REFERENCE"]["candidate"] in ns["declared_candidates"]()


def test_cache_id_is_cell_scoped():
    """iou=0.3 proposals must never collide with the frozen iou=0.7 caches."""
    a = ns["cache_id"]("x.jpeg", "train", "ea_iou030")
    b = ns["cache_id"]("x.jpeg", "train", "frozen_control")
    assert a != b and a.endswith(".json.gz")


def test_cache_id_rejects_paths_and_bad_splits():
    with pytest.raises(ValueError):
        ns["cache_id"]("sub/x.jpeg", "train", "ea_iou030")
    with pytest.raises(ValueError):
        ns["cache_id"]("x.jpeg", "validation", "ea_iou030")


# --- defect 2: the scoring cohort must contain negatives -------------------------

def test_scoring_cohort_rejects_positives_only():
    """V14's exact failure: a cohort of matched proposals only."""
    with pytest.raises(ValueError, match="negatives"):
        ns["verify_scoring_cohort"](np.linspace(0.31, 0.91, 200))


def test_scoring_cohort_accepts_a_realistic_population():
    """Mirrors the measured V14 build distribution: ~70% below IoU 0.5."""
    population = np.concatenate([np.zeros(400), np.linspace(0.0, 0.5, 400),
                                 np.linspace(0.5, 0.87, 300)])
    health = ns["verify_scoring_cohort"](population)
    assert health["negative_fraction"] >= ns["MIN_NEGATIVE_FRACTION"]
    assert health["samples"] == len(population)


def test_scoring_cohort_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        ns["verify_scoring_cohort"]([])


def test_scoring_cohort_threshold_is_the_declared_one():
    assert ns["MIN_NEGATIVE_FRACTION"] == 0.20
    assert ns["NEGATIVE_IOU_CEILING"] == 0.10


# --- defect 3: the head diagnostic must actually measure the head ----------------

def test_head_alignment_detects_a_real_signal():
    actual = np.linspace(0, 1, 200)
    alignment = ns["verify_head_alignment"](actual.copy(), actual)
    assert alignment["observed_auc"] == pytest.approx(1.0)
    assert alignment["permutation_gap"] > 0.3


def test_head_alignment_exposes_a_misaligned_diagnostic():
    """A shuffled-vs-truth pairing must look like noise, as V14's did."""
    rng = np.random.default_rng(0)
    actual = np.linspace(0, 1, 400)
    scores = rng.permutation(actual)
    alignment = ns["verify_head_alignment"](scores, actual)
    assert abs(alignment["observed_auc"] - 0.5) < 0.12
    assert alignment["permutation_gap"] < 0.2


def test_constant_head_scores_exactly_half():
    """The V14 head was effectively constant; AUC must not flatter it."""
    actual = np.linspace(0, 1, 100)
    assert ns["ranking_auc"](np.full(100, 0.65), actual > 0.5) == pytest.approx(0.5)


def test_ranking_auc_requires_both_classes():
    assert math.isnan(ns["ranking_auc"]([0.1, 0.2], [True, True]))


def test_ranking_auc_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        ns["ranking_auc"]([0.1, 0.2], [True])


def test_calibration_error_is_mean_absolute():
    assert ns["calibration_error"]([0.5, 0.5], [0.4, 0.6]) == pytest.approx(0.1)


# --- the learned score is reported, never used to select -------------------------

def test_keep_fraction_frontier_is_monotone_in_kept_count():
    rng = np.random.default_rng(1)
    actual = rng.random(300)
    frontier = ns["keep_fraction_frontier"](actual.copy(), actual, 400)
    assert [row["keep_fraction"] for row in frontier] == list(ns["KEEP_FRACTIONS"])
    assert [row["kept"] for row in frontier] == sorted(row["kept"] for row in frontier)


def test_keep_fraction_frontier_uses_fractions_not_absolute_cuts():
    """A head with a narrow output range must still produce a usable curve.

    This is the reporting half of the defect-2 fix: V14's absolute gates all filtered
    zero proposals because every score sat above them.
    """
    actual = np.linspace(0, 1, 100)
    narrow = 0.6 + 0.001 * actual
    frontier = ns["keep_fraction_frontier"](narrow, actual, 120)
    assert len(frontier) == len(ns["KEEP_FRACTIONS"])
    assert frontier[0]["kept"] == 10


def test_keep_fraction_frontier_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        ns["keep_fraction_frontier"]([0.1, 0.2], [0.3], 10)


def test_keep_fraction_frontier_handles_empty():
    assert ns["keep_fraction_frontier"]([], [], 10) == []


# --- control verification --------------------------------------------------------

def test_control_verification_accepts_the_frozen_reference():
    reference = ns["CONTROL_REFERENCE"]
    verified = ns["verify_control"]({reference["candidate"]:
                                     {"tune": list(reference["tune"]),
                                      "audit": list(reference["audit"])}})
    assert verified["tune"] == reference["tune"]


def test_control_verification_rejects_drifted_counts():
    reference = ns["CONTROL_REFERENCE"]
    drifted = list(reference["tune"])
    drifted[1] += 1
    with pytest.raises(ValueError, match="Control tune counts differ"):
        ns["verify_control"]({reference["candidate"]:
                              {"tune": drifted, "audit": list(reference["audit"])}})


def test_accepted_control_csv_hash_is_the_038_file():
    assert ns["ACCEPTED_CONTROL_CSV_SHA256"] == (
        "bf11690e9bcce73b0c627096cc1198d9fb0916650f7fa4b3f08f1167bb76bde9")


# --- selection and the predeclared bars ------------------------------------------

def test_selection_is_tune_only():
    """Audit must never select, even when it strongly prefers the other candidate."""
    results = {"t0.45_c0.30": {"tune": counts(100, 200, 50, 50), "audit": counts(1, 1, 99, 99)},
               "t0.40_c0.30": {"tune": counts(90, 180, 50, 50), "audit": counts(500, 400, 1, 1)}}
    assert ns["select_candidate"](results) == "t0.45_c0.30"


def test_selection_rejects_undeclared_candidates():
    with pytest.raises(ValueError, match="outside the predeclared grid"):
        ns["select_candidate"]({"t0.99_c0.99": {"tune": counts(1, 1, 0, 0),
                                               "audit": counts(1, 1, 0, 0)}})


def test_selection_rejects_empty_results():
    with pytest.raises(ValueError):
        ns["select_candidate"]({})


def test_v14_style_result_is_not_authorized_to_export():
    """V14's actual numbers must fail SUBMIT_BAR."""
    report = ns["evaluate_bars"]("t0.45_c0.30",
                                 results_at(counts(478.79, 725, 2051, 409),
                                            counts(191.34, 286, 859, 173)))
    assert report["export_authorized"] is False
    assert report["promotable"] is False
    assert "do not submit" in report["recommendation"]


def test_v13_parity_clears_submit_bar_but_not_the_promotion_gate():
    """The E024 lesson encoded: a tie is measurable, not promotable."""
    v13 = ns["BASELINES"]["v13_compose"]
    tune = counts(v13["tune_pq"] * (686 + 0.5 * (309 + 448)), 686, 309, 448)
    audit = counts(v13["audit_pq"] * (265 + 0.5 * (88 + 194)), 265, 88, 194)
    report = ns["evaluate_bars"]("t0.45_c0.30", results_at(tune, audit))
    assert report["export_authorized"] is True
    assert report["promotable"] is False
    assert "do not spend a submission" in report["recommendation"]


def test_a_large_gain_is_promotable():
    tune = counts(0.46 * (800 + 0.5 * (250 + 334)), 800, 250, 334)
    audit = counts(0.48 * (300 + 0.5 * (80 + 159)), 300, 80, 159)
    report = ns["evaluate_bars"]("t0.45_c0.30", results_at(tune, audit))
    assert report["selected_tune_pq"] - ns["BASELINES"]["v13_compose"]["tune_pq"] >= 0.010
    assert report["promotable"] is True
    assert report["recommendation"] == "promote and submit"


def test_promotion_requires_audit_not_to_regress():
    """Tune clears +0.010 but audit falls below V13, so it must not promote."""
    tune = counts(0.46 * (800 + 0.5 * (250 + 334)), 800, 250, 334)
    audit = counts(0.4502 * (265 + 0.5 * (88 + 194)), 265, 88, 194)
    report = ns["evaluate_bars"]("t0.45_c0.30", results_at(tune, audit))
    assert report["export_authorized"] is True
    assert report["promotable"] is False


def test_submit_bar_and_gate_are_bound_to_measured_baselines():
    assert ns["SUBMIT_BAR"]["min_tune_pq"] == ns["BASELINES"]["v13_compose"]["tune_pq"]
    assert ns["SUBMIT_BAR"]["min_audit_pq"] == ns["BASELINES"]["e015_control"]["audit_pq"]
    assert ns["PROMOTION_GATE"]["required_tune_delta_vs_v13"] == 0.010


# --- the axis question this run exists to answer ---------------------------------

def test_report_states_whether_the_capacity_hypothesis_is_supported():
    """meanIoU above V13's 0.6855 is the actual result V15 is measuring."""
    tune = counts(0.75 * 700, 700, 300, 434)
    report = ns["evaluate_bars"]("t0.45_c0.30", results_at(tune, counts(200, 280, 90, 179)))
    assert report["capacity_hypothesis"]["supported"] is True
    assert report["capacity_hypothesis"]["v15_mean_matched_iou"] == pytest.approx(0.75)


def test_capacity_hypothesis_unsupported_when_meaniou_regresses():
    """V14's 0.6604 against V13's 0.6855."""
    tune = counts(0.6604 * 725, 725, 309, 409)
    report = ns["evaluate_bars"]("t0.45_c0.30", results_at(tune, counts(190, 286, 88, 173)))
    assert report["capacity_hypothesis"]["supported"] is False


def test_axis_metrics_separates_the_two_axes():
    axes = ns["axis_metrics"](counts(478.79, 725, 2051, 409))
    assert axes["recall"] == pytest.approx(725 / 1134, abs=1e-6)
    assert axes["mean_matched_iou"] == pytest.approx(478.79 / 725, abs=1e-6)
    assert axes["fp"] == 2051


def test_axis_metrics_handles_no_predictions():
    axes = ns["axis_metrics"](counts(0, 0, 0, 1134))
    assert axes["recall"] == 0.0 and axes["mean_matched_iou"] == 0.0


def test_pq_is_zero_when_nothing_is_counted():
    assert ns["pq"](counts(0, 0, 0, 0)) == 0.0


# --- defect 1, checked structurally ---------------------------------------------

def test_build_mask_head_always_requests_imagenet_weights():
    """V14 did `resnet34(weights=DEFAULT if pretrained else None)` with pretrained
    silently False. V15 must have no conditional on the weights argument.

    Checked against the AST, not the raw text, so the module docstring is free to
    quote the old defective line while describing it.
    """
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "build_mask_head")
    calls = [node for node in ast.walk(function)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
             and node.func.id == "resnet34"]
    assert len(calls) == 1
    weights = next(kw.value for kw in calls[0].keywords if kw.arg == "weights")
    # An ast.IfExp here is precisely the V14 defect.
    assert not isinstance(weights, ast.IfExp), "weights must not be conditional"
    assert ast.unparse(weights) == "ResNet34_Weights.DEFAULT"


def test_build_mask_head_refuses_random_init():
    """Calling with pretrained=False must raise before any torch work happens."""
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "build_mask_head")
    guard = next(node for node in ast.walk(function)
                 if isinstance(node, ast.If) and isinstance(node.test, ast.UnaryOp))
    assert any(isinstance(n, ast.Raise) for n in ast.walk(guard))
    assert function.args.defaults[-1].value is True


def test_focal_tversky_is_documented_as_open_not_retired():
    note = ns["focal_tversky_disabled_note"]()
    assert "NOT retired" in note and "NOT tested here" in note


# --- the negative-record training path (the first smoke failure) -----------------

RUNTIME = Path(__file__).resolve().parents[1] / "kaggle_inference/v15_runtime.py"
runtime_tree = ast.parse(RUNTIME.read_text())


def scoring_cohort_body():
    """build_scoring_cohort's source with the docstring stripped.

    The docstring deliberately quotes `by_file` and `detector` while explaining what
    the function must not do, so structural checks must look at code only.
    """
    function = next(n for n in runtime_tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "build_scoring_cohort")
    body = function.body[1:] if isinstance(function.body[0], ast.Expr) else function.body
    return "\n".join(ast.unparse(node) for node in body)


def test_scoring_cohort_never_reads_the_heldout_split_by_file():
    """The first smoke run died with KeyError on a TRAINING filename because
    runtime["split"]'s by_file only covers held-out validation files."""
    source = scoring_cohort_body()
    assert "by_file" not in source
    # It must derive its own file->image_id map from the cohort's own sources.
    assert "training_sources" in source


def test_scoring_cohort_reuses_cached_proposals_not_the_detector():
    """Negatives must cost zero extra detector inference: it reads the caches that
    build_training_cohort already wrote and never predicts."""
    source = scoring_cohort_body()
    assert "read_views" in source
    for forbidden in ("YOLO", ".predict(", "generate_views", "arm_views"):
        assert forbidden not in source


def test_negative_records_carry_no_gt_target():
    """A negative has no `target` key, so nothing can read GT that is absent."""
    dataset = next(n for n in runtime_tree.body if isinstance(n, ast.ClassDef)
                   and n.name == "MaskHeadDataset")
    getitem = next(n for n in dataset.body if isinstance(n, ast.FunctionDef)
                   and n.name == "__getitem__")
    source = ast.unparse(getitem)
    assert 'record.get(\'negative\'' in source or 'record.get("negative"' in source
    assert "np.zeros_like(seed)" in source


def test_mask_loss_is_restricted_to_positive_samples():
    """Negatives must train the IoU head ONLY, or the meanIoU comparison to V13
    stops being apples-to-apples."""
    function = next(n for n in runtime_tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "train")
    source = ast.unparse(function)
    assert "keep = weight > 0" in source
    assert "o[keep] for o in outputs" in source


def test_dataset_returns_a_three_tuple_with_the_weight():
    dataset = next(n for n in runtime_tree.body if isinstance(n, ast.ClassDef)
                   and n.name == "MaskHeadDataset")
    getitem = next(n for n in dataset.body if isinstance(n, ast.FunctionDef)
                   and n.name == "__getitem__")
    returns = [n for n in ast.walk(getitem) if isinstance(n, ast.Return)]
    assert len(returns) == 1
    assert isinstance(returns[0].value, ast.Tuple) and len(returns[0].value.elts) == 3


def test_scoring_cohort_counts_negatives_as_zero_iou():
    """verify_scoring_cohort must be fed 0.0 for negatives, not their matched_iou."""
    function = next(n for n in runtime_tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "build_scoring_cohort")
    assert "0.0 if r['negative'] else r['matched_iou']" in ast.unparse(function)
