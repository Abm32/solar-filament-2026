"""V13 composition logic: grid discipline, control verification, selection, and bars.

Extracts the pure-Python parts of v13_compose.py by AST so the GPU-only helpers
(generate_views, refine, evaluate_arm, export, run_compose) and their torch /
ultralytics / cv2 imports are never touched, matching the pattern in
test_detector_flags.py and test_v12_resolution.py.
"""
import ast
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

SOURCE = Path(__file__).resolve().parents[1] / "kaggle_inference/v13_compose.py"
tree = ast.parse(SOURCE.read_text())

constant_names = {"CELL", "CONTROL_CELL", "CONTROL", "ARM", "V11_RUN_IDENTITY",
                  "DETECTOR_RESOLUTION", "DETECTOR_CONFIDENCE_FLOOR", "POLICY",
                  "MATCH_IOU", "MODE", "MIN_AREA", "COUNTS", "THRESHOLDS",
                  "CONFIDENCES", "CONTROL_REFERENCE", "PARENTS", "SUBMIT_BAR",
                  "PROMOTION_GATE"}
function_names = {"pq", "candidate_key", "declared_candidates", "cache_id",
                  "verify_grid_matches_frozen_constants", "verify_control",
                  "select_candidate", "evaluate_bars", "digest_json",
                  "serialized_rle", "restored_rle"}
body = [node for node in tree.body
        if (isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in constant_names for t in node.targets))
        or (isinstance(node, ast.FunctionDef) and node.name in function_names)]
module = ast.Module(body=body, type_ignores=[])
ns = {"pd": pd, "Path": Path, "hashlib": hashlib, "json": json}
exec(compile(module, str(SOURCE), "exec"), ns)


class FrozenTTA:
    """Stand-in for detector_tta's frozen tuning constants."""
    LOCAL_THRESHOLDS = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60)
    LOCAL_CONFIDENCES = (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55)


def test_cell_is_exactly_the_winning_e_a_cell():
    cell = ns["CELL"]
    # E023's winner was box iou=0.3 with retina_masks/agnostic_nms both True.
    assert cell["iou"] == 0.3
    assert cell["retina_masks"] is True
    assert cell["agnostic_nms"] is True
    control = ns["CONTROL_CELL"]
    assert control["iou"] == 0.7, "the control must be the Ultralytics default"
    assert control["retina_masks"] is True and control["agnostic_nms"] is True
    assert cell["name"] != control["name"]


def test_grid_matches_the_frozen_forty_two_candidate_e_a_grid():
    assert ns["verify_grid_matches_frozen_constants"](FrozenTTA) is True
    assert len(ns["declared_candidates"]()) == 42
    assert len(set(ns["declared_candidates"]())) == 42


def test_grid_verification_rejects_a_widened_threshold_grid():
    class Widened(FrozenTTA):
        LOCAL_THRESHOLDS = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)
    with pytest.raises(ValueError, match="Threshold grid differs"):
        ns["verify_grid_matches_frozen_constants"](Widened)


def test_grid_verification_rejects_a_widened_confidence_grid():
    class Widened(FrozenTTA):
        LOCAL_CONFIDENCES = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55)
    with pytest.raises(ValueError, match="Confidence grid differs"):
        ns["verify_grid_matches_frozen_constants"](Widened)


def test_control_reference_pins_the_accepted_e015_counts():
    reference = ns["CONTROL_REFERENCE"]
    assert reference["candidate"] == "t0.40_c0.40"
    assert reference["tune"] == [pytest.approx(426.5473399071961), 626, 211, 508]
    assert reference["audit"] == [pytest.approx(182.75261599994468), 265, 88, 194]
    # The control candidate must itself be inside the predeclared grid.
    assert reference["candidate"] in ns["declared_candidates"]()


def test_pq_matches_the_panoptic_quality_definition():
    assert ns["pq"]([426.5473399071961, 626, 211, 508]) == pytest.approx(0.4328232774299302)
    assert ns["pq"]([182.75261599994468, 265, 88, 194]) == pytest.approx(0.4501295960589769)
    assert ns["pq"]([0.0, 0, 0, 0]) == 0.0


def test_verify_control_accepts_the_exact_frozen_counts():
    reference = ns["CONTROL_REFERENCE"]
    results = {reference["candidate"]: dict(tune=list(reference["tune"]),
                                            audit=list(reference["audit"]))}
    observed = ns["verify_control"](results)
    assert observed["tune"][1:] == reference["tune"][1:]


def test_verify_control_rejects_a_single_changed_integer_count():
    reference = ns["CONTROL_REFERENCE"]
    drifted = list(reference["tune"])
    drifted[1] += 1
    results = {reference["candidate"]: dict(tune=drifted, audit=list(reference["audit"]))}
    with pytest.raises(ValueError, match="Control tune counts differ"):
        ns["verify_control"](results)


def test_verify_control_rejects_a_drifted_iou_sum():
    reference = ns["CONTROL_REFERENCE"]
    drifted = list(reference["audit"])
    drifted[0] += 1e-3
    results = {reference["candidate"]: dict(tune=list(reference["tune"]), audit=drifted)}
    with pytest.raises(ValueError, match="Control audit counts differ"):
        ns["verify_control"](results)


def test_cache_id_separates_iou030_from_the_frozen_iou070_caches():
    """A collision here would silently score iou=0.7 proposals as if they were 0.3."""
    name, split = "20141006224634Lh.jpeg", "train"
    a = ns["cache_id"](name, split, ns["CELL"]["name"])
    b = ns["cache_id"](name, split, ns["CONTROL_CELL"]["name"])
    assert a != b
    assert ns["cache_id"](name, "test", ns["CELL"]["name"]) != a, "splits must not collide"
    assert a.endswith(".json.gz") and len(a) == len("0123456789abcdef.json.gz")


def test_cache_id_rejects_paths_and_unknown_splits():
    with pytest.raises(ValueError):
        ns["cache_id"]("sub/dir/a.jpeg", "train", "ea_iou030")
    with pytest.raises(ValueError):
        ns["cache_id"]("a.jpeg", "validation", "ea_iou030")


def test_rle_round_trip_preserves_counts():
    rle = dict(size=[2048, 2048], counts=b"abc123")
    assert ns["restored_rle"](ns["serialized_rle"](rle)) == rle


def _results(pairs):
    """{candidate: (tune_pq_target, audit_pq_target)} -> count structures.

    Uses tp only, so PQ reduces to iou_sum/tp and the target is exact.
    """
    out = {}
    for key, (tune, audit) in pairs.items():
        out[key] = dict(tune=[tune * 100, 100, 0, 0], audit=[audit * 100, 100, 0, 0])
    return out


def test_select_candidate_maximizes_tuning_pq_only():
    results = _results({"t0.40_c0.30": (0.4400, 0.1000),
                        "t0.40_c0.40": (0.4300, 0.9000),
                        "t0.45_c0.30": (0.4350, 0.5000)})
    # The 0.90-audit row must NOT win; audit never selects.
    assert ns["select_candidate"](results) == "t0.40_c0.30"


def test_select_candidate_rejects_a_candidate_outside_the_declared_grid():
    results = _results({"t0.40_c0.30": (0.44, 0.45), "t0.99_c0.99": (0.99, 0.99)})
    with pytest.raises(ValueError, match="outside the predeclared grid"):
        ns["select_candidate"](results)


def test_submit_bar_is_anchored_to_the_better_parent_and_the_accepted_control():
    bar, parents = ns["SUBMIT_BAR"], ns["PARENTS"]
    assert bar["min_tune_pq"] == pytest.approx(parents["ea_iou030"]["tune_pq"])
    assert bar["min_audit_pq"] == pytest.approx(parents["e015_control"]["audit_pq"])
    assert parents["v11_proposal_centered_e4"]["public_score"] == 0.39


def test_export_authorized_when_composition_matches_the_better_parent():
    results = _results({"t0.40_c0.30": (0.4400, 0.4600)})
    report = ns["evaluate_bars"]("t0.40_c0.30", results)
    assert report["export_authorized"] is True
    assert report["recommendation"] == "export and measure one submission"


def test_export_blocked_when_composition_falls_below_the_better_parent():
    results = _results({"t0.40_c0.30": (0.4350, 0.4600)})  # below E-A's 0.436726
    report = ns["evaluate_bars"]("t0.40_c0.30", results)
    assert report["export_authorized"] is False
    assert report["promotable"] is False
    assert "do not spend a submission" in report["recommendation"]


def test_export_blocked_when_audit_regresses_below_the_accepted_control():
    results = _results({"t0.40_c0.30": (0.4400, 0.4400)})  # audit under E015's 0.450130
    report = ns["evaluate_bars"]("t0.40_c0.30", results)
    assert report["export_authorized"] is False


def test_promotion_requires_beating_v11_by_the_declared_margin_on_tune():
    v11 = ns["PARENTS"]["v11_proposal_centered_e4"]
    gate = ns["PROMOTION_GATE"]
    assert gate["required_tune_delta_vs_v11"] == pytest.approx(0.003)
    # Just short of the margin: authorized to measure, not to promote.
    short = _results({"t0.40_c0.30": (v11["tune_pq"] + 0.002, v11["audit_pq"] + 0.001)})
    assert ns["evaluate_bars"]("t0.40_c0.30", short)["promotable"] is False
    clear = _results({"t0.40_c0.30": (v11["tune_pq"] + 0.003, v11["audit_pq"] + 0.001)})
    report = ns["evaluate_bars"]("t0.40_c0.30", clear)
    assert report["promotable"] is True
    assert report["recommendation"] == "promote"


def test_promotion_blocked_when_audit_falls_below_v11_despite_a_tune_win():
    v11 = ns["PARENTS"]["v11_proposal_centered_e4"]
    results = _results({"t0.40_c0.30": (v11["tune_pq"] + 0.010, v11["audit_pq"] - 0.001)})
    report = ns["evaluate_bars"]("t0.40_c0.30", results)
    assert report["export_authorized"] is True
    assert report["promotable"] is False


def test_report_records_signed_deltas_against_all_three_parents():
    results = _results({"t0.40_c0.30": (0.4400, 0.4600)})
    deltas = ns["evaluate_bars"]("t0.40_c0.30", results)["deltas_vs_parents"]
    assert set(deltas) == {"e015_control", "ea_iou030", "v11_proposal_centered_e4"}
    assert deltas["e015_control"]["tune"] == pytest.approx(0.4400 - 0.4328232774299302)
    assert deltas["v11_proposal_centered_e4"]["tune"] == pytest.approx(0.4400 - 0.438001)
    # A composition below a parent must be reported as negative, not hidden.
    worse = _results({"t0.40_c0.30": (0.4300, 0.4400)})
    d = ns["evaluate_bars"]("t0.40_c0.30", worse)["deltas_vs_parents"]
    assert d["ea_iou030"]["tune"] < 0 and d["v11_proposal_centered_e4"]["tune"] < 0


def test_digest_json_is_order_independent_and_rejects_nan():
    assert ns["digest_json"](dict(a=1, b=2)) == ns["digest_json"](dict(b=2, a=1))
    with pytest.raises(ValueError):
        ns["digest_json"](dict(a=float("nan")))
