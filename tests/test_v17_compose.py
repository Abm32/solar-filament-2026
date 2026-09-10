"""Tests for the V17 composition: mask_ratio=1 detector x V13's measured recipe.

V17's whole value depends on three things being airtight, so they are pinned here:

  1. **Cache-id disjointness.** Five proposal populations now exist for the same image
     (V6@0.7 frozen, V6@0.3 from V13, and arm_r1@{0.3,0.7} for each of two seeds).
     Silently reusing one for another would invalidate the comparison, exactly the class
     of error the project's cell-scoped cache ids were introduced to prevent.
  2. **The predeclared grid cannot widen.** 2 cells x 6 thresholds x 7 confidences = 84,
     bound to detector_tta's frozen constants, with selection refusing anything outside.
  3. **The bars are V13's measured numbers**, so "cleared the bar" means beating the best
     candidate ever measured, not beating a convenient baseline.

Follows test_v13_compose.py: the pure-Python parts are extracted by AST so the GPU-only
helpers (generate_views, refine, evaluate_arm, export, run_compose) and their torch /
ultralytics imports are never touched, since this environment has no GPU.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def extract(source_path, constant_names, function_names):
    """Exec only the CPU-safe constants and pure functions from a runtime module."""
    tree = ast.parse(Path(source_path).read_text())
    body = [node for node in tree.body
            if (isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id in constant_names for t in node.targets))
            or (isinstance(node, ast.FunctionDef) and node.name in function_names)]
    module = ast.Module(body=body, type_ignores=[])
    namespace = {"pd": pd, "np": np, "Path": Path, "hashlib": hashlib, "json": json}
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace


V17_CONSTANTS = {"CELLS", "CONTROL_CELL", "CONTROL", "ARM_DETECTORS", "PRIMARY_SEED",
                 "MASK_RATIO", "ARM", "V11_RUN_IDENTITY", "V6_DETECTOR_SHA256",
                 "DETECTOR_RESOLUTION", "DETECTOR_CONFIDENCE_FLOOR", "POLICY", "MATCH_IOU",
                 "MODE", "MIN_AREA", "COUNTS", "THRESHOLDS", "CONFIDENCES",
                 "CONTROL_REFERENCE", "PARENTS", "SUBMIT_BAR", "PROMOTION_GATE",
                 "V16_TRAINING_NOISE_FLOOR"}
V17_FUNCTIONS = {"pq", "candidate_key", "cell_candidate_key", "declared_candidates",
                 "cache_id", "verify_grid_matches_frozen_constants", "verify_control",
                 "select_candidate", "evaluate_bars", "seed_agreement", "digest_json",
                 "serialized_rle", "restored_rle"}

v17 = extract(ROOT / "kaggle_inference/v17_compose.py", V17_CONSTANTS, V17_FUNCTIONS)
v13 = extract(ROOT / "kaggle_inference/v13_compose.py",
              {"CELL", "CONTROL_CELL", "THRESHOLDS", "CONFIDENCES", "CONTROL_REFERENCE",
               "DETECTOR_RESOLUTION", "DETECTOR_CONFIDENCE_FLOOR"},
              {"cache_id", "candidate_key", "declared_candidates"})


class FrozenTTA:
    """Stand-in for detector_tta's frozen tuning constants."""
    LOCAL_THRESHOLDS = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60)
    LOCAL_CONFIDENCES = (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55)


# --- declared configuration -------------------------------------------------------

def test_cells_are_the_declared_box_iou_pair():
    assert {cell["iou"] for cell in v17["CELLS"]} == {0.3, 0.7}
    assert v17["CONTROL_CELL"]["iou"] == 0.7
    assert v17["MASK_RATIO"] == 1
    # retina_masks/agnostic_nms must match E-A/V13's winning cell byte-for-byte.
    for cell in v17["CELLS"]:
        assert cell["retina_masks"] is True
        assert cell["agnostic_nms"] is True


def test_grid_is_exactly_84_candidates_and_matches_frozen_constants():
    assert v17["THRESHOLDS"] == v13["THRESHOLDS"]
    assert v17["CONFIDENCES"] == v13["CONFIDENCES"]
    candidates = v17["declared_candidates"]()
    assert len(candidates) == 84 == len(set(candidates))
    assert v17["verify_grid_matches_frozen_constants"](FrozenTTA) is True


def test_grid_verification_rejects_a_widened_threshold_set():
    class Widened:
        LOCAL_THRESHOLDS = tuple(list(v17["THRESHOLDS"]) + [0.65])
        LOCAL_CONFIDENCES = v17["CONFIDENCES"]
    with pytest.raises(ValueError, match="Threshold grid differs"):
        v17["verify_grid_matches_frozen_constants"](Widened)


def test_grid_verification_rejects_a_widened_confidence_set():
    class Widened:
        LOCAL_THRESHOLDS = v17["THRESHOLDS"]
        LOCAL_CONFIDENCES = tuple(list(v17["CONFIDENCES"]) + [0.60])
    with pytest.raises(ValueError, match="Confidence grid differs"):
        v17["verify_grid_matches_frozen_constants"](Widened)


def test_both_v16_seeds_are_declared_with_distinct_detectors():
    assert set(v17["ARM_DETECTORS"]) == {"seed1", "seed2"}
    digests = {spec["detector_sha256"] for spec in v17["ARM_DETECTORS"].values()}
    assert len(digests) == 2
    assert v17["PRIMARY_SEED"] in v17["ARM_DETECTORS"]


def test_declared_detector_hashes_match_the_completed_v16_runs():
    """The pinned hashes must be the ones V16 actually recorded."""
    for label, output in (("seed1", "outputs/kaggle_v16_full"),
                          ("seed2", "outputs/kaggle_v16_full_seed2")):
        manifest_path = ROOT / output / "v16_manifest.json"
        if not manifest_path.exists():
            pytest.skip(f"{output} not pulled")
        recorded = json.loads(manifest_path.read_text())["training"]["arm_r1"]
        assert recorded["mask_ratio"] == 1
        assert v17["ARM_DETECTORS"][label]["detector_sha256"] == recorded["checkpoint_sha256"]


# --- cache identity ---------------------------------------------------------------

def test_cache_ids_are_disjoint_across_cells_seeds_and_detectors():
    """No two proposal populations may ever share a cache file name."""
    seeds = [spec["detector_sha256"] for spec in v17["ARM_DETECTORS"].values()]
    ids = set()
    for detector in seeds + [v17["V6_DETECTOR_SHA256"]]:
        for cell in v17["CELLS"]:
            for split in ("train", "test"):
                ids.add(v17["cache_id"]("a.jpeg", split, cell["name"], detector))
    # 3 detectors x 2 cells x 2 splits, all distinct.
    assert len(ids) == 12


def test_cache_id_differs_from_v13s_scheme_for_the_same_cell():
    """V13's iou=0.3 caches must never be mistaken for V17's mask_ratio=1 ones."""
    v13_id = v13["cache_id"]("a.jpeg", "train", "ea_iou030")
    v17_id = v17["cache_id"]("a.jpeg", "train", "mr1_iou030",
                             v17["ARM_DETECTORS"]["seed1"]["detector_sha256"])
    assert v13_id != v17_id


def test_cache_id_varies_with_mask_ratio():
    seed = v17["ARM_DETECTORS"]["seed1"]["detector_sha256"]
    assert (v17["cache_id"]("a.jpeg", "train", "mr1_iou030", seed, mask_ratio=1)
            != v17["cache_id"]("a.jpeg", "train", "mr1_iou030", seed, mask_ratio=2))


def test_cache_id_rejects_paths_and_unknown_splits():
    seed = v17["ARM_DETECTORS"]["seed1"]["detector_sha256"]
    with pytest.raises(ValueError):
        v17["cache_id"]("dir/a.jpeg", "train", "mr1_iou030", seed)
    with pytest.raises(ValueError):
        v17["cache_id"]("a.jpeg", "validation", "mr1_iou030", seed)


# --- selection --------------------------------------------------------------------

def counts(iou_sum, tp, fp, fn):
    return [iou_sum, tp, fp, fn]


def test_selection_is_tune_only_and_ignores_a_better_audit():
    results = {
        "mr1_iou030/t0.45_c0.30": dict(tune=counts(10.0, 20, 0, 0), audit=counts(1.0, 20, 0, 0)),
        "mr1_iou070/t0.45_c0.30": dict(tune=counts(9.0, 20, 0, 0), audit=counts(99.0, 20, 0, 0)),
    }
    assert v17["select_candidate"](results) == "mr1_iou030/t0.45_c0.30"


def test_selection_rejects_candidates_outside_the_declared_grid():
    with pytest.raises(ValueError, match="outside the predeclared grid"):
        v17["select_candidate"]({"mr1_iou030/t0.99_c0.99": dict(tune=counts(1.0, 1, 0, 0),
                                                               audit=counts(1.0, 1, 0, 0))})


def test_selection_breaks_ties_deterministically_by_name():
    results = {
        "mr1_iou070/t0.45_c0.30": dict(tune=counts(5.0, 10, 0, 0), audit=counts(1.0, 10, 0, 0)),
        "mr1_iou030/t0.45_c0.30": dict(tune=counts(5.0, 10, 0, 0), audit=counts(1.0, 10, 0, 0)),
    }
    reversed_results = dict(reversed(list(results.items())))
    assert v17["select_candidate"](results) == v17["select_candidate"](reversed_results)


# --- bars -------------------------------------------------------------------------

def test_submit_bar_is_v13s_measured_result():
    assert v17["SUBMIT_BAR"]["min_tune_pq"] == 0.441777
    assert v17["PARENTS"]["v13_compose"]["tune_pq"] == 0.441777
    assert v17["SUBMIT_BAR"]["min_audit_pq"] == 0.450130
    assert v17["PROMOTION_GATE"]["required_tune_delta_vs_v13"] == 0.010


def bars_for(tune_pq, audit_pq):
    """Build a one-candidate result set with exact PQ values (tp=1 => pq == iou_sum)."""
    key = "mr1_iou030/t0.45_c0.30"
    results = {key: dict(tune=counts(tune_pq, 1, 0, 0), audit=counts(audit_pq, 1, 0, 0))}
    return v17["evaluate_bars"](key, results)


def test_below_v13_is_not_authorized_and_not_promotable():
    report = bars_for(0.4400, 0.4600)
    assert report["export_authorized"] is False
    assert report["promotable"] is False
    assert "do not spend a submission" in report["recommendation"]


def test_matching_v13_authorizes_export_but_does_not_promote():
    report = bars_for(0.441777, 0.4600)
    assert report["export_authorized"] is True
    assert report["promotable"] is False
    assert report["recommendation"] == "export and measure one submission"


def test_clearing_the_promotion_gate_requires_both_margin_and_audit():
    # +0.010 tune but audit below V13's -> authorized, not promotable.
    assert bars_for(0.451777, 0.4600)["promotable"] is False
    # Both cleared.
    promoted = bars_for(0.451777, 0.464644)
    assert promoted["promotable"] is True
    assert promoted["recommendation"] == "promote and submit"


def test_audit_floor_blocks_export_even_with_a_strong_tune():
    assert bars_for(0.4600, 0.4400)["export_authorized"] is False


def test_bars_report_deltas_against_every_parent_including_v16():
    report = bars_for(0.441777, 0.464644)
    assert set(report["deltas_vs_parents"]) == set(v17["PARENTS"])
    assert report["deltas_vs_parents"]["v13_compose"]["tune"] == pytest.approx(0.0)
    assert report["deltas_vs_parents"]["v16_arm_r1_seed1"]["tune"] > 0
    assert report["inference_only"] is True
    assert "does not apply" in report["training_noise_floor_note"]


# --- control verification ---------------------------------------------------------

def control_counts(tune, audit):
    key = v17["cell_candidate_key"](v17["CONTROL_CELL"]["name"], 0.40, 0.40)
    return {key: dict(tune=tune, audit=audit)}


def test_control_verification_accepts_the_exact_e015_counts():
    reference = v17["CONTROL_REFERENCE"]
    verified = v17["verify_control"](control_counts(list(reference["tune"]),
                                                   list(reference["audit"])))
    assert [int(v) for v in verified["tune"][1:]] == [626, 211, 508]


def test_control_verification_rejects_a_single_changed_count():
    reference = v17["CONTROL_REFERENCE"]
    broken = list(reference["tune"])
    broken[1] = 625
    with pytest.raises(ValueError, match="Control tune counts differ"):
        v17["verify_control"](control_counts(broken, list(reference["audit"])))


def test_control_verification_rejects_drifted_iou_sum():
    reference = v17["CONTROL_REFERENCE"]
    broken = list(reference["tune"])
    broken[0] += 0.01
    with pytest.raises(ValueError, match="Control tune counts differ"):
        v17["verify_control"](control_counts(broken, list(reference["audit"])))


def test_control_reference_matches_v13s_accepted_reference():
    """Both runs must gate on the same accepted E015 baseline."""
    assert v17["CONTROL_REFERENCE"]["tune"][1:] == v13["CONTROL_REFERENCE"]["tune"][1:]
    assert v17["CONTROL_REFERENCE"]["audit"][1:] == v13["CONTROL_REFERENCE"]["audit"][1:]


# --- seed agreement ---------------------------------------------------------------

def test_seed_agreement_reports_spread_without_influencing_selection():
    key = "mr1_iou030/t0.45_c0.30"
    per_seed = {
        "seed1": {key: dict(tune=counts(0.4500, 1, 0, 0), audit=counts(0.4700, 1, 0, 0))},
        "seed2": {key: dict(tune=counts(0.4460, 1, 0, 0), audit=counts(0.4660, 1, 0, 0))},
    }
    report = v17["seed_agreement"](per_seed, key)
    assert report["tune_pq_spread"] == pytest.approx(0.0040)
    assert report["both_clear_submit_bar"] is True
    assert set(report["per_seed"]) == {"seed1", "seed2"}


def test_seed_agreement_flags_when_one_seed_misses_the_bar():
    key = "mr1_iou030/t0.45_c0.30"
    per_seed = {
        "seed1": {key: dict(tune=counts(0.4500, 1, 0, 0), audit=counts(0.4700, 1, 0, 0))},
        "seed2": {key: dict(tune=counts(0.4300, 1, 0, 0), audit=counts(0.4400, 1, 0, 0))},
    }
    report = v17["seed_agreement"](per_seed, key)
    assert report["both_clear_submit_bar"] is False
    assert report["tune_pq_spread"] == pytest.approx(0.0200)


# --- run hygiene ------------------------------------------------------------------

def test_module_declares_it_neither_trains_nor_submits():
    source = (ROOT / "kaggle_inference/v17_compose.py").read_text()
    assert "def run_compose" in source
    # No training entry points may leak in from V16.
    assert "model.train(" not in source
    assert ".train(data=" not in source


def test_pq_matches_the_projects_definition():
    assert v17["pq"]([2.0, 4.0, 2.0, 2.0]) == pytest.approx(2.0 / 6.0)
    assert v17["pq"]([0.0, 0, 0, 0]) == 0.0
    assert v17["pq"]([1.0, 1, 0, 0]) == pytest.approx(1.0)
