"""Tests for the three 2026-09-09 zero-GPU analyses (E027/E028).

These scripts are the basis for a *stop optimising* recommendation, so the risk they
carry is the V14 failure mode in its most consequential form: a diagnostic that
confidently describes a different population than it claims, used to justify not
spending GPU. The tests below pin the arithmetic that the conclusions rest on:

  * the display-interval and feasibility logic in `analyze_public_transfer` (the
    claim "a constant offset is impossible" must follow from the interval algebra,
    not from a coding slip);
  * `cross_source_nms` and `aggregate` in `analyze_union_ensemble` (the claim
    "extra recall does not survive its false positives" depends entirely on FP
    being counted as n_pred - matched, and on NMS actually suppressing duplicates);
  * `oracle_pq` in `analyze_detector_complementarity` (the union ceiling of 0.5849).
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
from pycocotools import mask as mask_utils

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_public_transfer import (  # noqa: E402
    SUBMISSIONS, affine_feasibility, constant_offset_feasibility, display_interval,
)
from analyze_detector_complementarity import oracle_pq  # noqa: E402
from analyze_union_ensemble import aggregate, cross_source_nms, score_pool  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------------------
# analyze_public_transfer
# --------------------------------------------------------------------------------------

def test_display_interval_round_is_symmetric_and_half_a_cent_wide():
    lo, hi = display_interval(0.39, "round")
    assert lo == pytest.approx(0.385)
    assert hi == pytest.approx(0.395)


def test_display_interval_truncate_is_one_sided():
    lo, hi = display_interval(0.39, "truncate")
    assert lo == pytest.approx(0.39)
    assert hi == pytest.approx(0.40)


def test_display_interval_rejects_unknown_mode():
    with pytest.raises(ValueError):
        display_interval(0.39, "floor_to_even")


def test_the_five_submissions_are_the_measured_ones():
    """Guards against a silent edit to the calibration set the conclusion rests on."""
    assert len(SUBMISSIONS) == 5
    locals_ = [round(local, 6) for _n, local, _p, _r in SUBMISSIONS]
    publics = [public for _n, _l, public, _r in SUBMISSIONS]
    refs = [ref for _n, _l, _p, ref in SUBMISSIONS]
    assert locals_ == [0.432823, 0.437313, 0.438001, 0.441777, 0.448703]
    assert publics == [0.38, 0.39, 0.39, 0.39, 0.39]
    assert refs == ["56026663", "56050554", "56049359", "56076685", "56103401"]
    # local PQ must be strictly increasing: that is what makes the flat public
    # score informative rather than merely noisy.
    assert locals_ == sorted(locals_)


@pytest.mark.parametrize("mode", ["round", "truncate"])
def test_constant_offset_is_infeasible_across_all_five(mode):
    """The headline falsification of ceiling_analysis section 1."""
    result = constant_offset_feasibility(mode)
    assert result["feasible"] is False
    assert result["intersection_lo"] > result["intersection_hi"]


@pytest.mark.parametrize("mode", ["round", "truncate"])
def test_constant_offset_would_be_feasible_without_v17(mode, monkeypatch):
    """Control: the model only breaks once V17 is added.

    If this failed, the 'V17 falsifies it' claim would be wrong -- the model would
    already have been broken by the first four points.
    """
    import analyze_public_transfer as module
    monkeypatch.setattr(module, "SUBMISSIONS", SUBMISSIONS[:4])
    result = module.constant_offset_feasibility(mode)
    assert result["feasible"] is True


@pytest.mark.parametrize("mode", ["round", "truncate"])
def test_unit_slope_is_ruled_out_but_shallower_slopes_survive(mode):
    grid = [round(x * 0.01, 2) for x in range(0, 201)]
    feasible = affine_feasibility(mode, grid)
    slopes = {entry["a"] for entry in feasible}
    assert 1.00 not in slopes, "a=1.0 (constant offset) must be ruled out"
    assert slopes, "some affine model must survive"
    assert max(slopes) == pytest.approx(0.87)


@pytest.mark.parametrize("mode", ["round", "truncate"])
def test_local_060_does_not_reach_public_055_at_any_feasible_slope(mode):
    """The strategic consequence: ceiling_analysis's target was understated."""
    grid = [round(x * 0.01, 2) for x in range(0, 201)]
    feasible = affine_feasibility(mode, grid)
    assert feasible
    best = max(entry["public_at_local_0_60"] for entry in feasible)
    assert best < 0.55, f"expected local 0.60 to fall short of public 0.55, got {best}"


# --------------------------------------------------------------------------------------
# analyze_detector_complementarity
# --------------------------------------------------------------------------------------

def test_oracle_pq_matches_the_pq_definition():
    # 10 GT, 8 matched at mean IoU 0.5, 4 FP -> iou_sum 4.0 over (8 + 2 + 1) = 11
    assert oracle_pq(8, 0.5, 10, 4) == pytest.approx(4.0 / 11.0)


def test_oracle_pq_is_perfect_only_with_full_recall_and_no_fp():
    assert oracle_pq(10, 1.0, 10, 0) == pytest.approx(1.0)
    assert oracle_pq(10, 1.0, 10, 4) == pytest.approx(10 / 12)
    assert oracle_pq(0, 0.0, 10, 0) == pytest.approx(0.0)


def test_union_ceiling_is_below_the_060_target():
    """The measured union oracle: recall 0.8131 at meanIoU 0.6522 over 1134 GT.

    Even with perfect selection and ZERO false positives, a 6-way ensemble cannot
    reach the local 0.60 that the (now-falsified) transfer model asked for.
    """
    matched, n_gt, mean_iou = 922, 1134, 0.6522
    assert oracle_pq(matched, mean_iou, n_gt, 0) < 0.60
    assert oracle_pq(matched, mean_iou, n_gt, 0) == pytest.approx(0.5849, abs=5e-4)
    # and at V17's actual false-positive count it is far lower
    assert oracle_pq(matched, mean_iou, n_gt, 306) == pytest.approx(0.5092, abs=5e-4)


# --------------------------------------------------------------------------------------
# analyze_union_ensemble
# --------------------------------------------------------------------------------------

def box_rle(x0, y0, x1, y1, size=64):
    mask = np.zeros((size, size), dtype=np.uint8, order="F")
    mask[y0:y1, x0:x1] = 1
    return mask_utils.encode(mask)


def test_cross_source_nms_suppresses_a_duplicate_and_keeps_the_higher_score():
    a = box_rle(0, 0, 20, 20)
    duplicate = box_rle(0, 0, 20, 20)
    kept, scores = cross_source_nms([a, duplicate], [0.6, 0.9], nms_iou=0.5)
    assert len(kept) == 1
    assert scores == [0.9], "NMS must keep the higher-scoring duplicate"


def test_cross_source_nms_keeps_disjoint_proposals():
    a = box_rle(0, 0, 20, 20)
    b = box_rle(30, 30, 50, 50)
    kept, _ = cross_source_nms([a, b], [0.5, 0.4], nms_iou=0.5)
    assert len(kept) == 2


def test_cross_source_nms_threshold_is_respected():
    """Two boxes overlapping at IoU ~0.33: suppressed at 0.3, kept at 0.5."""
    a = box_rle(0, 0, 20, 10)
    b = box_rle(10, 0, 30, 10)
    iou = float(np.asarray(mask_utils.iou([a], [b], [0]), dtype=np.float64).ravel()[0])
    assert 0.3 < iou < 0.5
    assert len(cross_source_nms([a, b], [0.9, 0.8], nms_iou=0.3)[0]) == 1
    assert len(cross_source_nms([a, b], [0.9, 0.8], nms_iou=0.5)[0]) == 2


def test_cross_source_nms_handles_an_empty_pool():
    assert cross_source_nms([], [], nms_iou=0.5) == ([], [])


def test_score_pool_applies_the_confidence_gate():
    gt = [box_rle(0, 0, 20, 20)]
    proposals = [box_rle(0, 0, 20, 20), box_rle(30, 30, 50, 50)]
    high = score_pool(proposals, [0.9, 0.9], gt, confidence=0.5)
    assert high["n_pred"] == 2 and high["matched"] == 1
    gated = score_pool(proposals, [0.9, 0.1], gt, confidence=0.5)
    assert gated["n_pred"] == 1 and gated["matched"] == 1


def test_aggregate_counts_false_positives_as_unmatched_predictions():
    """This is the arithmetic the 'clean negative' verdict depends on."""
    rows = [{"n_gt": 10, "n_pred": 12, "matched": 8, "iou_sum": 6.0}]
    out = aggregate(rows, "x")
    assert out["fp"] == 4          # 12 predicted - 8 matched
    assert out["fn"] == 2          # 10 GT - 8 matched
    assert out["recall"] == pytest.approx(0.8)
    assert out["mean_matched_iou"] == pytest.approx(0.75)
    assert out["keep_all_pq"] == pytest.approx(6.0 / (8 + 0.5 * 4 + 0.5 * 2))


def test_aggregate_sums_over_images_rather_than_averaging_rates():
    """Per-image rate averaging is the classic silent-population bug; pin the sum."""
    rows = [{"n_gt": 5, "n_pred": 5, "matched": 5, "iou_sum": 5.0},
            {"n_gt": 95, "n_pred": 0, "matched": 0, "iou_sum": 0.0}]
    out = aggregate(rows, "x")
    assert out["n_gt"] == 100
    assert out["recall"] == pytest.approx(0.05)   # not 0.5


def test_aggregate_is_safe_on_empty_input():
    out = aggregate([], "x")
    assert out["recall"] == 0.0 and out["keep_all_pq"] == 0.0


def test_more_proposals_can_lower_pq_even_while_recall_rises():
    """The mechanism behind the negative ensemble result, stated as a test."""
    single = aggregate([{"n_gt": 100, "n_pred": 60, "matched": 55, "iou_sum": 38.5}], "single")
    union = aggregate([{"n_gt": 100, "n_pred": 110, "matched": 60, "iou_sum": 42.0}], "union")
    assert union["recall"] > single["recall"]
    assert union["keep_all_pq"] < single["keep_all_pq"]


# --------------------------------------------------------------------------------------
# committed artifacts
# --------------------------------------------------------------------------------------

def test_committed_reports_agree_with_the_recorded_conclusions():
    """If a report is regenerated with different inputs, this fails loudly."""
    import json
    transfer = ROOT / "outputs/public_transfer_2026-09-09.json"
    complementarity = ROOT / "outputs/v16_proposal_analysis/detector_complementarity.json"
    union = ROOT / "outputs/v16_proposal_analysis/union_ensemble.json"
    for path in (transfer, complementarity, union):
        if not path.exists():
            pytest.skip(f"{path.name} not generated in this checkout")

    t = json.loads(transfer.read_text())
    for mode in ("round", "truncate"):
        assert t["constant_offset"][mode]["feasible"] is False
        assert t["affine"][mode]["slope_max"] == pytest.approx(0.87)

    c = json.loads(complementarity.read_text())
    tune = c["groups"]["tune"]
    assert tune["n_gt"] == 1134
    assert tune["union"]["all_sources"]["missed_by_every_source"] == 212
    assert tune["union"]["all_sources"]["recall"] == pytest.approx(0.8131, abs=1e-3)
    assert tune["union"]["oracle_pq"]["at_zero_fp"] < 0.60

    u = json.loads(union.read_text())
    sources = u["sources"]
    wins = 0
    for confidence in u["confidences"]:
        key = f"c{confidence:.2f}"
        best_single = max(u["groups"]["tune"][s][key]["keep_all_pq"] for s in sources)
        if u["groups"]["tune"]["UNION_all"][key]["keep_all_pq"] > best_single:
            wins += 1
    assert wins <= 1, "the union ensemble must remain a clean negative on tune"
