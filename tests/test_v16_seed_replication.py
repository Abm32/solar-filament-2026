"""Tests for scripts/analyze_v16_seed_replication.py.

The central risk this script carries is the one that invalidated V14: a diagnostic that
silently describes a different population than it claims. The per-file aggregation is the
place that can go wrong, because `v16_image_diagnostics.csv` has SEVERAL rows per
`file_name` (one per annotator `image_id`). An early version of this script overwrote
instead of summing and produced a confident bootstrap interval whose observed delta
disagreed with `v16_selection.json` (+0.003929 vs the true -0.001220). These tests pin
that behaviour and the guard that catches it.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_v16_seed_replication import (  # noqa: E402
    ARM_R1, ARM_R2, CONTROL, _assert_matches_selection, paired_bootstrap, pq,
    read_diagnostics,
)

ROOT = Path(__file__).resolve().parents[1]


def write_diagnostics(path, rows):
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["candidate", "file_name", "image_id", "group", "target",
                        "iou_sum", "tp", "fp", "fn"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def row(candidate, file_name, image_id, group, iou_sum, tp, fp, fn):
    return dict(candidate=candidate, file_name=file_name, image_id=image_id, group=group,
                target="coco", iou_sum=iou_sum, tp=tp, fp=fp, fn=fn)


def test_pq_is_recognition_quality_weighted():
    # iou_sum / (tp + 0.5*fp + 0.5*fn)
    assert pq([(2.0, 4.0, 2.0, 2.0)]) == pytest.approx(2.0 / 6.0)
    assert pq([]) == 0.0
    # A denominator of zero must not raise.
    assert pq([(0.0, 0.0, 0.0, 0.0)]) == 0.0


def test_pq_aggregates_across_records_before_dividing():
    """PQ of a set is NOT the mean of per-record PQs."""
    records = [(1.0, 1.0, 0.0, 0.0), (0.0, 0.0, 2.0, 2.0)]
    assert pq(records) == pytest.approx(1.0 / 3.0)
    mean_of_ratios = (1.0 + 0.0) / 2
    assert pq(records) != pytest.approx(mean_of_ratios)


def test_read_diagnostics_sums_annotator_variants_per_file(tmp_path):
    """Multiple image_id rows for one file_name must SUM, not overwrite.

    This is the exact regression that produced a wrong bootstrap interval.
    """
    path = tmp_path / "d.csv"
    write_diagnostics(path, [
        row(CONTROL, "a.jpeg", "010101-a", "tune", 1.5, 2, 1, 1),
        row(CONTROL, "a.jpeg", "010102-a", "tune", 2.5, 3, 0, 2),
    ])
    data = read_diagnostics(path)
    assert data[(CONTROL, "tune")]["a.jpeg"] == (4.0, 5.0, 1.0, 3.0)


def test_read_diagnostics_separates_candidate_and_group(tmp_path):
    path = tmp_path / "d.csv"
    write_diagnostics(path, [
        row(CONTROL, "a.jpeg", "1-a", "tune", 1.0, 1, 0, 0),
        row(CONTROL, "b.jpeg", "1-b", "audit", 2.0, 2, 0, 0),
        row(ARM_R1, "a.jpeg", "1-a", "tune", 3.0, 3, 0, 0),
    ])
    data = read_diagnostics(path)
    assert set(data) == {(CONTROL, "tune"), (CONTROL, "audit"), (ARM_R1, "tune")}
    assert data[(CONTROL, "tune")]["a.jpeg"][0] == 1.0
    assert data[(ARM_R1, "tune")]["a.jpeg"][0] == 3.0


def selection_stub(control_tune, arm_r2_tune, arm_r1_tune, arm_r1_audit):
    return {"harness_check": {"control_tune_pq": control_tune,
                              "arm_r2_tune_pq": arm_r2_tune},
            "gate": {"selected_tune_pq": arm_r1_tune,
                     "selected_audit_pq": arm_r1_audit}}


def consistent_diagnostics():
    """One file per candidate/group, so PQ is trivially computable."""
    data = {}
    for candidate in (CONTROL, ARM_R2, ARM_R1):
        data[(candidate, "tune")] = {"a.jpeg": (2.0, 4.0, 2.0, 2.0)}   # PQ = 1/3
        data[(candidate, "audit")] = {"b.jpeg": (3.0, 6.0, 0.0, 0.0)}  # PQ = 1/2
    return data


def test_assert_matches_selection_passes_when_consistent():
    third, half = 2.0 / 6.0, 0.5
    _assert_matches_selection(consistent_diagnostics(),
                              selection_stub(third, third, third, half), "ok")


def test_assert_matches_selection_rejects_disagreement():
    """A mis-aggregated CSV must abort rather than report a confident interval."""
    with pytest.raises(AssertionError, match="per-file aggregation is wrong"):
        _assert_matches_selection(consistent_diagnostics(),
                                  selection_stub(0.99, 2.0 / 6.0, 2.0 / 6.0, 0.5), "bad")


def test_paired_bootstrap_observed_delta_is_the_full_sample_delta():
    control = {f"f{i}.jpeg": (1.0, 2.0, 1.0, 1.0) for i in range(8)}
    arm = {f"f{i}.jpeg": (1.5, 2.0, 1.0, 1.0) for i in range(8)}
    result = paired_bootstrap(arm, control, sorted(control), resamples=200)
    expected = pq(list(arm.values())) - pq(list(control.values()))
    assert result["observed_delta"] == pytest.approx(expected)
    assert result["p_delta_gt_0"] == pytest.approx(1.0)


def test_paired_bootstrap_is_deterministic_for_a_fixed_seed():
    control = {f"f{i}.jpeg": (1.0, 2.0, 1.0, 1.0) for i in range(10)}
    arm = {f"f{i}.jpeg": (1.0 + 0.1 * i, 2.0, 1.0, 1.0) for i in range(10)}
    files = sorted(control)
    first = paired_bootstrap(arm, control, files, resamples=300, seed=7)
    second = paired_bootstrap(arm, control, files, resamples=300, seed=7)
    assert first == second
    other = paired_bootstrap(arm, control, files, resamples=300, seed=8)
    assert other["ci95_low"] != first["ci95_low"]


def test_paired_bootstrap_detects_no_difference_as_zero_delta():
    control = {f"f{i}.jpeg": (1.0, 2.0, 1.0, 1.0) for i in range(6)}
    result = paired_bootstrap(dict(control), control, sorted(control), resamples=200)
    assert result["observed_delta"] == pytest.approx(0.0)
    assert result["ci95_low"] == pytest.approx(0.0)
    assert result["ci95_high"] == pytest.approx(0.0)


REAL_RUNS = [ROOT / "outputs/kaggle_v16_full", ROOT / "outputs/kaggle_v16_full_seed2"]


@pytest.mark.parametrize("base", REAL_RUNS, ids=lambda p: p.name)
def test_real_run_diagnostics_reproduce_shipped_selection(base):
    """End-to-end guard on the actual pulled artifacts, if present."""
    diagnostics_path = base / "v16_image_diagnostics.csv"
    selection_path = base / "v16_selection.json"
    if not diagnostics_path.exists() or not selection_path.exists():
        pytest.skip(f"{base.name} artifacts not pulled")
    diagnostics = read_diagnostics(diagnostics_path)
    selection = json.loads(selection_path.read_text())
    _assert_matches_selection(diagnostics, selection, base.name)


@pytest.mark.parametrize("base", REAL_RUNS, ids=lambda p: p.name)
def test_real_run_control_reproduces_e015_counts(base):
    """Both seeds share the frozen control, which must equal E015's exact counts."""
    selection_path = base / "v16_selection.json"
    if not selection_path.exists():
        pytest.skip(f"{base.name} artifacts not pulled")
    counts = json.loads(selection_path.read_text())["candidate_counts"][CONTROL]
    assert counts["tune"] == [pytest.approx(426.5473399071962), 626.0, 211.0, 508.0]
    assert counts["audit"] == [pytest.approx(182.75261599994462), 265.0, 88.0, 194.0]
