"""E-A cell logic, control reproduction, and gate evaluation without GPU imports."""
import ast
import json
from pathlib import Path

import pandas as pd
import pytest

SOURCE = Path(__file__).resolve().parents[1] / "kaggle_inference/detector_flags.py"
tree = ast.parse(SOURCE.read_text())

# CELLS / CONTROL_CELL / GATE are module-level constant assignments; the three
# functions under test (cell_by_name, verify_control_reproduces_accepted,
# evaluate_gate) have no torch/ultralytics/cv2 dependency. Extract both without
# importing the GPU-only helpers (proposal_views_for_cell, evaluate_cell,
# run_experiment), matching the extraction pattern in test_detector_tta.py.
constant_names = {"CELLS", "CONTROL_CELL", "GATE", "CONTROL_REFERENCE"}
function_names = {"cell_by_name", "verify_control_reproduces_accepted", "evaluate_gate"}
body = [node for node in tree.body
        if (isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in constant_names for t in node.targets))
        or (isinstance(node, ast.FunctionDef) and node.name in function_names)]
module = ast.Module(body=body, type_ignores=[])
ns = {"pd": pd}
exec(compile(module, str(SOURCE), "exec"), ns)


def test_cells_declare_four_unique_flag_combinations_including_the_control():
    cells = ns["CELLS"]
    assert len(cells) == 4
    names = [cell["name"] for cell in cells]
    assert len(set(names)) == 4, "cell names must be unique"
    combos = {(cell["retina_masks"], cell["agnostic_nms"], cell["iou"]) for cell in cells}
    assert len(combos) == 4, "flag tuples must be unique"
    assert all(cell["retina_masks"] is True for cell in cells), (
        "retina_masks=False cells were dropped: Ultralytics returns those masks "
        "at internal mask-head resolution, not original image coordinates "
        "(github.com/ultralytics/ultralytics/issues/8691)")
    assert ns["CONTROL_CELL"] in names
    control = ns["cell_by_name"](ns["CONTROL_CELL"])
    assert control == dict(name=ns["CONTROL_CELL"], retina_masks=True, agnostic_nms=True, iou=0.7)


def test_cell_by_name_rejects_unknown_names():
    with pytest.raises(ValueError):
        ns["cell_by_name"]("not_a_real_cell")


def test_gate_requires_both_tune_delta_and_nonnegative_audit():
    gate = ns["GATE"]
    assert gate["required_tune_delta"] == pytest.approx(0.003)
    assert gate["require_nonnegative_audit"] is True


def _row(cell, candidate, tune_pq, audit_pq):
    return dict(cell=cell, candidate=candidate, tune_pq=tune_pq, audit_pq=audit_pq)


def test_evaluate_gate_passes_a_cell_that_clears_both_thresholds():
    control = ns["CONTROL_CELL"]
    other = [c["name"] for c in ns["CELLS"] if c["name"] != control][0]
    combined = pd.DataFrame([
        _row(control, "r1024_refined_constrained_t0.50_tta1_c0.40", 0.4328, 0.4501),
        _row(other, "r1024_refined_constrained_t0.45_tta1_c0.40", 0.4328 + 0.0035, 0.4501 + 0.001),
    ])
    report = ns["evaluate_gate"](combined)
    assert report["any_cell_passed_gate"] is True
    passed = {row["cell"]: row for row in report["cells"]}
    assert passed[other]["gate_passed"] is True
    assert passed[other]["tune_delta"] == pytest.approx(0.0035)
    assert passed[control]["gate_passed"] is False, "the control cell can never pass its own gate"
    assert report["recommendation"] == f"promote {other}"


def test_evaluate_gate_rejects_a_cell_below_the_tune_threshold():
    control = ns["CONTROL_CELL"]
    other = [c["name"] for c in ns["CELLS"] if c["name"] != control][0]
    combined = pd.DataFrame([
        _row(control, "ctrl", 0.4328, 0.4501),
        _row(other, "cand", 0.4328 + 0.001, 0.4501 + 0.001),  # below +0.003 gate
    ])
    report = ns["evaluate_gate"](combined)
    assert report["any_cell_passed_gate"] is False
    assert report["recommendation"] == "no cell clears the gate; E-A measured and dropped"


def test_evaluate_gate_rejects_a_cell_with_negative_audit_delta_despite_tune_gain():
    control = ns["CONTROL_CELL"]
    other = [c["name"] for c in ns["CELLS"] if c["name"] != control][0]
    combined = pd.DataFrame([
        _row(control, "ctrl", 0.4328, 0.4501),
        _row(other, "cand", 0.4328 + 0.01, 0.4501 - 0.001),  # tune clears, audit regresses
    ])
    report = ns["evaluate_gate"](combined)
    assert report["any_cell_passed_gate"] is False
    row = {r["cell"]: r for r in report["cells"]}[other]
    assert row["tune_delta"] >= ns["GATE"]["required_tune_delta"]
    assert row["gate_passed"] is False


def test_evaluate_gate_picks_the_best_tune_pq_row_per_cell_before_comparing():
    control = ns["CONTROL_CELL"]
    other = [c["name"] for c in ns["CELLS"] if c["name"] != control][0]
    combined = pd.DataFrame([
        _row(control, "ctrl", 0.4328, 0.4501),
        _row(other, "cand_weak", 0.4328 + 0.001, 0.4501 + 0.001),
        _row(other, "cand_strong", 0.4328 + 0.004, 0.4501 + 0.002),
    ])
    report = ns["evaluate_gate"](combined)
    row = {r["cell"]: r for r in report["cells"]}[other]
    assert row["candidate"] == "cand_strong"
    assert row["gate_passed"] is True


def test_evaluate_gate_orders_cells_by_descending_tune_delta():
    control = ns["CONTROL_CELL"]
    others = [c["name"] for c in ns["CELLS"] if c["name"] != control][:2]
    combined = pd.DataFrame([
        _row(control, "ctrl", 0.4328, 0.4501),
        _row(others[0], "low", 0.4328 + 0.001, 0.4501),
        _row(others[1], "high", 0.4328 + 0.005, 0.4501 + 0.002),
    ])
    report = ns["evaluate_gate"](combined)
    ordered_cells = [row["cell"] for row in report["cells"]]
    assert ordered_cells.index(others[1]) < ordered_cells.index(others[0])
    assert report["recommendation"] == f"promote {others[1]}"


def _control_rows(reference, tune_split=(0.6, 0.4), audit_split=(0.5, 0.5)):
    """Two per-image rows per group that sum exactly to the reference totals."""
    rows = []
    for group, splits in (("tune", tune_split), ("audit", audit_split)):
        iou, tp, fp, fn = reference[group]
        rows.append(dict(candidate=reference["candidate"], file_name=f"{group}_a.jpeg",
                         image_id=f"{group}1", group=group, target="coco",
                         iou_sum=iou * splits[0], tp=tp, fp=fp, fn=fn))
        rows.append(dict(candidate=reference["candidate"], file_name=f"{group}_b.jpeg",
                         image_id=f"{group}2", group=group, target="coco",
                         iou_sum=iou * splits[1], tp=0, fp=0, fn=0))
    # Legacy rows must be ignored: they re-score the canonical-best annotator against
    # a rounded target and are excluded from tune/audit PQ by construction.
    rows.append(dict(candidate=reference["candidate"], file_name="tune_a.jpeg",
                     image_id="tune1", group="legacy", target="rounded_legacy",
                     iou_sum=999.0, tp=999, fp=999, fn=999))
    # A different threshold/confidence candidate must be ignored too.
    rows.append(dict(candidate="r1024_refined_constrained_t0.55_tta1_c0.40",
                     file_name="tune_a.jpeg", image_id="tune1", group="tune", target="coco",
                     iou_sum=42.0, tp=42, fp=42, fn=42))
    return pd.DataFrame(rows)


def test_control_reference_pins_the_flip_consensus_e015_counts():
    reference = ns["CONTROL_REFERENCE"]
    # The accepted 0.38 run is flip_consensus at t0.40/c0.40, not the single-view
    # t0.50 sweep that sits beside the refiner checkpoint. Regression guard for the
    # 2026-09-07 E-A failure, where the control was diffed against the wrong arm.
    assert reference["candidate"] == "r1024_refined_constrained_t0.40_tta1_c0.40"
    assert reference["tune"] == [pytest.approx(426.5473399071961), 626, 211, 508]
    assert reference["audit"] == [pytest.approx(182.75261599994468), 265, 88, 194]


def test_verify_control_reproduces_accepted_passes_on_matching_frozen_counts(tmp_path):
    reference = ns["CONTROL_REFERENCE"]
    cell_output = tmp_path / "cell"
    cell_output.mkdir()
    _control_rows(reference).to_csv(cell_output / "inference_image_diagnostics.csv", index=False)
    observed = ns["verify_control_reproduces_accepted"](cell_output)
    assert observed["tune"][1:] == reference["tune"][1:]
    assert observed["audit"][1:] == reference["audit"][1:]


def test_verify_control_reproduces_accepted_fails_on_a_changed_control_count(tmp_path):
    reference = ns["CONTROL_REFERENCE"]
    cell_output = tmp_path / "cell"
    cell_output.mkdir()
    rows = _control_rows(reference)
    control = (rows.candidate == reference["candidate"]) & (rows.group == "tune")
    rows.loc[rows[control].index[0], "tp"] += 1
    rows.to_csv(cell_output / "inference_image_diagnostics.csv", index=False)
    with pytest.raises(ValueError, match="Control tune counts differ"):
        ns["verify_control_reproduces_accepted"](cell_output)


def test_verify_control_reproduces_accepted_fails_on_a_drifted_iou_sum(tmp_path):
    reference = ns["CONTROL_REFERENCE"]
    cell_output = tmp_path / "cell"
    cell_output.mkdir()
    rows = _control_rows(reference)
    control = (rows.candidate == reference["candidate"]) & (rows.group == "audit")
    rows.loc[rows[control].index[0], "iou_sum"] += 1e-3
    rows.to_csv(cell_output / "inference_image_diagnostics.csv", index=False)
    with pytest.raises(ValueError, match="Control audit counts differ"):
        ns["verify_control_reproduces_accepted"](cell_output)


def test_verify_control_reproduces_accepted_rejects_a_missing_control_candidate(tmp_path):
    cell_output = tmp_path / "cell"
    cell_output.mkdir()
    pd.DataFrame([
        dict(candidate="some_other_candidate", file_name="a.jpeg", image_id=1, group="tune",
             target="coco", iou_sum=1.0, tp=1, fp=0, fn=0),
    ]).to_csv(cell_output / "inference_image_diagnostics.csv", index=False)
    with pytest.raises(ValueError, match="missing from the sweep"):
        ns["verify_control_reproduces_accepted"](cell_output)


def test_verify_control_reproduces_accepted_requires_both_groups(tmp_path):
    """Tune-only rows must not satisfy the check; the audit half must exist too."""
    reference = ns["CONTROL_REFERENCE"]
    cell_output = tmp_path / "cell"
    cell_output.mkdir()
    pd.DataFrame([
        dict(candidate=reference["candidate"], file_name="a.jpeg", image_id=1, group="tune",
             target="coco", iou_sum=reference["tune"][0], tp=reference["tune"][1],
             fp=reference["tune"][2], fn=reference["tune"][3]),
    ]).to_csv(cell_output / "inference_image_diagnostics.csv", index=False)
    with pytest.raises(ValueError, match="no audit rows"):
        ns["verify_control_reproduces_accepted"](cell_output)


def test_verify_control_ignores_legacy_and_other_candidate_rows(tmp_path):
    """The 999-valued legacy row and the t0.55 row would break an unfiltered sum."""
    reference = ns["CONTROL_REFERENCE"]
    cell_output = tmp_path / "cell"
    cell_output.mkdir()
    rows = _control_rows(reference)
    assert (rows.target == "rounded_legacy").any() and (rows.candidate != reference["candidate"]).any()
    rows.to_csv(cell_output / "inference_image_diagnostics.csv", index=False)
    observed = ns["verify_control_reproduces_accepted"](cell_output)
    assert observed["tune"][1] == reference["tune"][1] != 999

