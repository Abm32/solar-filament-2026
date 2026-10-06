import ast
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils

from scripts import train_v22_pointrend as v22


ROOT = Path(__file__).resolve().parents[1]


def rectangle(shape, y0, y1, x0, x1):
    mask = np.zeros(shape, dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def test_frozen_schedule_grid_and_bars_match_design():
    assert v22.STEPS_PER_EPOCH == 235
    assert v22.MAX_ITER == 4700
    assert v22.CHECKPOINT_EPOCHS == (5, 10, 15, 20)
    assert v22.CHECKPOINT_PERIOD == 1175
    assert v22.CONFIDENCES == tuple(round(0.05 * value, 2) for value in range(1, 11))
    assert v22.SUBMIT_BAR_TUNE == 0.448703
    assert v22.SUBMIT_BAR_AUDIT == 0.470763
    assert v22.PROMOTION_GATE_TUNE == 0.55


def test_real_canonical_populations_and_split_isolation():
    paths = {"train": v22.TRAIN_JSON, "tune": v22.TUNE_JSON, "audit": v22.AUDIT_JSON}
    assert {group: v22.population(path) for group, path in paths.items()} == v22.EXPECTED
    names = {group: v22.split_filenames(path) for group, path in paths.items()}
    assert not names["train"] & names["tune"]
    assert not names["train"] & names["audit"]
    assert not names["tune"] & names["audit"]


def test_pq_is_strictly_greater_than_half_and_reports_sq_rq():
    target = rectangle((10, 10), 0, 2, 0, 2)
    exact = target.copy()
    half = rectangle((10, 10), 0, 2, 0, 1)
    counts = v22.pq_counts([exact, half], [target])
    result = v22.metrics(counts)
    assert result == {
        "pq": 2.0 / 3.0,
        "sq": 1.0,
        "rq": 2.0 / 3.0,
        "tp_iou": 1.0,
        "tp": 1,
        "fp": 1,
        "fn": 0,
    }


def test_ownership_is_score_ordered_disjoint_and_min_area_guarded():
    shape = (16, 16)
    strong = rectangle(shape, 1, 9, 1, 9)
    overlap = rectangle(shape, 5, 13, 5, 13)
    tiny = rectangle(shape, 14, 16, 14, 16)
    accepted = v22.ownership_masks(
        np.stack([overlap, tiny, strong]), np.array([0.8, 0.95, 0.9]), threshold=0.5
    )
    assert len(accepted) == 2
    assert np.array_equal(accepted[0], strong)
    assert not np.any(accepted[0] & accepted[1])
    assert int(accepted[1].sum()) == int((overlap & ~strong).sum())


def test_tune_selection_ignores_audit_and_breaks_ties_as_frozen():
    runs = [
        (5, {"results": {"0.1": {"pq": 0.4, "audit": 0.99}, "0.2": {"pq": 0.5, "audit": 0.0}}}),
        (10, {"results": {"0.2": {"pq": 0.5, "audit": 1.0}, "0.3": {"pq": 0.49, "audit": 1.0}}}),
    ]
    epoch, confidence, run = v22.select_tune_candidate(runs)
    assert epoch == 5
    assert confidence == 0.2
    assert run is runs[0][1]


def test_threshold_parser_rejects_grid_expansion():
    assert v22.parse_thresholds("0.05,0.5") == (0.05, 0.5)
    try:
        v22.parse_thresholds("0.55")
    except ValueError as error:
        assert "outside frozen grid" in str(error)
    else:
        raise AssertionError("expanded threshold grid was accepted")


def test_v22_never_exports_or_submits():
    source = (ROOT / "scripts/train_v22_pointrend.py").read_text()
    tree = ast.parse(source)
    called_names = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name))
    }
    assert "to_csv" not in called_names
    assert "submit" not in called_names
    assert "competitions_submit" not in called_names
    assert "checkpointer.save" not in source
    assert "shutil.copy2" in source
    assert "subprocess" in source


def test_official_checkpoint_hash_is_pinned_not_a_placeholder():
    assert len(v22.WEIGHTS_SHA256) == 64
    int(v22.WEIGHTS_SHA256, 16)
    assert v22.WEIGHTS_SHA256 == "2a658c57f565b1badc0e324790b86d50d60f2c30139a85e33b93d0e4746897af"
