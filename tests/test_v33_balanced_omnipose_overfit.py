import ast
from pathlib import Path

from scripts import train_v33_balanced_omnipose_overfit as v33

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "scripts/train_v33_balanced_omnipose_overfit.py").read_text()


def test_v33_is_a_separate_bounded_gate():
    assert v33.MAX_STEPS == 2000
    assert v33.EVAL_STEPS == (0, 250, 500, 1000, 1500, 2000)
    assert v33.TARGET_PQ == 0.85
    assert v33.MEMORY_GATE_MIB == 14336
    assert v33.FPN_CHANNELS == 128
    assert v33.FOREGROUND_THRESHOLD == 0.0
    assert v33.HEAD_BIASES == (-2.0, 0.0, 0.0, 0.0, -2.0)


def test_v33_uses_stride4_fpn_and_explicit_foreground():
    assert "features = (c2, c3, c4, c5)" in SOURCE
    assert "torch.nn.GroupNorm(16, FPN_CHANNELS)" in SOURCE
    assert "torch.nn.Conv2d(FPN_CHANNELS, 5, 1)" in SOURCE
    assert "foreground_logit >= FOREGROUND_THRESHOLD" in SOURCE
    assert "iscell=foreground" in SOURCE


def test_v33_sparse_losses_are_not_full_tile_regression():
    assert "flow_error.permute(0, 2, 3, 1)[foreground].mean()" in SOURCE
    assert "** 2)[foreground].mean()" in SOURCE
    assert "2.0 * foreground_loss + 5.0 * flow_loss + distance_loss + boundary_loss" in SOURCE
    assert "terms.append(values[positive].mean())" in SOURCE
    assert "terms.append(values[negative].mean())" in SOURCE


def test_v33_has_no_tune_audit_test_export_or_submission_path():
    tree = ast.parse(SOURCE)
    calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name))
    }
    assert "test_images" not in SOURCE and "submission.csv" not in SOURCE
    assert "submit" not in calls and "to_csv" not in calls
    assert '"reads_tune": False' in SOURCE
    assert '"reads_audit": False' in SOURCE
    assert '"reads_test": False' in SOURCE


def test_v33_design_was_frozen_and_records_measured_imbalance():
    design = (ROOT / "docs/v33_design_2026-09-13.md").read_text()
    assert "0.001435916" in design
    assert "695.42:1" in design
    assert "V32 remains closed" in design
    assert "before implementation staging or any V33 model step" in design


def test_v33_upstream_decoder_nan_is_fail_closed():
    assert 'if "contains NaN" not in str(error)' in SOURCE
    assert 'return [], "upstream_decoder_nan"' in SOURCE
    assert 'diagnostics["decoder_errors"] == 0' in SOURCE
