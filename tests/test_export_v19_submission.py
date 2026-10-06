"""Guard candidate identity and both canonical V19 export bars."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


export = load("export_v19_under_test", ROOT / "scripts/export_v19_submission.py")


def _authorized_result(**overrides):
    result = {
        "checkpoint_sha256": "a" * 64,
        "eval_resolution": 1024,
        "selected_threshold": 0.75,
        "best_tune_pq": 0.46,
        "audit_pq": 0.48,
        "submit_bar_tune": 0.448703,
        "submit_bar_audit": 0.470763,
        "cleared_submit_bar": True,
        "cleared_promotion_gate": False,
        "export_authorized": True,
        "populations": export.EXPECTED_POPULATIONS,
    }
    result.update(overrides)
    return result


def _write_result(tmp_path, **overrides):
    path = tmp_path / "v19_threshold_calibration.json"
    path.write_text(json.dumps(_authorized_result(**overrides)))
    return path


def test_refuses_export_when_submit_bar_not_cleared(tmp_path):
    path = _write_result(
        tmp_path, best_tune_pq=0.30, audit_pq=0.31,
        cleared_submit_bar=False, export_authorized=False,
    )
    with pytest.raises(RuntimeError, match="candidate-identity checks gate export"):
        export.load_and_check_result(path)


def test_allows_export_when_canonical_artifact_clears_both_bars(tmp_path):
    result = export.load_and_check_result(_write_result(tmp_path))
    assert result["selected_threshold"] == 0.75
    assert result["eval_resolution"] == 1024


def test_rejects_stale_true_flag_when_audit_bar_fails(tmp_path):
    path = _write_result(tmp_path, audit_pq=0.45)
    with pytest.raises(RuntimeError, match="audit=0.45"):
        export.load_and_check_result(path)


@pytest.mark.parametrize("overrides", [
    {"export_authorized": False},
    {"selected_threshold": None},
    {"eval_resolution": 0},
    {"populations": {"tune": {}, "audit": {}}},
])
def test_rejects_incomplete_or_noncanonical_calibration(tmp_path, overrides):
    path = _write_result(tmp_path, **overrides)
    with pytest.raises((RuntimeError, TypeError)):
        export.load_and_check_result(path)


def test_checkpoint_hash_must_match_calibrated_candidate(tmp_path):
    checkpoint = tmp_path / "best_model"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"actual checkpoint")
    path = _write_result(tmp_path)
    with pytest.raises(RuntimeError, match="Checkpoint hash mismatch"):
        export.load_and_check_result(path, checkpoint)


def test_matching_checkpoint_hash_is_accepted(tmp_path):
    checkpoint = tmp_path / "best_model"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"actual checkpoint")
    actual_sha = export.checkpoint_sha256(checkpoint)
    path = _write_result(tmp_path, checkpoint_sha256=actual_sha)
    result = export.load_and_check_result(path, checkpoint)
    assert result["checkpoint_sha256"] == actual_sha


def test_boundary_constants_and_min_area_match_protocol():
    assert export.SUBMIT_BAR_TUNE == 0.448703
    assert export.SUBMIT_BAR_AUDIT == 0.470763
    assert export.MIN_MASK_AREA == 20


def test_export_has_no_force_bypass_and_uses_calibrated_candidate():
    source = (ROOT / "scripts/export_v19_submission.py").read_text()
    assert '"--force"' not in source
    assert 'result["selected_threshold"]' in source
    assert 'result["eval_resolution"]' in source
