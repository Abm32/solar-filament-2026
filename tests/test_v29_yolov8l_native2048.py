import ast
from pathlib import Path

from scripts import probe_v29_yolov8l_native2048 as v29

ROOT = Path(__file__).resolve().parents[1]


def test_public_upstream_regime_is_frozen():
    assert v29.MODEL_NAME == "yolov8l-seg.pt"
    assert v29.IMAGE_SIZE == 2048
    assert v29.MASK_RATIO == 4
    assert v29.BATCH_SIZE == 1
    assert v29.EPOCHS == 1


def test_probe_uses_complete_canonical_training_population():
    assert v29.EXPECTED_TRAIN_RECORDS == 937
    assert v29.EXPECTED_TRAIN_INSTANCES == 6606
    assert v29.EXPECTED_VAL_RECORDS == 141


def test_probe_is_not_v18_mask_ratio_one_configuration():
    source = (ROOT / "scripts/probe_v29_yolov8l_native2048.py").read_text()
    assert "yolov8l-seg.pt" in source
    assert "mask_ratio=MASK_RATIO" in source
    assert v29.MASK_RATIO != 1


def test_probe_cannot_read_audit_test_or_submit():
    source = (ROOT / "scripts/probe_v29_yolov8l_native2048.py").read_text()
    tree = ast.parse(source)
    calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name))
    }
    assert "test/test_images" not in source
    assert "submission.csv" not in source
    assert "submit" not in calls and "competitions_submit" not in calls


def test_memory_probe_runs_a_complete_epoch_not_a_fraction():
    source = (ROOT / "scripts/probe_v29_yolov8l_native2048.py").read_text()
    assert "fraction=" not in source
    assert "epochs=EPOCHS" in source
    assert "val=False" in source
