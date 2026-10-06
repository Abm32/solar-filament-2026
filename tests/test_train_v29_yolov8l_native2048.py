import ast
from pathlib import Path

from scripts import train_v29_yolov8l_native2048 as v29

ROOT = Path(__file__).resolve().parents[1]


def test_full_training_contract_matches_feasible_public_regime():
    assert v29.EPOCHS == 20
    assert v29.IMAGE_SIZE == 2048
    assert v29.MASK_RATIO == 4
    assert v29.GLOBAL_BATCH == 4
    assert v29.MODEL_SHA256 == "ca3ec94c445aeaf79c2b57982fb451060f18a7561e2c20b4d26bb9dc1fcabf93"


def test_full_training_uses_all_four_t4s_and_writes_checkpoints():
    source = (ROOT / "scripts/train_v29_yolov8l_native2048.py").read_text()
    assert "device=[0, 1, 2, 3]" in source
    assert "save_period=5" in source
    assert "val=False" in source


def test_full_training_cannot_read_test_or_submit():
    source = (ROOT / "scripts/train_v29_yolov8l_native2048.py").read_text()
    tree = ast.parse(source)
    calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name))
    }
    assert "test/test_images" not in source
    assert "submission.csv" not in source
    assert "submit" not in calls and "competitions_submit" not in calls
