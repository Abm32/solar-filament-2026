import ast
import json
from pathlib import Path

import pytest

from scripts import train_v37_rfdetr_gate as v37

ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS = ROOT / (
    "data/raw/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train/"
    "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
)
IMAGES = ANNOTATIONS.parent / "train_images"
SOURCE = (ROOT / "scripts/train_v37_rfdetr_gate.py").read_text()


def test_v37_contract_matches_frozen_design():
    assert v37.RFDETR_VERSION == "1.10.1"
    assert v37.RESOLUTION == 768
    assert v37.MASK_DOWNSAMPLE_RATIO == 2
    assert v37.MAX_EPOCHS == 1
    assert v37.BATCH_SIZE == 1
    assert v37.MEMORY_GATE_MIB == 14336
    assert v37.TIME_GATE_SECONDS == 1200
    assert v37.SELECTION_ID == "e561c523b60993e93f82ce03116737ce512f54f4103b4557c29e409c88c92653"
    assert v37.CHECKPOINT_SHA256 == "a44613a4ecd6b5ba61a62002c600b0b6cb7a9da2936a45317ec4b62c635fb99b"
    assert v37.WHEEL_SHA256 == "96fa7a8ecbcaeea5f5150ff0b503699d2d4d75a32a87eeabfc0ab90485076a52"


def test_selection_is_exact_fit_only_population():
    document = json.loads(ANNOTATIONS.read_text())
    records = v37.selection_records(document)
    assert len(records) == 16
    assert len({row["file_name"] for row in records}) == 9
    assert sum(row["annotations"] for row in records) == 57


def test_converter_builds_three_identical_fit_only_splits(tmp_path):
    output = tmp_path / "dataset"
    manifest = v37.build_dataset(ANNOTATIONS, IMAGES, output)
    assert manifest["selection_id"] == v37.SELECTION_ID
    assert len(manifest["records"]) == 16
    assert len(manifest["physical_source_names"]) == 9
    hashes = set()
    for split in ("train", "valid", "test"):
        data = json.loads((output / split / "_annotations.coco.json").read_text())
        assert len(data["images"]) == 16 and len(data["annotations"]) == 57
        assert len(list((output / split).glob("*.jpeg"))) == 16
        hashes.add(v37.digest(output / split / "_annotations.coco.json"))
        assert all(row["category_id"] == 1 and row["area"] > 0 for row in data["annotations"])
        assert all(row["bbox"][2] > 0 and row["bbox"][3] > 0 for row in data["annotations"])
    assert len(hashes) == 1


def test_converter_rejects_forbidden_paths(tmp_path):
    with pytest.raises(RuntimeError, match="firewall"):
        v37.build_dataset(ANNOTATIONS, tmp_path / "test_images", tmp_path / "out")


def test_runner_has_no_tune_audit_test_export_or_submission_path():
    tree = ast.parse(SOURCE)
    calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name))
    }
    assert "submission.csv" not in SOURCE
    assert "to_csv" not in calls and "submit" not in calls and "export" not in calls
    assert '"reads_tune": False' in SOURCE
    assert '"reads_audit": False' in SOURCE
    assert '"reads_test": False' in SOURCE
    assert "epochs=MAX_EPOCHS" in SOURCE
    assert "run_test=False" in SOURCE
    assert "wandb=False" in SOURCE
    assert "multi_scale=False" in SOURCE
    assert "scale_jitter=False" in SOURCE
    assert '"Tesla T4" not in str(gpu_name)' in SOURCE
    assert "tuple(gpu_capability or ()) != (7, 5)" in SOURCE
    assert 'result["infrastructure_blocked"] = not accelerator_ok' in SOURCE
    assert "repair Azure environment without changing V37 model/data/gates" in SOURCE
    assert "01_{manifest['records'][0]['image_id']}.jpeg" in SOURCE


def test_design_was_frozen_before_implementation():
    design = (ROOT / "docs/v37_design_2026-09-14.md").read_text()
    assert "before V37 dependency installation" in design
    assert "V37 is a separately frozen runtime-portability gate" in design
    assert "PQ >=0.85" in design or "PQ >=0.85" in design.replace(" ", "")
