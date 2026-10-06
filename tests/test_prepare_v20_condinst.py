"""Tests for rule-safe V20 CondInst split preparation."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


prep = load("prepare_v20_under_test", ROOT / "scripts/prepare_v20_condinst.py")
DATA = ROOT / "data/raw/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"
ANNOTATIONS = DATA / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
requires_data = pytest.mark.skipif(not ANNOTATIONS.exists(), reason="competition data absent")


def test_protocol_bars_match_v17_and_target():
    assert prep.SUBMIT_BAR_TUNE == 0.448703
    assert prep.SUBMIT_BAR_AUDIT == 0.470763
    assert prep.PROMOTION_GATE_TUNE == 0.55


@requires_data
def test_real_conversion_preserves_every_record_annotation_and_file(tmp_path):
    contract = prep.prepare(ANNOTATIONS, tmp_path)
    assert contract["populations"]["train"]["records"] == 937
    assert contract["populations"]["tune"]["records"] == 150
    assert contract["populations"]["audit"]["records"] == 67
    assert sum(x["annotations"] for x in contract["populations"].values()) == 8199
    assert sum(x["files"] for x in contract["populations"].values()) == 707
    assert sum(x["records"] for x in contract["populations"].values()) == 1154


@requires_data
def test_outputs_are_one_class_integer_id_coco_and_split_isolated(tmp_path):
    prep.prepare(ANNOTATIONS, tmp_path)
    groups = {}
    for group in prep.EXPECTED:
        data = json.loads((tmp_path / f"condinst_{group}.json").read_text())
        assert data["categories"] == [{"id": 1, "name": "filament", "supercategory": "filament"}]
        assert all(isinstance(image["id"], int) for image in data["images"])
        assert all(isinstance(annotation["id"], int) for annotation in data["annotations"])
        assert all(annotation["category_id"] == 1 for annotation in data["annotations"])
        assert all("source_image_id" in image for image in data["images"])
        assert all("source_annotation_id" in annotation for annotation in data["annotations"])
        groups[group] = {image["file_name"] for image in data["images"]}
    assert groups["train"].isdisjoint(groups["tune"])
    assert groups["train"].isdisjoint(groups["audit"])
    assert groups["tune"].isdisjoint(groups["audit"])


@requires_data
def test_contract_excludes_test_leaking_and_unapproved_external_labels(tmp_path):
    contract = prep.prepare(ANNOTATIONS, tmp_path)
    policy = json.dumps(contract["data_policy"])
    assert "full public MAGFiLO archive" in policy
    assert "AHAS/MHAS" in policy
    assert contract["status"] == "PREPARED_NOT_AUTHORIZED_FOR_GPU_OR_SUBMISSION"
    assert contract["protocol"]["export_requires_both_bars"] is True


def test_population_guard_rejects_dropped_annotator_record():
    source = {
        "info": {}, "licenses": [],
        "images": [{"id": "x", "file_name": "a.jpeg", "width": 1, "height": 1}],
        "annotations": [],
    }
    with pytest.raises(ValueError, match="population mismatch"):
        prep.convert_split(source, {"a.jpeg"}, "tune")
