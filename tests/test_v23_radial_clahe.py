import ast
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts import evaluate_v23_radial_clahe as evaluate
from scripts import train_v23_radial_clahe as train

ROOT = Path(__file__).resolve().parents[1]


def test_transform_is_deterministic_native_and_offdisk_zero():
    image_path = next((ROOT / "data/raw").rglob("20160411134234Ch.jpeg"))
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    first, first_info = train.transform_image(gray)
    second, second_info = train.transform_image(gray)
    assert first.shape == gray.shape == (2048, 2048)
    assert first.dtype == np.uint8
    assert np.array_equal(first, second)
    assert first_info == second_info
    disk = train.fit_disk(gray)
    yy, xx = np.indices(gray.shape)
    outside = np.hypot(xx - disk["cx"], yy - disk["cy"]) / disk["radius"] > train.DISK_LIMIT
    assert not first[outside].any()
    assert train.CENTER_TOLERANCE_FRACTION == 0.06
    assert first_info["changed_fraction"] > 0.5
    assert first_info["transformed_std"] > 0


def test_disk_fit_rejects_invalid_images():
    with pytest.raises(ValueError, match="contour not found"):
        train.fit_disk(np.zeros((2048, 2048), dtype=np.uint8))
    with pytest.raises(ValueError, match="expected 2048x2048"):
        train.fit_disk(np.zeros((32, 32), dtype=np.uint8))


def test_real_split_matches_pinned_v6_population(monkeypatch):
    annotation = next((ROOT / "data/raw").rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    provenance_path = ROOT / "outputs/v6_inference_source/v6_provenance.json"
    monkeypatch.setattr(train, "ANNOTATIONS", annotation)
    _, images, annotations = train.load_population()
    provenance = json.loads(provenance_path.read_text())
    split = train.dataset_partitions(images, annotations, provenance)
    assert len(split["train_files"]) == 566
    assert len(split["valid_files"]) == 141
    assert len(split["train_ids"]) == 937
    assert len(split["valid_ids"]) == 141
    assert not set(split["train_files"]) & set(split["valid_files"])


def test_training_configuration_is_frozen_maskratio1():
    args = train.training_arguments()
    smoke = train.training_arguments(epochs=1)
    assert args["epochs"] == 20
    assert smoke["fraction"] == 1.0
    assert args["batch"] == train.GLOBAL_BATCH == 1
    assert train.training_arguments(seed=train.REPLICATION_SEED)["seed"] == 20260912
    assert args["imgsz"] == train.TRAIN_RESOLUTION == 896
    assert args["mask_ratio"] == 1
    assert args["seed"] == 42
    assert args["resume"] is False
    assert args["optimizer"] == "AdamW"


def test_candidate_grid_and_tune_gate_are_frozen():
    candidates = {
        evaluate.candidate_key(cell["name"], threshold, confidence)
        for cell in evaluate.CELLS
        for threshold in evaluate.THRESHOLDS
        for confidence in evaluate.CONFIDENCES
    }
    assert len(candidates) == 84
    assert evaluate.TUNE_REQUIRED == pytest.approx(evaluate.V17_TUNE_PQ + 0.01)
    assert evaluate.MATCHED_CONTROL_DELTA == 0.01
    assert evaluate.V17_AUDIT_PQ == pytest.approx(0.4707634285309831)
    for candidate in candidates:
        cell, threshold, confidence = evaluate.parse_candidate(candidate)
        assert cell["name"] in {"iou030", "iou070"}
        assert threshold in evaluate.THRESHOLDS
        assert confidence in evaluate.CONFIDENCES


def test_tune_selection_requires_both_gates(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "OUTPUT", tmp_path)
    base = {
        "group": "tune",
        "results": {"iou030/t0.45_c0.30": {"pq": 0.45}},
        "raw_proposals": {
            cell: {"recall_gt_0_5": 0.7, "mean_best_iou": 0.6, "max_best_iou": 0.9}
            for cell in ("iou030", "iou070")
        },
    }
    control = {**base, "arm": "control"}
    arm = json.loads(json.dumps({**base, "arm": "transformed"}))
    arm["results"]["iou030/t0.45_c0.30"]["pq"] = evaluate.TUNE_REQUIRED + 0.001
    control_replica = json.loads(json.dumps({**base, "arm": "control-replica"}))
    arm_replica = json.loads(json.dumps({**base, "arm": "transformed-replica"}))
    control_replica["results"]["iou030/t0.45_c0.30"]["pq"] = 0.44
    arm_replica["results"]["iou030/t0.45_c0.30"]["pq"] = 0.441
    (tmp_path / "control_tune.json").write_text(json.dumps(control))
    (tmp_path / "transformed_tune.json").write_text(json.dumps(arm))
    (tmp_path / "control-replica_tune.json").write_text(json.dumps(control_replica))
    (tmp_path / "transformed-replica_tune.json").write_text(json.dumps(arm_replica))
    report = evaluate.select_tune()
    assert not report["audit_authorized"]  # only +0.0097 vs matched control
    control["results"]["iou030/t0.45_c0.30"]["pq"] = 0.44
    (tmp_path / "control_tune.json").write_text(json.dumps(control))
    report = evaluate.select_tune()
    assert report["audit_authorized"]
    arm_replica["results"]["iou030/t0.45_c0.30"]["pq"] = 0.439
    (tmp_path / "transformed-replica_tune.json").write_text(json.dumps(arm_replica))
    report = evaluate.select_tune()
    assert not report["audit_authorized"]


def test_scripts_never_export_or_submit():
    for relative in ("scripts/train_v23_radial_clahe.py", "scripts/evaluate_v23_radial_clahe.py"):
        source = (ROOT / relative).read_text()
        tree = ast.parse(source)
        called = {
            node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name))
        }
        assert "submit" not in called
        assert "to_csv" not in called or "evaluate_v23" in relative
        assert "competitions_submit" not in called
        assert "test/test_images" not in source or "train_v23" not in relative


def test_transform_contract_digest_is_order_stable():
    left = {"b": 2, "a": [1, 3]}
    right = {"a": [1, 3], "b": 2}
    assert train.digest_json(left) == train.digest_json(right)
    assert train.digest_json(left) == hashlib.sha256(
        json.dumps(left, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
