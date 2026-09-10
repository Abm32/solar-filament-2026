"""Guard the V12 resolution comparison: matched budget, honest gate, real dataset.

These run on CPU. The module must stay importable without torch/ultralytics so
the local suite can check the protocol before any GPU time is spent.
"""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


v12 = load("v12_resolution_under_test", ROOT / "kaggle_inference/v12_resolution.py")

DATA_ROOT = ROOT / "data/raw/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"
ANNOTATIONS = DATA_ROOT / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
PROVENANCE = ROOT / "outputs/v6_inference_source/v6_provenance.json"


def load_coco():
    coco = json.loads(ANNOTATIONS.read_text())
    images = {image["id"]: image for image in coco["images"]}
    annotations = {}
    for entry in coco["annotations"]:
        annotations.setdefault(entry["image_id"], []).append(entry)
    return images, annotations


# --- protocol: the arms must be a controlled single-variable comparison --------

def test_arms_differ_only_in_training_resolution():
    arguments = {arm: v12.training_arguments(v12.ARM_RESOLUTION[arm], v12.GLOBAL_BATCH)
                 for arm in v12.ARMS}
    assert v12.verify_matched_budget(arguments) is True
    a, b = dict(arguments["res1024"]), dict(arguments["res1536"])
    assert a.pop("imgsz") == 1024 and b.pop("imgsz") == 1536
    assert a == b, "arms may only differ in imgsz"


def test_matched_budget_rejects_any_other_difference():
    arguments = {arm: v12.training_arguments(v12.ARM_RESOLUTION[arm], v12.GLOBAL_BATCH)
                 for arm in v12.ARMS}
    arguments["res1536"]["lr0"] = 5e-4
    with pytest.raises(ValueError, match="differ outside imgsz"):
        v12.verify_matched_budget(arguments)


def test_both_arms_must_share_one_global_batch():
    """A batch difference changes BatchNorm stats and warmup updates, so it is rejected."""
    arguments = {"res1024": v12.training_arguments(1024, 4),
                 "res1536": v12.training_arguments(1536, 2)}
    with pytest.raises(ValueError, match="batch"):
        v12.verify_matched_budget(arguments)


def test_stripped_checkpoints_forbid_resume():
    """V6 best/last have optimizer=None; resuming them would be incoherent."""
    for resolution in v12.ARM_RESOLUTION.values():
        assert v12.training_arguments(resolution, v12.GLOBAL_BATCH)["resume"] is False


def test_training_config_matches_the_audited_values():
    arguments = v12.training_arguments(1024, v12.GLOBAL_BATCH)
    expected = dict(epochs=20, optimizer="AdamW", lr0=1e-4, lrf=0.1, warmup_epochs=1.0,
                    nbs=64, seed=42, single_cls=True, mask_ratio=2, overlap_mask=False,
                    mosaic=0.3, close_mosaic=5, save_period=5, patience=0,
                    hsv_v=0.3, degrees=10, translate=0.1, scale=0.3, fliplr=0.0, flipud=0.0)
    for key, value in expected.items():
        assert arguments[key] == value, key


def test_rejects_unaudited_resolutions_and_batches():
    with pytest.raises(ValueError):
        v12.training_arguments(2048, v12.GLOBAL_BATCH)
    with pytest.raises(ValueError):
        v12.training_arguments(1024, 8)


# --- the 2x2 grid must be declared before any score is seen -------------------

def test_declared_candidates_cover_the_full_two_by_two_plus_control():
    candidates = v12.declared_candidates()
    assert v12.CONTROL in candidates
    for arm in v12.ARMS:
        for infer in v12.INFERENCE_RESOLUTIONS:
            assert f"v12_{arm}_infer{infer}" in candidates
    assert len(candidates) == 1 + len(v12.ARMS) * len(v12.INFERENCE_RESOLUTIONS)


def test_candidates_separate_train_resolution_from_inference_resolution():
    """V9 changed only inference resolution; this grid must distinguish the two."""
    candidates = v12.declared_candidates()
    grid = {(c["train_resolution"], c["infer_resolution"])
            for name, c in candidates.items() if name != v12.CONTROL}
    assert grid == {(1024, 1024), (1024, 1536), (1536, 1024), (1536, 1536)}


# --- caching must never silently reuse V6 proposals ---------------------------

def test_cache_key_separates_checkpoint_and_resolution():
    a = v12.cache_id("x.jpeg", "train", 1024, "aa" * 32)
    b = v12.cache_id("x.jpeg", "train", 1536, "aa" * 32)
    c = v12.cache_id("x.jpeg", "train", 1024, "bb" * 32)
    assert len({a, b, c}) == 3


def test_cache_key_rejects_paths_and_unknown_splits():
    with pytest.raises(ValueError):
        v12.cache_id("sub/x.jpeg", "train", 1024, "aa" * 32)
    with pytest.raises(ValueError):
        v12.cache_id("x.jpeg", "val", 1024, "aa" * 32)


# --- control verification is the trust anchor --------------------------------

def test_control_must_reproduce_the_frozen_v6_counts():
    assert v12.verify_control({"tune": list(v12.V6_REFERENCE["tune"]),
                               "audit": list(v12.V6_REFERENCE["audit"])}) is True


def test_control_rejects_drifted_counts():
    drifted = {"tune": list(v12.V6_REFERENCE["tune"]), "audit": list(v12.V6_REFERENCE["audit"])}
    drifted["tune"] = [drifted["tune"][0], 625, 211, 508]
    with pytest.raises(ValueError, match="differ from the frozen V6 reference"):
        v12.verify_control(drifted)


def test_control_reference_is_the_accepted_038_pipeline():
    """These are E015's exact tuning/audit counts, whose export scored 0.38."""
    assert v12.V6_REFERENCE["tune"][1:] == [626, 211, 508]
    assert abs(v12.pq(v12.V6_REFERENCE["tune"]) - 0.4328232774299302) < 1e-12


# --- selection and the gate --------------------------------------------------

def test_selection_uses_tuning_only_and_ignores_audit():
    results = {v12.CONTROL: {"tune": [10, 20, 5, 5], "audit": [1, 2, 1, 1]},
               "v12_res1024_infer1024": {"tune": [10, 20, 5, 5], "audit": [99, 99, 0, 0]},
               "v12_res1536_infer1536": {"tune": [30, 40, 5, 5], "audit": [0, 1, 9, 9]}}
    assert v12.select_arm(results) == "v12_res1536_infer1536"


def test_gate_reports_failure_honestly_without_blocking_measurement():
    results = {v12.CONTROL: {"tune": [426.5473399, 626, 211, 508], "audit": [182.752616, 265, 88, 194]},
               "v12_res1536_infer1536": {"tune": [428.0, 627, 211, 507], "audit": [183.0, 265, 88, 194]}}
    gate = v12.promotion_gate("v12_res1536_infer1536", results)
    assert gate["promotable"] is False
    assert 0 < gate["tune_delta"] < v12.GATE["required_tune_delta"]


def test_gate_requires_nonnegative_audit_delta():
    results = {v12.CONTROL: {"tune": [426.5473399, 626, 211, 508], "audit": [182.752616, 265, 88, 194]},
               "v12_res1536_infer1536": {"tune": [460.0, 660, 211, 474], "audit": [150.0, 240, 100, 219]}}
    gate = v12.promotion_gate("v12_res1536_infer1536", results)
    assert gate["tune_delta"] >= v12.GATE["required_tune_delta"]
    assert gate["audit_delta"] < 0 and gate["promotable"] is False


def test_gate_passes_only_on_a_real_gain():
    results = {v12.CONTROL: {"tune": [426.5473399, 626, 211, 508], "audit": [182.752616, 265, 88, 194]},
               "v12_res1536_infer1536": {"tune": [452.0, 655, 205, 479], "audit": [190.0, 272, 86, 187]}}
    gate = v12.promotion_gate("v12_res1536_infer1536", results)
    assert gate["tune_delta"] >= v12.GATE["required_tune_delta"]
    assert gate["audit_delta"] >= 0 and gate["promotable"] is True


# --- dataset regeneration, against the real annotations ----------------------

def test_split_reconstruction_matches_pinned_v6_provenance():
    images, annotations = load_coco()
    provenance = json.loads(PROVENANCE.read_text())
    partitions = v12.dataset_partitions(images, annotations, provenance)
    assert (len(partitions["train_files"]), len(partitions["valid_files"])) == (566, 141)
    assert (partitions["train_entries"], partitions["train_instances"]) == (937, 6606)
    assert not set(partitions["train_files"]) & set(partitions["valid_files"])


def test_split_reconstruction_rejects_a_tampered_provenance():
    images, annotations = load_coco()
    provenance = json.loads(PROVENANCE.read_text())
    provenance = dict(provenance, training_files=provenance["training_files"][:-1])
    with pytest.raises(ValueError, match="differs from pinned V6 provenance"):
        v12.dataset_partitions(images, annotations, provenance)


def test_polygon_conversion_normalizes_clips_and_keeps_six_decimals():
    line = v12.polygon_to_yolo_line([0, 0, 2048, 1024, -50, 4096], 2048, 2048)
    parts = line.split()
    assert parts[0] == "0"
    assert parts[1:] == ["0.000000", "0.000000", "1.000000", "0.500000", "0.000000", "1.000000"]


def test_polygon_conversion_rejects_degenerate_polygons():
    with pytest.raises(ValueError):
        v12.polygon_to_yolo_line([0, 0, 1, 1], 2048, 2048)


def test_dataset_export_writes_both_splits_and_validates(tmp_path):
    """The downloaded export had 391 train labels and NO val labels; catch that."""
    images, annotations = load_coco()
    provenance = json.loads(PROVENANCE.read_text())
    partitions = v12.dataset_partitions(images, annotations, provenance)
    # Keep the fixture small: two training filenames and two validation filenames.
    small = dict(partitions)
    train_names = partitions["train_files"][:2]
    valid_names = partitions["valid_files"][:2]
    by_file = {}
    for image_id, info in images.items():
        by_file.setdefault(info["file_name"], []).append(image_id)
    best = v12.best_annotator_ids(images, annotations)
    small["train_ids"] = [i for n in train_names for i in sorted(by_file[n])]
    small["valid_ids"] = [best[n] for n in valid_names]
    small["train_instances"] = sum(
        len([p for e in annotations.get(i, []) for p in e["segmentation"] if len(p) >= 6])
        for i in small["train_ids"])
    root = tmp_path / "ds"
    counts = v12.write_yolo_dataset(root, images, annotations, small,
                                    DATA_ROOT / "train/train_images")
    assert counts["train"] == len(small["train_ids"]) and counts["val"] == len(small["valid_ids"])
    assert v12.verify_yolo_dataset(root, small)["train_instances"] == small["train_instances"]
    assert "0: filament" in (root / "data.yaml").read_text()


def test_dataset_verification_catches_missing_validation_labels(tmp_path):
    images, annotations = load_coco()
    provenance = json.loads(PROVENANCE.read_text())
    partitions = v12.dataset_partitions(images, annotations, provenance)
    by_file = {}
    for image_id, info in images.items():
        by_file.setdefault(info["file_name"], []).append(image_id)
    best = v12.best_annotator_ids(images, annotations)
    small = dict(partitions)
    small["train_ids"] = [i for n in partitions["train_files"][:2] for i in sorted(by_file[n])]
    small["valid_ids"] = [best[n] for n in partitions["valid_files"][:2]]
    small["train_instances"] = sum(
        len([p for e in annotations.get(i, []) for p in e["segmentation"] if len(p) >= 6])
        for i in small["train_ids"])
    root = tmp_path / "ds"
    v12.write_yolo_dataset(root, images, annotations, small, DATA_ROOT / "train/train_images")
    for path in (root / "labels/val").glob("*.txt"):
        path.unlink()
    with pytest.raises(ValueError, match="expected"):
        v12.verify_yolo_dataset(root, small)


def test_dataset_export_refuses_to_clobber_an_existing_directory(tmp_path):
    root = tmp_path / "ds"
    root.mkdir()
    images, annotations = load_coco()
    provenance = json.loads(PROVENANCE.read_text())
    partitions = v12.dataset_partitions(images, annotations, provenance)
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        v12.write_yolo_dataset(root, images, annotations, partitions,
                               DATA_ROOT / "train/train_images")


# --- the module must not smuggle in a submission or a GPU import -------------

def test_module_never_submits_or_exports_a_submission():
    source = (ROOT / "kaggle_inference/v12_resolution.py").read_text()
    for forbidden in ("competition_submit", "submission.csv", "filament_id", "export_submission"):
        assert forbidden not in source, forbidden


def test_module_is_cpu_importable():
    """torch/ultralytics must load inside runtime paths, not at module import."""
    source = (ROOT / "kaggle_inference/v12_resolution.py").read_text()
    header = source.split("def sha256", 1)[0]
    for forbidden in ("import torch", "from ultralytics", "import ultralytics"):
        assert forbidden not in header, forbidden


def test_frozen_postprocessing_constants_are_unchanged():
    """The comparison is detector-only; postprocessing must match the 0.38/0.39 run."""
    assert (v12.CONFIDENCE, v12.THRESHOLD, v12.MIN_AREA) == (0.4, 0.4, 20)
    assert (v12.POLICY, v12.MATCH_IOU) == ("flip_consensus", 0.5)
    assert v12.ULTRALYTICS == "8.4.138"
    assert v12.V6_DETECTOR_SHA256 == (
        "0abf568f5ffa504979a44c6ac4997c231adc38cb3c6d998461295c4dad3a38b8")
