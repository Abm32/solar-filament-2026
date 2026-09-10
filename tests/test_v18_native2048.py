"""Guard the V18 native-2048 detector retrain: honest gates, real dataset, and
the resolution/mask_ratio combination this run actually tests.

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


v18 = load("v18_native2048_under_test", ROOT / "kaggle_inference/v18_native2048.py")

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


# --- the declared axes: resolution AND mask_ratio, unlike V16 (mask_ratio only)

def test_arm_trains_at_the_measured_feasible_resolution_with_mask_ratio_1():
    """TRAIN_RESOLUTION is 1536, not literally 2048: the smoke grid (2026-09-08)
    measured that full native imgsz=2048 OOMs on a single Kaggle T4 even at
    batch=1; 1536/batch=2 is the largest configuration confirmed to fit."""
    arguments = v18.training_arguments(v18.TRAIN_RESOLUTION, v18.ARM_MASK_RATIO, v18.GLOBAL_BATCH)
    assert arguments["imgsz"] == v18.TRAIN_RESOLUTION == 1536
    assert arguments["mask_ratio"] == v18.ARM_MASK_RATIO == 1


def test_control_and_arm_declare_different_inference_resolutions():
    """Unlike V16 (both candidates fixed at 1024), V18's control stays at its
    accepted native 1024 while the arm moves to TRAIN_RESOLUTION -- resolution
    is itself the axis under test, so the two inference resolutions must differ."""
    candidates = v18.declared_candidates()
    assert candidates[v18.CONTROL]["infer_resolution"] == 1024
    assert candidates[v18.ARM]["infer_resolution"] == v18.TRAIN_RESOLUTION == 1536
    assert candidates[v18.CONTROL]["infer_resolution"] != candidates[v18.ARM]["infer_resolution"]


def test_control_and_arm_declare_different_mask_ratios():
    candidates = v18.declared_candidates()
    assert candidates[v18.CONTROL]["mask_ratio"] == v18.CONTROL_MASK_RATIO == 2
    assert candidates[v18.ARM]["mask_ratio"] == v18.ARM_MASK_RATIO == 1


def test_training_config_matches_the_v12_v16_audited_recipe_except_imgsz_and_mask_ratio():
    """Copied verbatim from V12/V16's audited recipe; imgsz and mask_ratio are
    the two new axes this run moves together."""
    arguments = v18.training_arguments(v18.TRAIN_RESOLUTION, v18.ARM_MASK_RATIO, v18.GLOBAL_BATCH)
    expected = dict(epochs=15, optimizer="AdamW", lr0=1e-4, lrf=0.1, warmup_epochs=1.0,
                    nbs=64, seed=42, single_cls=True, overlap_mask=False,
                    mosaic=0.3, close_mosaic=5, save_period=5, patience=0,
                    hsv_v=0.3, degrees=10, translate=0.1, scale=0.3, fliplr=0.0, flipud=0.0)
    for key, value in expected.items():
        assert arguments[key] == value, key
    assert arguments["imgsz"] == 1536
    assert arguments["mask_ratio"] == 1


def test_rejects_unaudited_batches():
    with pytest.raises(ValueError):
        v18.training_arguments(1536, 1, 8)


def test_run_v18_sets_the_expandable_segments_allocator_before_any_cuda_use():
    """Regression test for the 2026-09-09 failure: smoke_probe() set
    PYTORCH_CUDA_ALLOC_CONF but run_v18() (the actual full-training entrypoint)
    never did, so the full run trained without the allocator configuration the
    smoke test's "fits" verdict relied on and later OOM'd via fragmentation
    that expandable_segments is meant to reduce. Source-inspected because
    run_v18() requires a CUDA GPU and cannot run in this CPU-only suite."""
    import inspect
    source = inspect.getsource(v18.run_v18)
    set_line = next((line for line in source.splitlines()
                     if "PYTORCH_CUDA_ALLOC_CONF" in line), None)
    assert set_line is not None, "run_v18 must configure the CUDA allocator itself"
    argparse_index = source.index("argparse.ArgumentParser")
    alloc_index = source.index("PYTORCH_CUDA_ALLOC_CONF")
    assert alloc_index < argparse_index, (
        "Allocator config must be set before argument parsing / any CUDA import path")


def test_batch_defaults_are_conservative_for_high_resolution():
    """imgsz=1536 is over 2x the pixel count of 1024; the declared batch must
    not silently match V16's imgsz=1024 batch of 4. Measured 2026-09-08: batch=2
    fit in the smoke probe (peak 11.9GB of 14.56GB) but the 2026-09-09 full run
    still OOM'd under sustained training via allocator fragmentation, so
    GLOBAL_BATCH was fixed to 1 (no separate fallback needed or possible)."""
    assert v18.GLOBAL_BATCH < 4
    assert v18.GLOBAL_BATCH == v18.FALLBACK_BATCH == 1


def test_train_seed_overrides_only_the_seed_field():
    default = v18.training_arguments(v18.TRAIN_RESOLUTION, v18.ARM_MASK_RATIO, v18.GLOBAL_BATCH)
    overridden = v18.training_arguments(v18.TRAIN_RESOLUTION, v18.ARM_MASK_RATIO, v18.GLOBAL_BATCH,
                                        train_seed=20260908)
    assert default["seed"] == v18.DEFAULT_TRAIN_SEED == 42
    assert overridden["seed"] == 20260908
    a, b = dict(default), dict(overridden)
    assert a.pop("seed") != b.pop("seed")
    assert a == b, "train_seed must not change any other training argument"


def test_stripped_checkpoint_forbids_resume():
    assert v18.training_arguments(v18.TRAIN_RESOLUTION, v18.ARM_MASK_RATIO,
                                  v18.GLOBAL_BATCH)["resume"] is False


# --- the candidate grid must be declared before any score is seen -------------

def test_declared_candidates_are_control_plus_exactly_one_arm():
    candidates = v18.declared_candidates()
    assert set(candidates) == {v18.CONTROL, v18.ARM}
    assert len(candidates) == 2


# --- caching must separate control (1024) and arm (1536) populations ----------

def test_cache_key_separates_resolution_mask_ratio_and_checkpoint():
    a = v18.cache_id("x.jpeg", "train", 2, 1024, "aa" * 32)
    b = v18.cache_id("x.jpeg", "train", 1, 1536, "aa" * 32)
    c = v18.cache_id("x.jpeg", "train", 1, 1536, "bb" * 32)
    d = v18.cache_id("x.jpeg", "train", 2, 1536, "aa" * 32)  # same res, different ratio
    assert len({a, b, c, d}) == 4


def test_cache_key_rejects_paths_and_unknown_splits():
    with pytest.raises(ValueError):
        v18.cache_id("sub/x.jpeg", "train", 1, 1536, "aa" * 32)
    with pytest.raises(ValueError):
        v18.cache_id("x.jpeg", "val", 1, 1536, "aa" * 32)


def test_control_and_arm_cache_ids_can_never_collide():
    """The whole point of including resolution in the cache key: control (1024)
    and arm (1536) proposals for the same file/checkpoint must land in different
    cache entries even if some other field matched."""
    control_id = v18.cache_id("x.jpeg", "train", v18.CONTROL_MASK_RATIO,
                              v18.CONTROL_INFERENCE_RESOLUTION, v18.V6_DETECTOR_SHA256)
    arm_id = v18.cache_id("x.jpeg", "train", v18.ARM_MASK_RATIO,
                          v18.ARM_INFERENCE_RESOLUTION, "ff" * 32)
    assert control_id != arm_id


# --- control verification is the trust anchor ---------------------------------

def test_control_must_reproduce_the_frozen_v6_counts():
    assert v18.verify_control({"tune": list(v18.V6_REFERENCE["tune"]),
                               "audit": list(v18.V6_REFERENCE["audit"])}) is True


def test_control_rejects_drifted_counts():
    drifted = {"tune": list(v18.V6_REFERENCE["tune"]), "audit": list(v18.V6_REFERENCE["audit"])}
    drifted["tune"] = [drifted["tune"][0], 625, 211, 508]
    with pytest.raises(ValueError, match="differ from the frozen V6 reference"):
        v18.verify_control(drifted)


def test_control_reference_is_the_accepted_038_pipeline():
    assert v18.V6_REFERENCE["tune"][1:] == [626, 211, 508]
    assert abs(v18.pq(v18.V6_REFERENCE["tune"]) - 0.4328232774299302) < 1e-12


# --- raw-proposal diagnostic: the number this run is actually trying to move --

def test_raw_proposal_iou_uses_only_the_unflipped_view():
    from pycocotools import mask as mask_utils
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[:2, :2] = 1
    rle_a = mask_utils.encode(np.asfortranarray(mask))
    gt = [rle_a]
    views = [([rle_a], [0.9]), ([], []), ([], []), ([], [])]
    result = v18.raw_proposal_iou(views, gt)
    assert len(result) == 1
    assert result[0] == pytest.approx(1.0)


def test_raw_proposal_iou_handles_no_proposals_or_no_gt():
    from pycocotools import mask as mask_utils
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[:2, :2] = 1
    rle = mask_utils.encode(np.asfortranarray(mask))
    assert v18.raw_proposal_iou([], []) == []
    assert v18.raw_proposal_iou([([], [])], []) == []
    assert v18.raw_proposal_iou([([], [])], [rle]) == [0.0]


# --- selection and the two independent bars ------------------------------------

def test_selection_uses_tuning_only_and_ignores_audit():
    results = {v18.CONTROL: {"tune": [10, 20, 5, 5], "audit": [1, 2, 1, 1]},
               v18.ARM: {"tune": [30, 40, 5, 5], "audit": [0, 1, 9, 9]}}
    assert v18.select_arm(results) == v18.ARM


def test_below_submit_bar_is_not_exported():
    results = {v18.ARM: {"tune": [200.0, 400, 300, 700], "audit": [80.0, 150, 120, 300]}}
    gate = v18.submission_and_promotion(v18.ARM, results)
    assert gate["export_authorized"] is False
    assert gate["promotable"] is False
    assert "below submit bar" in gate["recommendation"]


def test_clears_submit_bar_but_not_promotion_gate_is_recorded_not_submitted():
    tune_pq = v18.SUBMIT_BAR["min_tune_pq"] + 0.001
    audit_pq = v18.SUBMIT_BAR["min_audit_pq"] + 0.001
    tp, fp, fn = 700, 100, 100
    denom = tp + 0.5 * (fp + fn)
    results = {v18.ARM: {"tune": [tune_pq * denom, tp, fp, fn], "audit": [audit_pq * denom, tp, fp, fn]}}
    gate = v18.submission_and_promotion(v18.ARM, results)
    assert gate["export_authorized"] is True
    assert gate["tune_delta_vs_v13"] < v18.PROMOTION_GATE["required_tune_delta_vs_v13"]
    assert gate["promotable"] is False
    assert "not submit" in gate["recommendation"]


def test_clears_promotion_gate_only_with_a_real_margin_over_v13():
    tune_pq = (v18.V13_REFERENCE["tune_pq"]
              + v18.PROMOTION_GATE["required_tune_delta_vs_v13"] + 0.001)
    audit_pq = v18.V13_REFERENCE["audit_pq"] + 0.001
    tp, fp, fn = 700, 100, 100
    denom = tp + 0.5 * (fp + fn)
    results = {v18.ARM: {"tune": [tune_pq * denom, tp, fp, fn], "audit": [audit_pq * denom, tp, fp, fn]}}
    gate = v18.submission_and_promotion(v18.ARM, results)
    assert gate["export_authorized"] is True
    assert gate["promotable"] is True
    assert "consider spending a submission" in gate["recommendation"]


def test_submit_bar_and_promotion_gate_reference_v13_not_the_control():
    assert v18.SUBMIT_BAR["min_tune_pq"] == v18.V13_REFERENCE["tune_pq"] == 0.441777
    assert v18.PROMOTION_GATE["required_tune_delta_vs_v13"] == 0.01


# --- dataset regeneration, against the real annotations ------------------------

def test_split_reconstruction_matches_pinned_v6_provenance():
    images, annotations = load_coco()
    provenance = json.loads(PROVENANCE.read_text())
    partitions = v18.dataset_partitions(images, annotations, provenance)
    assert (len(partitions["train_files"]), len(partitions["valid_files"])) == (566, 141)
    assert (partitions["train_entries"], partitions["train_instances"]) == (937, 6606)
    assert not set(partitions["train_files"]) & set(partitions["valid_files"])


def test_split_reconstruction_rejects_a_tampered_provenance():
    images, annotations = load_coco()
    provenance = json.loads(PROVENANCE.read_text())
    provenance = dict(provenance, training_files=provenance["training_files"][:-1])
    with pytest.raises(ValueError, match="differs from pinned V6 provenance"):
        v18.dataset_partitions(images, annotations, provenance)


def test_polygon_conversion_normalizes_clips_and_keeps_six_decimals():
    line = v18.polygon_to_yolo_line([0, 0, 2048, 1024, -50, 4096], 2048, 2048)
    parts = line.split()
    assert parts[0] == "0"
    assert parts[1:] == ["0.000000", "0.000000", "1.000000", "0.500000", "0.000000", "1.000000"]


def test_polygon_conversion_rejects_degenerate_polygons():
    with pytest.raises(ValueError):
        v18.polygon_to_yolo_line([0, 0, 1, 1], 2048, 2048)


def test_dataset_export_writes_both_splits_and_validates(tmp_path):
    images, annotations = load_coco()
    provenance = json.loads(PROVENANCE.read_text())
    partitions = v18.dataset_partitions(images, annotations, provenance)
    train_names = partitions["train_files"][:2]
    valid_names = partitions["valid_files"][:2]
    by_file = {}
    for image_id, info in images.items():
        by_file.setdefault(info["file_name"], []).append(image_id)
    best = v18.best_annotator_ids(images, annotations)
    small = dict(partitions)
    small["train_ids"] = [i for n in train_names for i in sorted(by_file[n])]
    small["valid_ids"] = [best[n] for n in valid_names]
    small["train_instances"] = sum(
        len([p for e in annotations.get(i, []) for p in e["segmentation"] if len(p) >= 6])
        for i in small["train_ids"])
    root = tmp_path / "ds"
    counts = v18.write_yolo_dataset(root, images, annotations, small,
                                    DATA_ROOT / "train/train_images")
    assert counts["train"] == len(small["train_ids"]) and counts["val"] == len(small["valid_ids"])
    assert v18.verify_yolo_dataset(root, small)["train_instances"] == small["train_instances"]
    assert "0: filament" in (root / "data.yaml").read_text()


def test_dataset_verification_catches_missing_validation_labels(tmp_path):
    images, annotations = load_coco()
    provenance = json.loads(PROVENANCE.read_text())
    partitions = v18.dataset_partitions(images, annotations, provenance)
    by_file = {}
    for image_id, info in images.items():
        by_file.setdefault(info["file_name"], []).append(image_id)
    best = v18.best_annotator_ids(images, annotations)
    small = dict(partitions)
    small["train_ids"] = [i for n in partitions["train_files"][:2] for i in sorted(by_file[n])]
    small["valid_ids"] = [best[n] for n in partitions["valid_files"][:2]]
    small["train_instances"] = sum(
        len([p for e in annotations.get(i, []) for p in e["segmentation"] if len(p) >= 6])
        for i in small["train_ids"])
    root = tmp_path / "ds"
    v18.write_yolo_dataset(root, images, annotations, small, DATA_ROOT / "train/train_images")
    for path in (root / "labels/val").glob("*.txt"):
        path.unlink()
    with pytest.raises(ValueError, match="expected"):
        v18.verify_yolo_dataset(root, small)
