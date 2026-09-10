"""Guard the V16 mask_ratio comparison: matched budget, honest gates, real dataset.

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


v16 = load("v16_maskratio_under_test", ROOT / "kaggle_inference/v16_maskratio.py")

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

def test_arms_differ_only_in_mask_ratio():
    arguments = {arm: v16.training_arguments(v16.ARM_MASK_RATIO[arm], v16.GLOBAL_BATCH)
                 for arm in v16.ARMS}
    assert v16.verify_matched_budget(arguments) is True
    a, b = dict(arguments["arm_r2"]), dict(arguments["arm_r1"])
    assert a.pop("mask_ratio") == 2 and b.pop("mask_ratio") == 1
    assert a == b, "arms may only differ in mask_ratio"


def test_both_arms_train_at_the_fixed_imgsz():
    """Unlike V12, imgsz must NOT vary here -- mask_ratio is the isolated axis."""
    for arm in v16.ARMS:
        arguments = v16.training_arguments(v16.ARM_MASK_RATIO[arm], v16.GLOBAL_BATCH)
        assert arguments["imgsz"] == v16.TRAIN_RESOLUTION == 1024


def test_matched_budget_rejects_any_other_difference():
    arguments = {arm: v16.training_arguments(v16.ARM_MASK_RATIO[arm], v16.GLOBAL_BATCH)
                 for arm in v16.ARMS}
    arguments["arm_r1"]["lr0"] = 5e-4
    with pytest.raises(ValueError, match="differ outside mask_ratio"):
        v16.verify_matched_budget(arguments)


def test_matched_budget_rejects_a_smuggled_imgsz_difference():
    arguments = {arm: v16.training_arguments(v16.ARM_MASK_RATIO[arm], v16.GLOBAL_BATCH)
                 for arm in v16.ARMS}
    arguments["arm_r1"]["imgsz"] = 1536
    with pytest.raises(ValueError, match="fixed imgsz"):
        v16.verify_matched_budget(arguments)


def test_both_arms_must_share_one_global_batch():
    """A batch difference changes BatchNorm stats and warmup updates, so it is rejected."""
    arguments = {"arm_r2": v16.training_arguments(2, 4), "arm_r1": v16.training_arguments(1, 2)}
    with pytest.raises(ValueError, match="batch"):
        v16.verify_matched_budget(arguments)


def test_stripped_checkpoints_forbid_resume():
    """V6 best/last have optimizer=None; resuming them would be incoherent."""
    for mask_ratio in v16.ARM_MASK_RATIO.values():
        assert v16.training_arguments(mask_ratio, v16.GLOBAL_BATCH)["resume"] is False


def test_training_config_matches_the_v12_audited_values_except_mask_ratio():
    """Copied verbatim from V12's audited recipe; only mask_ratio is the new axis."""
    arguments = v16.training_arguments(1, v16.GLOBAL_BATCH)
    expected = dict(epochs=20, optimizer="AdamW", lr0=1e-4, lrf=0.1, warmup_epochs=1.0,
                    nbs=64, seed=42, single_cls=True, overlap_mask=False,
                    mosaic=0.3, close_mosaic=5, save_period=5, patience=0,
                    hsv_v=0.3, degrees=10, translate=0.1, scale=0.3, fliplr=0.0, flipud=0.0,
                    imgsz=1024)
    for key, value in expected.items():
        assert arguments[key] == value, key
    assert arguments["mask_ratio"] == 1


def test_rejects_unaudited_mask_ratios_and_batches():
    with pytest.raises(ValueError):
        v16.training_arguments(4, v16.GLOBAL_BATCH)
    with pytest.raises(ValueError):
        v16.training_arguments(1, 8)


def test_train_seed_overrides_only_the_seed_field():
    default = v16.training_arguments(1, v16.GLOBAL_BATCH)
    overridden = v16.training_arguments(1, v16.GLOBAL_BATCH, train_seed=20260908)
    assert default["seed"] == v16.DEFAULT_TRAIN_SEED == 42
    assert overridden["seed"] == 20260908
    a, b = dict(default), dict(overridden)
    assert a.pop("seed") != b.pop("seed")
    assert a == b, "train_seed must not change any other training argument"


def test_matched_budget_accepts_arms_sharing_a_non_default_train_seed():
    """A noise-pricing rerun uses one shared, non-42 train_seed for both arms;
    verify_matched_budget must still accept that -- it never varies across arms,
    only mask_ratio does."""
    arguments = {arm: v16.training_arguments(v16.ARM_MASK_RATIO[arm], v16.GLOBAL_BATCH,
                                             train_seed=20260908)
                for arm in v16.ARMS}
    assert v16.verify_matched_budget(arguments) is True
    assert {arguments[arm]["seed"] for arm in v16.ARMS} == {20260908}


# --- the candidate grid must be declared before any score is seen -------------

def test_declared_candidates_cover_control_plus_both_arms():
    candidates = v16.declared_candidates()
    assert v16.CONTROL in candidates
    for arm in v16.ARMS:
        assert f"v16_{arm}" in candidates
    assert len(candidates) == 1 + len(v16.ARMS)


def test_candidates_fix_inference_resolution_and_vary_mask_ratio():
    candidates = v16.declared_candidates()
    resolutions = {c["infer_resolution"] for c in candidates.values()}
    assert resolutions == {1024}
    ratios = {name: c["mask_ratio"] for name, c in candidates.items() if name != v16.CONTROL}
    assert ratios == {"v16_arm_r2": 2, "v16_arm_r1": 1}


# --- caching must never silently reuse V6/V12 proposals -----------------------

def test_cache_key_separates_checkpoint_and_mask_ratio():
    a = v16.cache_id("x.jpeg", "train", 1, "aa" * 32)
    b = v16.cache_id("x.jpeg", "train", 2, "aa" * 32)
    c = v16.cache_id("x.jpeg", "train", 1, "bb" * 32)
    assert len({a, b, c}) == 3


def test_cache_key_rejects_paths_and_unknown_splits():
    with pytest.raises(ValueError):
        v16.cache_id("sub/x.jpeg", "train", 1, "aa" * 32)
    with pytest.raises(ValueError):
        v16.cache_id("x.jpeg", "val", 1, "aa" * 32)


# --- control verification is the trust anchor ---------------------------------

def test_control_must_reproduce_the_frozen_v6_counts():
    assert v16.verify_control({"tune": list(v16.V6_REFERENCE["tune"]),
                               "audit": list(v16.V6_REFERENCE["audit"])}) is True


def test_control_rejects_drifted_counts():
    drifted = {"tune": list(v16.V6_REFERENCE["tune"]), "audit": list(v16.V6_REFERENCE["audit"])}
    drifted["tune"] = [drifted["tune"][0], 625, 211, 508]
    with pytest.raises(ValueError, match="differ from the frozen V6 reference"):
        v16.verify_control(drifted)


def test_control_reference_is_the_accepted_038_pipeline():
    """These are E015's exact tuning/audit counts, whose export scored 0.38."""
    assert v16.V6_REFERENCE["tune"][1:] == [626, 211, 508]
    assert abs(v16.pq(v16.V6_REFERENCE["tune"]) - 0.4328232774299302) < 1e-12


# --- raw-proposal diagnostic: the actual thing this run tests -----------------

def test_raw_proposal_iou_uses_only_the_unflipped_view():
    from pycocotools import mask as mask_utils
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[:2, :2] = 1
    rle_a = mask_utils.encode(np.asfortranarray(mask))
    gt = [rle_a]
    views = [([rle_a], [0.9]), ([], []), ([], []), ([], [])]
    result = v16.raw_proposal_iou(views, gt)
    assert len(result) == 1
    assert result[0] == pytest.approx(1.0)


def test_raw_proposal_iou_handles_no_proposals_or_no_gt():
    from pycocotools import mask as mask_utils
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[:2, :2] = 1
    rle = mask_utils.encode(np.asfortranarray(mask))
    assert v16.raw_proposal_iou([], []) == []
    assert v16.raw_proposal_iou([([], [])], []) == []
    assert v16.raw_proposal_iou([([], [])], [rle]) == [0.0]


# --- selection and the two independent bars ------------------------------------

def test_selection_uses_tuning_only_and_ignores_audit():
    results = {v16.CONTROL: {"tune": [10, 20, 5, 5], "audit": [1, 2, 1, 1]},
               "v16_arm_r2": {"tune": [10, 20, 5, 5], "audit": [99, 99, 0, 0]},
               "v16_arm_r1": {"tune": [30, 40, 5, 5], "audit": [0, 1, 9, 9]}}
    assert v16.select_arm(results) == "v16_arm_r1"


def test_below_submit_bar_is_not_exported():
    results = {"v16_arm_r1": {"tune": [200.0, 400, 300, 700], "audit": [80.0, 150, 120, 300]}}
    gate = v16.submission_and_promotion("v16_arm_r1", results)
    assert gate["export_authorized"] is False
    assert gate["promotable"] is False
    assert "below submit bar" in gate["recommendation"]


def test_clears_submit_bar_but_not_promotion_gate_is_recorded_not_submitted():
    """Mirrors V15's actual outcome: authorised to measure, not to submit."""
    tune_pq = v16.SUBMIT_BAR["min_tune_pq"] + 0.001
    audit_pq = v16.SUBMIT_BAR["min_audit_pq"] + 0.001
    tp, fp, fn = 700, 100, 100
    denom = tp + 0.5 * (fp + fn)
    results = {"v16_arm_r1": {"tune": [tune_pq * denom, tp, fp, fn],
                              "audit": [audit_pq * denom, tp, fp, fn]}}
    gate = v16.submission_and_promotion("v16_arm_r1", results)
    assert gate["export_authorized"] is True
    assert gate["tune_delta_vs_v13"] < v16.PROMOTION_GATE["required_tune_delta_vs_v13"]
    assert gate["promotable"] is False
    assert "not submit" in gate["recommendation"]


def test_clears_promotion_gate_only_with_a_real_margin_over_v13():
    tune_pq = (v16.V13_REFERENCE["tune_pq"]
              + v16.PROMOTION_GATE["required_tune_delta_vs_v13"] + 0.001)
    audit_pq = v16.V13_REFERENCE["audit_pq"] + 0.001
    tp, fp, fn = 700, 100, 100
    denom = tp + 0.5 * (fp + fn)
    results = {"v16_arm_r1": {"tune": [tune_pq * denom, tp, fp, fn],
                              "audit": [audit_pq * denom, tp, fp, fn]}}
    gate = v16.submission_and_promotion("v16_arm_r1", results)
    assert gate["export_authorized"] is True
    assert gate["promotable"] is True
    assert "consider spending a submission" in gate["recommendation"]


def test_submit_bar_and_promotion_gate_reference_v13_not_the_control():
    assert v16.SUBMIT_BAR["min_tune_pq"] == v16.V13_REFERENCE["tune_pq"] == 0.441777
    assert v16.PROMOTION_GATE["required_tune_delta_vs_v13"] == 0.01


# --- dataset regeneration, against the real annotations ------------------------

def test_split_reconstruction_matches_pinned_v6_provenance():
    images, annotations = load_coco()
    provenance = json.loads(PROVENANCE.read_text())
    partitions = v16.dataset_partitions(images, annotations, provenance)
    assert (len(partitions["train_files"]), len(partitions["valid_files"])) == (566, 141)
    assert (partitions["train_entries"], partitions["train_instances"]) == (937, 6606)
    assert not set(partitions["train_files"]) & set(partitions["valid_files"])


def test_split_reconstruction_rejects_a_tampered_provenance():
    images, annotations = load_coco()
    provenance = json.loads(PROVENANCE.read_text())
    provenance = dict(provenance, training_files=provenance["training_files"][:-1])
    with pytest.raises(ValueError, match="differs from pinned V6 provenance"):
        v16.dataset_partitions(images, annotations, provenance)


def test_dataset_partitions_ignores_train_seed_and_only_uses_split_seed():
    """The split must be identical regardless of any --train-seed override: it is
    reconstructed from SPLIT_SEED alone. dataset_partitions takes no seed argument,
    so this also documents that a noise-pricing rerun cannot accidentally reshuffle
    the pinned 566/141 split."""
    images, annotations = load_coco()
    provenance = json.loads(PROVENANCE.read_text())
    first = v16.dataset_partitions(images, annotations, provenance)
    second = v16.dataset_partitions(images, annotations, provenance)
    assert first["train_files"] == second["train_files"]
    assert first["valid_files"] == second["valid_files"]
    assert v16.SPLIT_SEED == 42


def test_polygon_conversion_normalizes_clips_and_keeps_six_decimals():
    line = v16.polygon_to_yolo_line([0, 0, 2048, 1024, -50, 4096], 2048, 2048)
    parts = line.split()
    assert parts[0] == "0"
    assert parts[1:] == ["0.000000", "0.000000", "1.000000", "0.500000", "0.000000", "1.000000"]


def test_polygon_conversion_rejects_degenerate_polygons():
    with pytest.raises(ValueError):
        v16.polygon_to_yolo_line([0, 0, 1, 1], 2048, 2048)


def test_dataset_export_writes_both_splits_and_validates(tmp_path):
    """The downloaded export had 391 train labels and NO val labels; catch that."""
    images, annotations = load_coco()
    provenance = json.loads(PROVENANCE.read_text())
    partitions = v16.dataset_partitions(images, annotations, provenance)
    small = dict(partitions)
    train_names = partitions["train_files"][:2]
    valid_names = partitions["valid_files"][:2]
    by_file = {}
    for image_id, info in images.items():
        by_file.setdefault(info["file_name"], []).append(image_id)
    best = v16.best_annotator_ids(images, annotations)
    small["train_ids"] = [i for n in train_names for i in sorted(by_file[n])]
    small["valid_ids"] = [best[n] for n in valid_names]
    small["train_instances"] = sum(
        len([p for e in annotations.get(i, []) for p in e["segmentation"] if len(p) >= 6])
        for i in small["train_ids"])
    root = tmp_path / "ds"
    counts = v16.write_yolo_dataset(root, images, annotations, small,
                                    DATA_ROOT / "train/train_images")
    assert counts["train"] == len(small["train_ids"]) and counts["val"] == len(small["valid_ids"])
    assert v16.verify_yolo_dataset(root, small)["train_instances"] == small["train_instances"]
    assert "0: filament" in (root / "data.yaml").read_text()


def test_dataset_verification_catches_missing_validation_labels(tmp_path):
    images, annotations = load_coco()
    provenance = json.loads(PROVENANCE.read_text())
    partitions = v16.dataset_partitions(images, annotations, provenance)
    by_file = {}
    for image_id, info in images.items():
        by_file.setdefault(info["file_name"], []).append(image_id)
    best = v16.best_annotator_ids(images, annotations)
    small = dict(partitions)
    small["train_ids"] = [i for n in partitions["train_files"][:2] for i in sorted(by_file[n])]
    small["valid_ids"] = [best[n] for n in partitions["valid_files"][:2]]
    small["train_instances"] = sum(
        len([p for e in annotations.get(i, []) for p in e["segmentation"] if len(p) >= 6])
        for i in small["train_ids"])
    root = tmp_path / "ds"
    v16.write_yolo_dataset(root, images, annotations, small, DATA_ROOT / "train/train_images")
    for path in (root / "labels/val").glob("*.txt"):
        path.unlink()
    with pytest.raises(ValueError, match="expected"):
        v16.verify_yolo_dataset(root, small)


def test_dataset_export_refuses_to_clobber_an_existing_directory(tmp_path):
    root = tmp_path / "ds"
    root.mkdir()
    images, annotations = load_coco()
    provenance = json.loads(PROVENANCE.read_text())
    partitions = v16.dataset_partitions(images, annotations, provenance)
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        v16.write_yolo_dataset(root, images, annotations, partitions,
                               DATA_ROOT / "train/train_images")


# --- the module must not smuggle in a submission or a GPU import --------------

def test_module_never_submits_or_exports_a_submission():
    source = (ROOT / "kaggle_inference/v16_maskratio.py").read_text()
    for forbidden in ("competition_submit", "submission.csv", "filament_id", "export_submission"):
        assert forbidden not in source, forbidden


def test_module_is_cpu_importable():
    """torch/ultralytics must load inside runtime paths, not at module import."""
    source = (ROOT / "kaggle_inference/v16_maskratio.py").read_text()
    header = source.split("def sha256", 1)[0]
    for forbidden in ("import torch", "from ultralytics", "import ultralytics"):
        assert forbidden not in header, forbidden


def test_frozen_postprocessing_constants_are_unchanged():
    """The comparison is detector-only; postprocessing must match the 0.38/0.39 run."""
    assert (v16.CONFIDENCE, v16.THRESHOLD, v16.MIN_AREA) == (0.4, 0.4, 20)
    assert (v16.POLICY, v16.MATCH_IOU) == ("flip_consensus", 0.5)
    assert v16.ULTRALYTICS == "8.4.138"
    assert v16.V6_DETECTOR_SHA256 == (
        "0abf568f5ffa504979a44c6ac4997c231adc38cb3c6d998461295c4dad3a38b8")
