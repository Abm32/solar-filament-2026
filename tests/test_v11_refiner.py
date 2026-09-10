"""V11 split, geometry, matched budgets, selection, and isolated staging contracts."""
import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from pycocotools import mask as mask_utils
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v11 = load("v11_test", "kaggle_inference/v11_refiner.py")
stager = load("v11_stager_test", "scripts/prepare_v11_refiner.py")


def pure_core():
    path = ROOT / "kaggle_inference/optimize_inference.py"
    tree = ast.parse(path.read_text())
    names = {"encode_crop", "clean_refined_predictions"}
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = dict(np=np, cv2=cv2, mask_utils=mask_utils, MIN_AREA=20)
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


def counts(score):
    return [score * 100, 100, 0, 0]


def test_protocol_is_bounded_and_smoke_is_distinct():
    full, smoke = v11.protocol(), v11.protocol(True)
    assert full["epochs"] == 4 and len(full["candidates"]) == 17
    assert full["training_precision"].startswith("float32")
    trained = [c for c in full["candidates"].values() if c["epoch"]]
    assert len({(c["arm"], c["epoch"]) for c in trained}) == 12
    assert sum(c["mode"] == "refined_free" for c in trained) == 4
    assert all(c["conditioned"] for c in trained if c["mode"] == "refined_free")
    assert smoke["epochs"] == 1 and len(smoke["candidates"]) == 5
    assert full["smoke_limits"] is None and smoke["smoke_limits"]["training_files"] == 3
    with pytest.raises(ValueError):
        v11.declared_candidates(7)


def test_training_split_rejects_validation_and_test_names():
    provenance = dict(training_files=["train-a", "train-b"], validation_files=["tune", "audit"])
    assert v11.verify_training_names(["train-a"], provenance, {"tune"}, {"audit"}) == ["train-a"]
    for names in (["tune"], ["audit"], ["test"], ["train-a", "train-a"], []):
        with pytest.raises(ValueError):
            v11.verify_training_names(names, provenance, {"tune"}, {"audit"})
    with pytest.raises(ValueError):
        v11.verify_training_names(["train-a"], provenance, {"tune"}, {"audit"}, require_full=True)
    with pytest.raises(ValueError):
        v11.verify_training_names(["train-a"], provenance, {"tune", "audit"}, {"audit"})


def test_matching_ignores_unmatched_ambiguous_and_duplicate_proposals():
    # p0/p1 compete for GT0, p2 ambiguous, p3 unmatched, p4 matches GT1.
    iou = np.array([[0.7, 0.8, 0.60, 0.05, 0.05], [0.01, 0.01, 0.25, 0.02, 0.55]])
    assert v11.match_positive_proposals(iou, [0.9, 0.4, 0.95, 0.99, 0.5]) == [(0, 1), (1, 4)]
    assert v11.match_positive_proposals(np.zeros((2, 0)), []) == []
    assert v11.match_positive_proposals(np.zeros((0, 2)), [0.1, 0.2]) == []
    assert v11.match_positive_proposals(np.array([[0.29]]), [0.99]) == []
    assert v11.match_positive_proposals(np.array([[0.3]]), [0.01]) == [(0, 0)]


def test_matching_keeps_annotator_targets_separate_and_ties_stable():
    # Same proposal can map differently in independent annotator records.
    assert v11.match_positive_proposals([[0.8], [0.01]], [0.4]) == [(0, 0)]
    assert v11.match_positive_proposals([[0.02], [0.75]], [0.4]) == [(1, 0)]
    assert v11.match_positive_proposals([[0.7, 0.7, 0.7]], [0.5, 0.7, 0.7]) == [(0, 1)]
    assert v11.match_positive_proposals([[0.4], [0.20]], [0.9]) == []
    with pytest.raises(ValueError):
        v11.match_positive_proposals([[np.nan]], [0.4])
    with pytest.raises(ValueError):
        v11.match_positive_proposals([[0.5]], [])


def test_native_crop_keeps_actual_seed_and_target_distinct():
    gray = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64).astype(np.uint8)
    seed, target = np.zeros((64, 64), np.uint8), np.zeros((64, 64), np.uint8)
    seed[20:30, 22:25] = 1
    target[18:32, 25:29] = 1
    bounds = (16, 16, 40, 40)
    image, actual_seed, actual_target = v11.crop_triplet(gray, seed, target, bounds, size=24)
    np.testing.assert_array_equal(actual_seed, seed[16:40, 16:40])
    np.testing.assert_array_equal(actual_target, target[16:40, 16:40])
    assert not np.array_equal(actual_seed, actual_target)
    np.testing.assert_allclose(image, gray[16:40, 16:40] / 255, atol=1e-7)
    with pytest.raises(ValueError):
        v11.crop_triplet(gray, seed, target, (-1, 0, 20, 20))


def test_jitter_is_deterministic_square_and_inside_native_image():
    for bounds in ((0, 0, 96, 96), (1900, 1900, 2048, 2048), (800, 500, 1100, 800)):
        a = v11.jitter_bounds(bounds, (2048, 2048), np.random.default_rng(12))
        b = v11.jitter_bounds(bounds, (2048, 2048), np.random.default_rng(12))
        assert a == b
        x0, y0, x1, y1 = a
        assert 0 <= x0 < x1 <= 2048 and 0 <= y0 < y1 <= 2048
        assert x1 - x0 == y1 - y0


def test_spatial_augmentation_is_joint_and_photometry_excludes_masks():
    seed = np.zeros((24, 24), np.uint8)
    seed[3:7, 8:14] = 1
    target = seed.copy()
    target[14:19, 5:10] = 1
    image = seed.astype(np.float32)
    for seed_value in range(20):
        aug_image, aug_seed, aug_target = v11.augment_triplet(image, seed, target, np.random.default_rng(seed_value))
        assert set(np.unique(aug_seed)) == {0, 1}
        assert set(np.unique(aug_target)) == {0, 1}
        np.testing.assert_array_equal(aug_image > 0, aug_seed > 0)
        assert aug_seed.sum() == seed.sum() and aug_target.sum() == target.sum()
        assert np.all(aug_seed <= aug_target)
        assert all(a.flags.c_contiguous for a in (aug_image, aug_seed, aug_target))


def test_identical_sample_order_preserves_cohort_without_proposal_oversampling():
    records = [dict(file_name=name) for name in ("a", "b", "a", "c", "b")]
    for epoch in range(1, 5):
        order = v11.epoch_sample_order(records, epoch)
        assert sorted(order) == list(range(5))
        assert order == v11.epoch_sample_order(records, epoch)
    rows = [dict(samples=17, updates=2, sample_order_sha256="same") for _ in range(4)]
    histories = {arm: [dict(row) for row in rows] for arm in v11.ARMS}
    v11.validate_matched_budgets(histories, 17, 4)
    histories[v11.ARMS[1]][2]["samples"] = 16
    with pytest.raises(ValueError):
        v11.validate_matched_budgets(histories, 17, 4)


def test_thin_target_erased_by_resize_stops_without_silent_sample_drop(tmp_path):
    gray = np.zeros((2048, 2048), np.uint8)
    target = np.zeros_like(gray)
    target[100, 100:1900] = 1  # Nearest-neighbor sample rows are multiples of eight.
    rle = mask_utils.encode(np.asfortranarray(target))
    serialized = dict(size=list(rle["size"]), counts=rle["counts"].decode())
    name = "thin.jpeg"
    cv2.imwrite(str(tmp_path / name), gray)
    record = dict(file_name=name, annotation_id=1, seed=serialized, target=serialized,
                  gt_bounds=[0, 0, 2048, 2048], proposal_bounds=[0, 0, 2048, 2048])
    for arm in v11.ARMS:
        dataset = v11.ProposalDataset([record], tmp_path, arm, 1)
        assert len(dataset) == 1
        with pytest.raises(ValueError, match="vanished on resize"):
            dataset.arrays(0, augment=False)
        assert len(dataset) == 1


def test_selection_uses_complete_tuning_only_and_prefers_control_on_tie():
    configs = v11.declared_candidates()
    tuning = {key: counts(0.44) for key in configs}
    assert v11.select_candidate(configs, tuning) == v11.CONTROL
    winner = "v11_proposal_conditioned_e3_refined_free"
    tuning[winner] = counts(0.46)
    assert v11.select_candidate(configs, tuning) == winner
    assert v11.audit_candidates(winner, configs) == [v11.CONTROL, winner]
    assert v11.audit_candidates(v11.CONTROL, configs) == [v11.CONTROL]
    with pytest.raises(ValueError):
        v11.audit_candidates(winner, configs, smoke=True)
    with pytest.raises(ValueError):
        v11.select_candidate(configs, {v11.CONTROL: counts(0.4)})


def test_promotion_requires_tune_gain_and_both_historical_audit_baselines():
    selected = "candidate"
    tuning = {v11.CONTROL: counts(0.4328), selected: counts(0.445)}
    audit = {v11.CONTROL: counts(0.4501), selected: counts(0.452)}
    assert v11.promotion_gate(selected, tuning, audit)["promotable"]
    audit[selected] = counts(0.451)
    assert not v11.promotion_gate(selected, tuning, audit)["promotable"]  # Below V10 audit.
    audit[selected] = counts(0.452)
    tuning[selected] = counts(0.44)
    assert not v11.promotion_gate(selected, tuning, audit)["promotable"]  # < +.01.
    assert not v11.promotion_gate(selected, tuning, audit, smoke=True)["promotable"]
    with pytest.raises(ValueError):
        v11.promotion_gate(selected, tuning, {})


def test_inference_refines_all_groups_then_confidence_filters_and_owns_pixels(tmp_path, monkeypatch):
    core = pure_core()
    masks = []
    bounds = [(10, 10, 30, 30), (20, 20, 40, 40), (80, 80, 100, 100)]
    for x0, y0, x1, y1 in bounds:
        mask = np.zeros((2048, 2048), np.uint8)
        mask[y0:y1, x0:x1] = 1
        masks.append(mask)
    rles = [mask_utils.encode(np.asfortranarray(mask)) for mask in masks]
    observed = []

    def crop_probabilities(base, model, gray, passed_masks, tta):
        observed.append((len(passed_masks), tta))
        return bounds, [], [np.ones((256, 256), np.float32) for _ in passed_masks]

    core.crop_probabilities = crop_probabilities
    monkeypatch.setattr(v11, "core", core, raising=False)
    monkeypatch.setattr(v11, "tta", SimpleNamespace(cluster_views=lambda views, policy: (rles, [0.8, 0.5, 0.2])), raising=False)
    monkeypatch.setattr(v11, "v10", SimpleNamespace(read_raw_views=lambda path: []), raising=False)
    path = tmp_path / "image.jpeg"
    cv2.imwrite(str(path), np.zeros((2048, 2048), np.uint8))
    runtime = dict(base=SimpleNamespace(), inventory={("train", path.name): "raw"})
    output = v11.predict_modes(runtime, None, path, "train", False, ["refined_free", "refined_constrained"])
    assert observed == [(3, True)]
    for mode in output:
        assert len(output[mode]) == 2
        a, b = [mask_utils.decode(rle).astype(bool) for rle in output[mode]]
        assert a.sum() == 400 and b.sum() == 300 and not (a & b).any()


def test_empty_refinement_falls_back_to_real_proposal_and_enforces_min_area():
    core = pure_core()
    mask = np.zeros((100, 100), np.uint8)
    mask[10:20, 10:20] = 1
    rles, scores = core.clean_refined_predictions([mask], [0.8], [(0, 0, 40, 40)],
                                                 [np.zeros((256, 256), np.float32)], 0.4, "refined_free")
    assert scores.tolist() == [0.8]
    np.testing.assert_array_equal(mask_utils.decode(rles[0]), mask)
    mask[10:20, 10:20] = 0
    mask[10, 10] = 1
    rles, _ = core.clean_refined_predictions([mask], [0.8], [(0, 0, 40, 40)],
                                            [np.zeros((256, 256), np.float32)], 0.4, "refined_free")
    assert rles == []


def test_stager_strips_only_main_and_rejects_ambiguous_entrypoints():
    source = "value = 1\nif __name__ == '__main__':\n    run()\nother = 2\n"
    assert stager.without_main(source) == "value = 1\nother = 2\n"
    with pytest.raises(ValueError):
        stager.without_main("value=1\n")
    with pytest.raises(ValueError):
        stager.without_main(source + "if __name__ == '__main__':\n    another()\n")


def test_real_staging_pins_sources_references_and_separate_notebook_ids(tmp_path):
    output = stager.stage(ROOT, tmp_path / "smoke", smoke=True)
    script = (output / "v11_refiner.py").read_text()
    tree = ast.parse(script)
    values = {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body
              if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
    assert list(values["SOURCES"]) == ["optimize_inference", "detector_tta", "v10_quality", "v11_refiner"]
    assert len(values["RAW_CACHE_SHA256"]) == 321
    import hashlib
    for name, source in values["SOURCES"].items():
        assert hashlib.sha256(source.encode()).hexdigest() == values["MODULE_SHA256"][name]
        assert stager.without_main((ROOT / "kaggle_inference" / f"{name}.py").read_text()) == source
    metadata = json.loads((output / "kernel-metadata.json").read_text())
    assert metadata["id"] == "abhimanyu122/solar-filament-v11-refiner-smoke"
    assert "timeout" not in metadata
    assert metadata["docker_image"].endswith("37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461")
    assert metadata["docker_image_pinning_type"] == "original"
    manifest = json.loads((output / "staging_manifest.json").read_text())
    assert manifest["protocol"]["smoke"] and manifest["v10_reference"] == v11.V10_REFERENCE
    assert manifest["script_sha256"] == v11.sha256(output / "v11_refiner.py")
    full = stager.stage(ROOT, tmp_path / "full")
    assert json.loads((full / "kernel-metadata.json").read_text())["id"] == "abhimanyu122/solar-filament-v11-refiner"
    with pytest.raises(ValueError):
        stager.stage(ROOT, tmp_path / "bad", train_limit=1)
