"""V10 geometry, control fidelity, source guards and selection without torch."""
import ast
import gzip
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import pytest
from pycocotools import mask as mask_utils

ROOT = Path(__file__).resolve().parents[1]


def cpu_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pure_functions(path, names, namespace):
    tree = ast.parse(path.read_text())
    module = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef)
                             and node.name in names], type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


v10 = cpu_module("v10_test_module", ROOT / "kaggle_inference/v10_quality.py")
core_ns = pure_functions(ROOT / "kaggle_inference/optimize_inference.py",
                        {"encode_crop", "encode", "clean_refined_predictions", "pq_counts", "pq_value",
                         "annotation_rle", "rounded_legacy_rle", "write_results"},
                        dict(np=np, cv2=cv2, pd=pd, mask_utils=mask_utils, MIN_AREA=20))
v10.core = SimpleNamespace(**core_ns)
tta_ns = pure_functions(ROOT / "kaggle_inference/detector_tta.py", {"cluster_views"},
                       dict(np=np, mask_utils=mask_utils, POLICIES=("single", "flip_max", "flip_consensus")))
v10.tta = SimpleNamespace(**tta_ns)


def rectangle(x0=2, x1=12, y0=5, y1=15, shape=(48, 48)):
    mask = np.zeros(shape, dtype=np.uint8)
    mask[y0:y1, x0:x1] = 1
    return mask


def encoded(mask):
    return mask_utils.encode(np.asfortranarray(mask))


def record(mask, score=0.8, probability=0.9, detector="v6", index=0, views=4):
    height, width = mask.shape
    group = dict(consensus_confidence=score, support_count=views, views=list(range(views)),
                 view_scores={i: score for i in range(views)})
    probabilities = np.where(mask, probability, 0).astype(np.float32)
    return v10.make_record(group, mask, (0, 0, width, height), probabilities,
                           0.4 if detector == "v6" else 0.45, detector, index)


def predictions(records, quality="none", rescue="none", shape=(48, 48)):
    config = next(c for c in v10.declared_candidates().values()
                  if c["quality"] == quality and c["rescue"] == rescue)
    return v10.build_predictions(records, config, shape)


def test_group_metadata_exactly_matches_existing_geometry_and_empty_votes():
    a, b, c = [encoded(rectangle(x0=x, x1=x + 10)) for x in (2, 5, 8)]
    views = [([a, b], [0.9, 0.8]), ([b], [0.7]), ([c], [0.6]), ([], [])]
    groups = v10.group_views(views)
    old_masks, old_scores = tta_ns["cluster_views"](views, "flip_consensus")
    assert [g["rle"] for g in groups] == old_masks
    assert [g["consensus_confidence"] for g in groups] == old_scores
    assert all(g["support_count"] == len(set(g["views"])) for g in groups)
    assert all(len(g["anchor_ious"]) == g["support_count"] for g in groups)
    assert groups[0]["support_count"] < 4
    assert v10.group_views([([], [])] * 4) == []
    with pytest.raises(ValueError):
        v10.group_views([([], [])] * 3)


@pytest.mark.parametrize("seed", range(5))
def test_control_cleanup_matches_original_before_and_after_confidence_cut(seed):
    rng = np.random.default_rng(seed)
    masks = [rectangle(x0=x, x1=x + 15) for x in (1, 4, 15, 25)]
    bounds = [(0, 0, 48, 48)] * 4
    probabilities = [rng.random((16, 16), dtype=np.float32) for _ in masks]
    probabilities[2][:] = 0  # Exact old empty-refinement fallback.
    scores = [0.4, 0.9, 0.4, 0.25]  # Stable tie ordering and suffix filtering.
    records = [v10.make_record(dict(consensus_confidence=s, support_count=4, views=[0, 1, 2, 3],
                               view_scores={i: s for i in range(4)}), m, b, p, 0.4, "v6", i)
               for i, (m, b, p, s) in enumerate(zip(masks, bounds, probabilities, scores))]
    expected, kept_scores = core_ns["clean_refined_predictions"](
        masks, scores, bounds, probabilities, 0.4, "refined_constrained")
    expected = [r for r, score in zip(expected, kept_scores) if score >= 0.4]
    actual = predictions({"v6": records, "v8": []})
    assert [r["rle"] for r in actual] == expected


def test_faint_thin_filament_is_preserved_by_control_and_empty_drop_is_isolated():
    mask = rectangle(3, 4, 2, 32)
    faint = record(mask, probability=0.39)
    assert faint["empty_fallback"]
    assert faint["retention_fraction"] == 0
    for quality in ("none", "quality_rank"):
        actual = predictions({"v6": [faint], "v8": []}, quality)
        assert actual[0]["rle"] == encoded(mask)
    assert predictions({"v6": [faint], "v8": []}, "drop_empty") == []
    assert predictions({"v6": [], "v8": []}) == []


def test_quality_gates_and_features_are_pre_overlap_and_do_not_change_control():
    mask = rectangle()
    weak, strong = record(mask, probability=0.49), record(mask, probability=0.70)
    assert weak["retention_fraction"] == 1
    assert not v10.quality_accepts(weak, "proposal50")
    assert v10.quality_accepts(strong, "proposal50")
    assert not v10.quality_accepts(dict(strong, support_count=2), "views3")
    assert v10.quality_accepts(dict(strong, support_count=3), "views3")
    # A tiny confident fragment cannot bypass poor original-proposal support.
    assert not v10.quality_accepts(dict(weak, foreground_probability=0.99), "proposal50")
    assert not weak["empty_fallback"]
    with pytest.raises(ValueError):
        v10.quality_accepts(weak, "unknown")


def test_reranking_rebuilds_ownership_from_untrimmed_masks():
    a, b = rectangle(2, 17), rectangle(10, 25)
    records = {"v6": [record(a, score=0.8, probability=0.5, index=0),
                       record(b, score=0.7, probability=0.95, index=1)], "v8": []}
    baseline = predictions(records)
    ranked = predictions(records, "quality_rank")
    assert baseline[0]["proposal_id"] == 0
    assert ranked[0]["proposal_id"] == 1
    np.testing.assert_array_equal(mask_utils.decode(ranked[0]["rle"]), b)
    assert mask_utils.decode([p["rle"] for p in ranked]).sum(axis=2).max() == 1
    # Reuse of the cache is lossless and non-mutating.
    assert predictions(records) == baseline


@pytest.mark.parametrize("rescue_x", [16, 17])
def test_v8_rescue_preserves_v6_pixel_ownership_and_distinct_neighbors(rescue_x):
    anchor, rescue = rectangle(2, 17), rectangle(rescue_x, rescue_x + 15)
    records = {"v6": [record(anchor, score=0.4)],
               "v8": [record(rescue, score=0.95, detector="v8")]}
    actual = predictions(records, rescue="rescue35")
    assert len(actual) == 2
    assert actual[0]["detector"] == "v6"
    np.testing.assert_array_equal(mask_utils.decode(actual[0]["rle"]), anchor)
    np.testing.assert_array_equal(mask_utils.decode(actual[1]["rle"]), rescue & ~anchor)


def test_rescue_rejects_heavily_clipped_residual_and_accepts_exact_ninety_percent():
    anchor = rectangle(2, 12)
    for x, expected_count in ((11, 2), (10, 1)):
        rescue = rectangle(x, x + 10)
        actual = predictions({"v6": [record(anchor)], "v8": [record(rescue, detector="v8")]}, rescue="rescue35")
        assert len(actual) == expected_count
        np.testing.assert_array_equal(mask_utils.decode(actual[0]["rle"]), anchor)


def test_duplicate_and_contained_v8_predictions_cannot_create_extra_instances():
    anchor = rectangle(2, 22)
    for rescue in (anchor, rectangle(4, 12)):
        actual = predictions({"v6": [record(anchor)], "v8": [record(rescue, detector="v8")]}, rescue="rescue35")
        assert len(actual) == 1


def test_crossing_thin_instances_keep_separate_ownership_and_existing_minimum_area():
    anchor, rescue = rectangle(3, 34, 20, 21), rectangle(20, 21, 3, 34)
    actual = predictions({"v6": [record(anchor)], "v8": [record(rescue, detector="v8")]}, rescue="rescue35")
    assert len(actual) == 2
    assert int(mask_utils.area(actual[1]["rle"])) == 30
    assert mask_utils.decode([p["rle"] for p in actual]).sum(axis=2).max() == 1
    too_small = record(rectangle(5, 6, 2, 21), detector="v8")
    assert predictions({"v6": [], "v8": [too_small]}, rescue="rescue35") == []


@pytest.mark.parametrize("change", ["views", "confidence", "quality", "empty"])
def test_v8_rescue_requires_separate_confidence_support_and_quality(change):
    candidate = record(rectangle(), detector="v8")
    if change == "views":
        candidate["support_count"] = 2
    elif change == "confidence":
        candidate["consensus_confidence"] = 0.34
    elif change == "quality":
        candidate["proposal_foreground_probability"] = 0.49
    else:
        candidate["empty_fallback"] = True
    assert predictions({"v6": [], "v8": [candidate]}, rescue="rescue35") == []


def test_candidate_eligibility_and_selection_do_not_use_audit_or_allow_unjustified_combinations():
    configs = v10.declared_candidates()
    assert len(configs) == 15
    isolated = v10.isolated_candidates(configs)
    assert len(isolated) == 7
    scores = {key: 0.4 for key in isolated}
    scores["v10_drop_empty_none"] = 0.41
    scores["v10_none_rescue35"] = 0.42
    eligible = v10.eligible_candidates(configs, scores)
    assert set(eligible) - set(isolated) == {"v10_drop_empty_rescue35"}
    scores["v10_drop_empty_rescue35"] = 0.42
    assert v10.select_candidate(eligible, scores) == "v10_none_rescue35"  # Simpler on a tie.
    assert v10.select_candidate(isolated, {key: 0.4 for key in isolated}) == v10.CONTROL


def source_fixture():
    provenance = dict(training_files=["train.jpeg"], validation_files=["tune.jpeg", "audit.jpeg"],
                      sha256={"yolo_best.pt": v10.SOURCE_PINS["v6"]["detector"]})
    metadata = dict(experiment=v10.SOURCE_PINS["v6"]["experiment"], source=provenance,
                    tune_files=["tune.jpeg"], audit_files=["audit.jpeg"], target_rasterization=v10.TARGET,
                    smoke_limit=0, refiner_checkpoint_sha256=v10.REFINER_SHA256,
                    detector_confidence_floor=0.1, resolutions=[1024], mask_match_iou=0.5)
    return provenance, metadata


@pytest.mark.parametrize("field", ["smoke_limit", "refiner_checkpoint_sha256", "target_rasterization",
                                   "tune_files", "detector_confidence_floor", "source"])
def test_source_manifest_guard_rejects_changed_settings(field):
    provenance, metadata = source_fixture()
    v10.verify_manifest(metadata, "v6", provenance, {"tune.jpeg"}, {"audit.jpeg"})
    metadata[field] = "changed"
    with pytest.raises(ValueError):
        v10.verify_manifest(metadata, "v6", provenance, {"tune.jpeg"}, {"audit.jpeg"})


def test_split_guard_rejects_leakage_even_when_manifest_agrees():
    provenance, metadata = source_fixture()
    provenance["training_files"].append("tune.jpeg")
    with pytest.raises(ValueError, match="Leaking"):
        v10.verify_manifest(metadata, "v6", provenance, {"tune.jpeg"}, {"audit.jpeg"})


def test_cache_id_is_original_kaggle_path_not_current_mount():
    import hashlib
    expected = hashlib.sha256((v10.ORIGINAL_ROOT + "/train/train_images/example.jpeg").encode()).hexdigest()[:16]
    assert v10.raw_cache_id("example.jpeg", "train") == expected + ".json.gz"
    assert v10.raw_cache_id("example.jpeg", "test") != expected + ".json.gz"
    with pytest.raises(ValueError):
        v10.raw_cache_id("../example.jpeg", "train")


def test_complete_raw_cache_preflight_rejects_missing_changed_or_unpinned_sources(tmp_path, monkeypatch):
    provenance, metadata = source_fixture()
    specs = [("train", "tune.jpeg"), ("train", "audit.jpeg"), ("test", "test.jpeg")]
    inventories = {}
    pins = {name: dict(pin) for name, pin in v10.SOURCE_PINS.items()}
    for detector in pins:
        directory = tmp_path / detector
        directory.mkdir()
        m = dict(metadata, experiment=pins[detector]["experiment"])
        if detector == "v8":
            m["detector"] = dict(sha256=pins[detector]["detector"])
        manifest = directory / "inference_manifest.json"
        manifest.write_text(json.dumps(m))
        pins[detector]["manifest"] = v10.sha256(manifest)
        cache_root = directory / "detector_proposal_cache"
        if detector == "v8":
            cache_root /= pins[detector]["detector"][:16]
        cache_root.mkdir(parents=True)
        inventories[detector] = {}
        for split, name in specs:
            cache = cache_root / v10.raw_cache_id(name, split)
            with gzip.open(cache, "wt") as stream:
                json.dump([dict(rles=[], scores=[])] * 4, stream)
            inventories[detector][cache.name] = v10.sha256(cache)
    monkeypatch.setattr(v10, "SOURCE_PINS", pins)
    roots, inventory = v10.resolve_cache_sources(tmp_path, provenance, {"tune.jpeg"}, {"audit.jpeg"}, specs, inventories)
    assert len(inventory["v6"]) == 3 and roots["v8"].name == "v8"
    changed = inventory["v8"][("test", "test.jpeg")]
    changed.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Missing/changed v8"):
        v10.resolve_cache_sources(tmp_path, provenance, {"tune.jpeg"}, {"audit.jpeg"}, specs, inventories)
    with pytest.raises(ValueError, match="pin complete"):
        v10.resolve_cache_sources(tmp_path, provenance, {"tune.jpeg"}, {"audit.jpeg"}, specs, None)


def test_canonical_control_count_guard_and_submission_sha(tmp_path, monkeypatch):
    row = dict(candidate=v10.CONTROL, file_name="a.jpeg", image_id="ann1", group="tune", target="coco",
               iou_sum=0.75, tp=1, fp=0, fn=1)
    path = tmp_path / "baseline.csv"
    pd.DataFrame([dict(row, candidate=v10.OLD_CONTROL)]).to_csv(path, index=False)
    v10.verify_control_counts([row], path, ["a.jpeg"])
    with pytest.raises(AssertionError):
        v10.verify_control_counts([dict(row, iou_sum=0.750000001)], path, ["a.jpeg"])
    with pytest.raises(ValueError):
        v10.verify_control_counts([row, row], path, ["a.jpeg"])
    export = tmp_path / "submission_control.csv"
    export.write_bytes(b"filament_id,segmentation_rle\n")
    monkeypatch.setattr(v10, "CONTROL_SHA256", v10.sha256(export))
    v10.verify_control_export(export)
    export.write_bytes(b"different")
    with pytest.raises(ValueError, match="hash mismatch"):
        v10.verify_control_export(export)


def test_refinement_keeps_all_proposals_and_original_batch_order(tmp_path, monkeypatch):
    groups = []
    masks = []
    for index in range(11):
        mask = rectangle(index + 1, index + 3, 1, 30, (2048, 2048))
        masks.append(mask)
        groups.append(dict(rle=encoded(mask), consensus_confidence=0.9 - index * 0.07,
                           support_count=4, views=[0, 1, 2, 3], view_scores={}))
    raw = tmp_path / "raw.json.gz"
    raw.write_bytes(b"raw identity")
    inventory = {d: {("train", "a.jpeg"): raw} for d in ("v6", "v8")}
    monkeypatch.setattr(v10, "read_raw_views", lambda p: "views")
    monkeypatch.setattr(v10, "group_views", lambda v: groups)
    monkeypatch.setattr(v10.tta, "cluster_views", lambda *a: ([g["rle"] for g in groups],
                                                             [g["consensus_confidence"] for g in groups]))
    monkeypatch.setattr(v10.cv2, "imread", lambda *a: np.zeros((2048, 2048), np.uint8))
    observed = []
    def crop_probabilities(base, refiner, gray, incoming, tta):
        observed.append(len(incoming))
        assert tta is True and len(incoming) == 11  # Includes candidates below both final cutoffs.
        for a, b in zip(incoming, masks):
            np.testing.assert_array_equal(a, b)
        return [(0, 0, 32, 32)] * len(incoming), [], [np.ones((16, 16), np.float32)] * len(incoming)
    monkeypatch.setattr(v10.core, "crop_probabilities", crop_probabilities, raising=False)
    result = v10.prepare_records(None, None, Path("a.jpeg"), "train", inventory, tmp_path, {"script": "frozen"})
    assert observed == [11, 11]
    assert len(result["v6"]) == len(result["v8"]) == 11
    second = v10.prepare_records(None, None, Path("a.jpeg"), "train", inventory, tmp_path, {"script": "frozen"})
    assert observed == [11, 11]  # Pure cache replay, no probability quantization.
    assert result == second
    with pytest.raises(ValueError, match="Stale"):
        v10.prepare_records(None, None, Path("a.jpeg"), "train", inventory, tmp_path, {"script": "changed"})


def test_paired_comparison_matches_existing_helper_and_keeps_annotators_together():
    rows = []
    for name, score in (("a.jpeg", 0.6), ("b.jpeg", 0.8)):
        for image_id in (1, 2):
            for candidate, offset in ((v10.CONTROL, 0), ("challenger", 0.03)):
                rows.append(dict(candidate=candidate, file_name=name, image_id=image_id, target="coco", group="tune",
                                 iou_sum=score + offset, tp=1, fp=0, fn=1))
    analysis = cpu_module("analysis_test_module", ROOT / "scripts/analyze_inference_sweep.py")
    expected = analysis.compare_group(pd.DataFrame(rows), "challenger", "tune", samples=100,
                                      control_candidate=v10.CONTROL)
    actual = v10.paired_comparison(rows, "challenger", "tune", samples=100)
    assert {key: actual[key] for key in expected} == expected
    assert actual["images"] == 2


def test_canonical_evaluation_keeps_all_annotators_and_v8_control_separate(tmp_path, monkeypatch):
    name = "a.jpeg"
    base = SimpleNamespace(CFG=SimpleNamespace(train_dir=tmp_path))
    images = {key: dict(file_name=name, width=2048, height=2048) for key in ("ann1", "ann2")}
    annotations = {"ann1": [dict(segmentation=[[2, 5, 12, 5, 12, 15, 2, 15]])],
                   "ann2": [dict(segmentation=[[22, 5, 32, 5, 32, 15, 22, 15]])]}
    records = {"v6": [record(rectangle())],
               "v8": [record(rectangle(22, 32), score=0.3, detector="v8")]}
    monkeypatch.setattr(v10, "prepare_records", lambda *a: records)
    split = ([name], {name}, set(), {name: "ann1"}, {name: ["ann1", "ann2"]})
    totals, diagnostics, source_diagnostics = {}, [], []
    config = {v10.CONTROL: v10.declared_candidates()[v10.CONTROL]}
    v10.evaluate_files(base, None, [name], config, images, annotations, split, {}, tmp_path,
                        {}, totals, diagnostics, source_diagnostics)
    canonical = [r for r in diagnostics if r["target"] == "coco"]
    assert len(canonical) == 2
    assert canonical[0]["tp"] == 1 and canonical[1]["fn"] == 1
    assert canonical[0]["rescue_tp"] == canonical[0]["rescue_fp"] == 0
    assert len(source_diagnostics) == 2
    assert source_diagnostics[0]["fp"] == 1 and source_diagnostics[1]["tp"] == 1
    assert v10.V8_CONTROL not in totals  # Never selectable.
    expected = pd.DataFrame([dict(row, candidate=v10.V8_OLD_CONTROL) for row in source_diagnostics])
    path = tmp_path / "v8_baseline.csv"
    expected.to_csv(path, index=False)
    v10.verify_control_counts(source_diagnostics, path, [name], v10.V8_CONTROL, v10.V8_OLD_CONTROL)


def test_unmeasured_audit_metrics_are_missing_not_partial_or_zero(tmp_path):
    configs = v10.isolated_candidates(v10.declared_candidates())
    totals = {key: {group: np.asarray([0.7, 1, 0, 1], dtype=float)
                    for group in ("tune", "audit", "all", "legacy", "canonical_best")}
              for key in configs}
    v10.write_v10_results(totals, configs, [], tmp_path, [v10.CONTROL])
    frame = pd.read_csv(tmp_path / "inference_candidates.csv")
    assert frame.loc[frame.candidate != v10.CONTROL, "audit_pq"].isna().all()
    assert frame.loc[frame.candidate != v10.CONTROL, "all_pq"].isna().all()
    assert frame.loc[frame.candidate == v10.CONTROL, "audit_pq"].notna().all()
    assert frame.tune_pq.notna().all()


def test_export_uses_shared_builder_for_all_180_test_images(tmp_path, monkeypatch):
    test_dir = tmp_path / "test_images"
    test_dir.mkdir()
    for index in range(180):
        (test_dir / f"image{index:03d}.jpeg").touch()
    base = SimpleNamespace(CFG=SimpleNamespace(test_dir=test_dir))
    mask = rectangle()
    records = {"v6": [record(mask)], "v8": []}
    monkeypatch.setattr(v10, "prepare_records", lambda *a: records)
    config = v10.declared_candidates()[v10.CONTROL]
    result = v10.export_config(base, None, config, {}, tmp_path, {}, "submission_control.csv")
    frame = pd.read_csv(tmp_path / "submission_control.csv")
    assert result["rows"] == 180
    assert frame.filament_id.tolist() == [f"image{i:03d}_1" for i in range(180)]
    expected = core_ns["encode_crop"](mask.astype(bool), [0, 0, 48, 48], (2048, 2048))["counts"].decode()
    assert set(frame.segmentation_rle) == {expected}


def test_stager_strips_only_main_guards_and_preserves_module_isolation():
    stager = cpu_module("v10_stager", ROOT / "scripts/prepare_v10_quality.py")
    source = "from __future__ import annotations\nx = 1\nif __name__ == '__main__':\n    raise RuntimeError()\n"
    result = stager.without_main(source)
    assert "__future__" in result and "raise RuntimeError" not in result
    with pytest.raises(ValueError):
        stager.without_main("x = 1\n")


def test_stage_real_pinned_sources_without_gpu_or_launch(tmp_path):
    if not (ROOT / "outputs/kaggle_detector_tta/detector_proposal_cache").is_dir():
        pytest.skip("Downloaded raw-cache artifacts unavailable")
    stager = cpu_module("v10_stage_real", ROOT / "scripts/prepare_v10_quality.py")
    output = stager.stage(ROOT, tmp_path / "staged", limit=3)
    metadata = json.loads((output / "kernel-metadata.json").read_text())
    assert metadata["id"] == "abhimanyu122/solar-filament-v10-quality-rescue"
    assert metadata["id"].split("/")[-1] == metadata["title"].lower().replace(" ", "-")
    assert metadata["is_private"] == "true"
    assert len(metadata["kernel_sources"]) == 3
    tree = ast.parse((output / "v10_quality.py").read_text())
    values = {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body
              if isinstance(node, ast.Assign)}
    assert set(values["SOURCES"]) == {"optimize_inference", "detector_tta", "v10_quality"}
    assert all(len(caches) == 321 for caches in values["RAW_CACHE_SHA256"].values())
    for source in values["SOURCES"].values():
        compile(source, "bundled_module", "exec")
    assert "run_v10()" in (output / "v10_quality.py").read_text()
    assert "sys.argv.extend(['--limit', '3'])" in (output / "v10_quality.py").read_text()
    assert json.loads((output / "staging_manifest.json").read_text())["smoke_limit"] == 3
    with pytest.raises(ValueError):
        stager.stage(ROOT, tmp_path / "invalid", limit=-1)


def test_real_raw_cache_coverage_geometry_and_grouping():
    test_dir = ROOT / "data/raw/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test/test_images"
    if not test_dir.is_dir():
        pytest.skip("Local competition artifacts unavailable")
    test_names = sorted(path.name for path in test_dir.glob("*.jpeg"))
    for detector, directory in (("v6", "kaggle_detector_tta"), ("v8", "kaggle_v8_coco_tta")):
        source = ROOT / "outputs" / directory
        metadata = json.loads((source / "inference_manifest.json").read_text())
        v10.verify_manifest(metadata, detector, metadata["source"], metadata["tune_files"], metadata["audit_files"])
        cache_root = source / "detector_proposal_cache"
        if detector == "v8":
            cache_root /= v10.SOURCE_PINS[detector]["detector"][:16]
        specs = [("train", name) for name in metadata["source"]["validation_files"]]
        specs += [("test", name) for name in test_names]
        assert len(specs) == 321
        assert set(p.name for p in cache_root.glob("*.json.gz")) == {v10.raw_cache_id(n, s) for s, n in specs}
        for index, (split, name) in enumerate(specs):
            views = v10.read_raw_views(cache_root / v10.raw_cache_id(name, split))
            if index % 32 == 0:
                groups = v10.group_views(views)
                expected_masks, expected_scores = tta_ns["cluster_views"](views, "flip_consensus")
                assert [g["rle"] for g in groups] == expected_masks
                assert [g["consensus_confidence"] for g in groups] == expected_scores
