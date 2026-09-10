"""Stage the pinned V17 composition notebook; never launches or submits anything.

V17 feeds V16's `mask_ratio=1` detector (`arm_r1`, both training seeds) into V13's
measured recipe: box `iou` swept over {0.3, 0.7}, V11's `proposal_centered_e4` refiner,
and the frozen 42-candidate threshold/confidence grid per cell.

This stager pins, by SHA-256: both V16 `arm_r1` detector checkpoints, the V11 refiner
checkpoint, all 321 frozen raw detector caches, the V6 source artifacts, the canonical
refiner, the canonical annotations, and every bundled module source. Nothing in the
notebook can drift onto a different detector, refiner, or split.

The frozen `iou=0.7` caches are still pinned even though every arm regenerates its own
proposals: V17's control arm reads them to reproduce the accepted 0.38 submission
byte-for-byte before any composition claim is made.

Unlike V13/V16 this run TRAINS NOTHING, so the detector checkpoints are inputs rather
than outputs and are verified here rather than produced.
"""
import argparse
import hashlib
import json
from pathlib import Path

from prepare_v11_refiner import load_helpers, without_main
from prepare_v11_export import verify_completed_run, RUN_IDENTITY, SOURCE_RUN

MODULES = ("optimize_inference", "detector_tta", "v10_quality", "v11_refiner", "v17_compose")
ARM = "proposal_centered"
SLUGS = {"full": "solar-filament-v17-compose", "smoke": "solar-filament-v17-compose-smoke"}
TITLES = {"full": "Solar Filament V17 Compose", "smoke": "Solar Filament V17 Compose Smoke"}
# Completed V16 runs whose arm_r1 checkpoints V17 consumes.
V16_RUNS = {
    "seed1": dict(run="abhimanyu122/solar-filament-v16-mask-ratio",
                  output="outputs/kaggle_v16_full"),
    "seed2": dict(run="abhimanyu122/solar-filament-v16-mask-ratio-seed2",
                  output="outputs/kaggle_v16_full_seed2"),
}


def verify_v16_arm(root, output_dir, label):
    """The arm_r1 checkpoint must match its own completed run's recorded hash.

    Guards against staging a stale or hand-copied checkpoint: the file's hash is
    checked against `v16_manifest.json`'s `training.arm_r1.checkpoint_sha256`, and the
    manifest is required to describe a mask_ratio=1 arm that actually completed.
    """
    base = Path(root) / output_dir
    manifest_path = base / "v16_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Missing V16 manifest for {label}: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    training = manifest.get("training", {}).get("arm_r1")
    if not training:
        raise ValueError(f"V16 manifest for {label} has no arm_r1 training record")
    if training.get("mask_ratio") != 1:
        raise ValueError(f"{label} arm_r1 is not mask_ratio=1")
    checkpoint = base / "checkpoints" / "arm_r1_best.pt"
    if not checkpoint.is_file():
        raise ValueError(f"Missing arm_r1 checkpoint for {label}: {checkpoint}")
    with checkpoint.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != training["checkpoint_sha256"]:
        raise ValueError(f"{label} arm_r1 checkpoint hash does not match its V16 manifest")
    selection = json.loads((base / "v16_selection.json").read_text())
    return dict(label=label, checkpoint_sha256=digest,
                run_identity=manifest.get("run_identity"),
                selection_run_identity=selection.get("run_identity"),
                epochs_completed=training.get("epochs_completed"),
                elapsed_seconds=training.get("elapsed_seconds"),
                raw_mean_iou=selection["raw_proposal_summaries"]["v16_arm_r1"]["mean"],
                raw_max_iou=selection["raw_proposal_summaries"]["v16_arm_r1"]["max"])


def stage(root, output, mode="full", limit=0, run_output=None, revision=""):
    root, output = Path(root), Path(output)
    if mode not in SLUGS:
        raise ValueError(f"Unknown staging mode {mode}")
    if revision and not revision.replace("-", "").isalnum():
        raise ValueError("Revision must be alphanumeric with dashes")
    if mode == "smoke" and limit <= 0:
        raise ValueError("Smoke mode requires a positive --limit")
    if mode == "full" and limit != 0:
        raise ValueError("Full mode must not set --limit (no smoke shortcuts)")
    run_output = Path(run_output) if run_output else root / "outputs/kaggle_v11_full"
    modules = {name: without_main((root / "kaggle_inference" / f"{name}.py").read_text())
               for name in MODULES}
    for name, source in modules.items():
        compile(source, name + ".py", "exec")
    v10, v11 = load_helpers(root, "v10_quality"), load_helpers(root, "v11_refiner")

    # The V11 refiner parent must be the completed, gate-failed run that scored 0.39.
    verify_completed_run(run_output)
    checkpoint = run_output / "checkpoints" / f"{ARM}_e4.pt"
    checkpoint_sha256 = v11.sha256(checkpoint)

    source_root = root / "outputs/v6_inference_source"
    if v11.sha256(source_root / "v6_provenance.json") != v10.PROVENANCE_SHA256:
        raise ValueError("Changed V6 provenance before staging")
    provenance = json.loads((source_root / "v6_provenance.json").read_text())
    for name, digest in provenance["sha256"].items():
        if v11.sha256(source_root / name) != digest:
            raise ValueError(f"Changed V6 artifact before staging: {name}")
    if provenance["sha256"]["yolo_best.pt"] != v10.SOURCE_PINS["v6"]["detector"]:
        raise ValueError("V6 detector is not the pinned frozen detector")

    cache_source = root / "outputs/kaggle_detector_tta"
    if v11.sha256(cache_source / "inference_manifest.json") != v10.SOURCE_PINS["v6"]["manifest"]:
        raise ValueError("Changed frozen V6 source manifest")
    if v11.sha256(cache_source / "inference_image_diagnostics.csv") != v10.BASELINE_DIAGNOSTICS_SHA256:
        raise ValueError("Changed frozen V6 diagnostics")
    paths = sorted((cache_source / "detector_proposal_cache").glob("*.json.gz"))
    if len(paths) != 321:
        raise ValueError("Download all 321 V6 validation/test raw caches before staging")
    inventories = {path.name: v11.sha256(path) for path in paths}

    data_root = root / "data/raw/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"
    tests = sorted(path.name for path in (data_root / "test/test_images").glob("*.jpeg"))
    expected = {v10.raw_cache_id(name, "train") for name in provenance["validation_files"]}
    expected.update(v10.raw_cache_id(name, "test") for name in tests)
    if len(tests) != 180 or set(inventories) != expected:
        raise ValueError("Raw-cache names do not cover the exact frozen validation/test split")
    annotations = data_root / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    if v11.sha256(annotations) != provenance["annotation_sha256"]:
        raise ValueError("Canonical annotation identity changed")

    # Both V16 arm_r1 detectors, verified against their own completed runs.
    arms = {label: verify_v16_arm(root, spec["output"], label)
            for label, spec in sorted(V16_RUNS.items())}
    arm_detector_sha256 = {label: entry["checkpoint_sha256"] for label, entry in arms.items()}
    if len(set(arm_detector_sha256.values())) != len(arm_detector_sha256):
        raise ValueError("Both seeds resolved to the same checkpoint hash")

    # The declared cells, grid, and bars must be exactly what the tests pin.
    compose = load_helpers(root, "v17_compose")
    if {cell["iou"] for cell in compose.CELLS} != {0.3, 0.7}:
        raise ValueError("V17 cells are not the declared box iou pair")
    if compose.CONTROL_CELL["iou"] != 0.7 or compose.MASK_RATIO != 1:
        raise ValueError("V17 control cell or mask_ratio is not as declared")
    if compose.CONTROL_REFERENCE["tune"][1:] != [626, 211, 508]:
        raise ValueError("V17 control reference is not the accepted E015 counts")
    if len(compose.declared_candidates()) != 84:
        raise ValueError("Expected exactly 84 predeclared candidates")
    if compose.SUBMIT_BAR["min_tune_pq"] != 0.441777:
        raise ValueError("SUBMIT_BAR must be V13's measured tune PQ")
    if compose.ARM_DETECTORS.keys() != arm_detector_sha256.keys():
        raise ValueError("Declared arm detectors do not match the staged V16 runs")
    for label, digest in arm_detector_sha256.items():
        if compose.ARM_DETECTORS[label]["detector_sha256"] != digest:
            raise ValueError(f"Declared {label} detector hash differs from the staged checkpoint")

    hashes = {name: hashlib.sha256(source.encode()).hexdigest() for name, source in modules.items()}
    argv_extra = ["--limit", str(limit)] if mode == "smoke" else []
    bundled = "import sys\nimport types\nimport hashlib\n\n"
    bundled += (f"SOURCES = {modules!r}\nMODULE_SHA256 = {hashes!r}\n"
                f"RAW_CACHE_SHA256 = {inventories!r}\nCHECKPOINT_SHA256 = {checkpoint_sha256!r}\n"
                f"ARM_DETECTOR_SHA256 = {arm_detector_sha256!r}\n\n")
    bundled += (
        "if __name__ == '__main__':\n"
        "    for name, source in SOURCES.items():\n"
        "        if hashlib.sha256(source.encode()).hexdigest() != MODULE_SHA256[name]:\n"
        "            raise ValueError('Bundled source hash mismatch: ' + name)\n"
        "        module = types.ModuleType(name)\n"
        "        module.__file__ = __file__\n"
        "        sys.modules[name] = module\n"
        "        exec(compile(source, name + '.py', 'exec'), module.__dict__)\n"
        "    runner = sys.modules['v17_compose']\n"
        "    runner.MODULE_SHA256 = MODULE_SHA256\n"
        "    runner.RAW_CACHE_SHA256 = RAW_CACHE_SHA256\n"
        "    runner.CHECKPOINT_SHA256 = CHECKPOINT_SHA256\n"
        "    runner.ARM_DETECTOR_SHA256 = ARM_DETECTOR_SHA256\n"
        f"    sys.argv = ['v17_compose'] + {argv_extra!r}\n"
        "    runner.run_compose()\n")
    compile(bundled, "v17_compose.py", "exec")
    output.mkdir(parents=True, exist_ok=True)
    (output / "v17_compose.py").write_text(bundled)

    metadata = json.loads((root / "kaggle_inference/kernel-metadata.json").read_text())
    slug = SLUGS[mode] + (f"-{revision}" if revision else "")
    title = TITLES[mode] + (f" {revision.upper()}" if revision else "")
    metadata.update(id=f"abhimanyu122/{slug}", title=title,
                    code_file="v17_compose.py", language="python", kernel_type="script",
                    is_private="true", enable_gpu="true", enable_internet="true",
                    machine_shape="NvidiaTeslaT4",
                    docker_image="gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461",
                    docker_image_pinning_type="original",
                    dataset_sources=["abhimanyu122/solar-filament-v6-checkpoints"],
                    competition_sources=["filament-segmentation-2026"],
                    kernel_sources=["abhimanyu122/solar-filament-coco-refiner",
                                    "abhimanyu122/solar-filament-detector-tta",
                                    SOURCE_RUN,
                                    V16_RUNS["seed1"]["run"],
                                    V16_RUNS["seed2"]["run"]])
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    manifest = dict(experiment="v17_mask_ratio1_x_v13_recipe", mode=mode, revision=revision,
                    trains=False, submits=False, limit=limit, arm=ARM,
                    cells=list(compose.CELLS), control_cell=compose.CONTROL_CELL,
                    mask_ratio=compose.MASK_RATIO, primary_seed=compose.PRIMARY_SEED,
                    arm_detectors=arms,
                    v11_source_run=SOURCE_RUN, v11_run_identity=RUN_IDENTITY,
                    v11_checkpoint_sha256=checkpoint_sha256,
                    v16_runs=V16_RUNS,
                    detector_sha256=provenance["sha256"]["yolo_best.pt"],
                    refiner_sha256=v10.REFINER_SHA256,
                    provenance_sha256=v10.PROVENANCE_SHA256,
                    annotation_sha256=provenance["annotation_sha256"],
                    expected_control_sha256=v10.CONTROL_SHA256,
                    cached_images=len(inventories),
                    cache_inventory_sha256=v11.digest_json(inventories),
                    declared_candidates=len(compose.declared_candidates()),
                    submit_bar=compose.SUBMIT_BAR, promotion_gate=compose.PROMOTION_GATE,
                    module_sha256=hashes,
                    script_sha256=v11.sha256(output / "v17_compose.py"),
                    note="Feeds V16's replicated mask_ratio=1 detector (both seeds) into V13's "
                         "recipe. Box iou is a declared 2-cell axis because E026 measured the "
                         "two levers as non-redundant at different iou values. Trains nothing, "
                         "so E026's detector-training noise floor does not apply. Control must "
                         "reproduce the accepted 0.38 CSV first. Export gated on SUBMIT_BAR; "
                         "submission stays manual.")
    (output / "staging_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mode", choices=sorted(SLUGS), default="full")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--run-output", type=Path)
    parser.add_argument("--revision", default="",
                        help="Slug suffix for a corrected re-run; a changed source may "
                             "never be pushed to an already-launched slug.")
    args = parser.parse_args()
    suffix = f"_{args.revision.replace('-', '_')}" if args.revision else ""
    default = root / f"outputs/v17_{args.mode}{suffix}_kernel"
    print(stage(root, args.output or default, args.mode, args.limit, args.run_output,
                args.revision))


if __name__ == "__main__":
    main()
