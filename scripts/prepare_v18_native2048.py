"""Stage the pinned V18 native-2048 detector retrain; never launches or submits.

Bundles the module sources inline with SHA-256 self-checks and pins the warm-start
checkpoint, the canonical refiner, the annotations, and the frozen V6 cache
manifest, so the notebook cannot silently train from different weights or score
against different targets. Adapted from prepare_v16_maskratio.py: one arm instead
of two, resolution is now also a varying axis (control stays at 1024, arm moves
to 2048), matching v18_native2048.py's single-arm design.
"""
import argparse
import hashlib
import json
from pathlib import Path

from prepare_v11_refiner import load_helpers, without_main


MODULES = ("optimize_inference", "detector_tta", "v10_quality", "v18_native2048")
SLUGS = {"full": "solar-filament-v18-native-2048",
         "smoke": "solar-filament-v18-native-2048-smoke",
         "smoke_timing": "solar-filament-v18-native-2048-smoke-timing"}
TITLES = {"full": "Solar Filament V18 Native 2048",
          "smoke": "Solar Filament V18 Native 2048 Smoke",
          "smoke_timing": "Solar Filament V18 Native 2048 Smoke Timing"}
# Measured 2026-09-08/09 via smoke_timing (fraction=0.02 vs 0.2 probes at imgsz=1536,
# batch=2): marginal cost ~0.55s/image, ~106s fixed overhead per training() call.
# Estimated 20-epoch full training: ~10,400s. Budgeted with eval overhead: ~13,500s,
# against 48,626s quota remaining (2026-09-08 ~19:16 IST), refresh 2026-09-12.


def stage(root, output, mode="full", batch=None, train_seed=None):
    root, output = Path(root), Path(output)
    if mode not in SLUGS:
        raise ValueError(f"Unknown staging mode {mode}")
    modules = {name: without_main((root / "kaggle_inference" / f"{name}.py").read_text())
               for name in MODULES}
    for name, source in modules.items():
        compile(source, name + ".py", "exec")
    v10 = load_helpers(root, "v10_quality")
    v18 = load_helpers(root, "v18_native2048")
    effective_batch = batch if batch is not None else v18.GLOBAL_BATCH
    if effective_batch not in (v18.GLOBAL_BATCH, v18.FALLBACK_BATCH):
        raise ValueError("Batch must be the declared global batch or its audited fallback")
    effective_seed = train_seed if train_seed is not None else v18.DEFAULT_TRAIN_SEED

    # Pinned V6 warm-start source and canonical refiner.
    source_root = root / "outputs/v6_inference_source"
    if v18.sha256(source_root / "v6_provenance.json") != v10.PROVENANCE_SHA256:
        raise ValueError("Changed V6 provenance before staging")
    provenance = json.loads((source_root / "v6_provenance.json").read_text())
    for name, digest in provenance["sha256"].items():
        if v18.sha256(source_root / name) != digest:
            raise ValueError(f"Changed V6 artifact before staging: {name}")
    if provenance["sha256"]["yolo_best.pt"] != v18.V6_DETECTOR_SHA256:
        raise ValueError("Warm-start checkpoint is not the pinned V6 detector")

    # Canonical annotations and the split this run must reproduce.
    data_root = root / "data/raw/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"
    annotations = data_root / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    if v18.sha256(annotations) != provenance["annotation_sha256"]:
        raise ValueError("Canonical annotation identity changed")
    coco = json.loads(annotations.read_text())
    images = {image["id"]: image for image in coco["images"]}
    grouped = {}
    for entry in coco["annotations"]:
        grouped.setdefault(entry["image_id"], []).append(entry)
    partitions = v18.dataset_partitions(images, grouped, provenance)

    # Frozen V6 cache manifest: the control must reproduce the accepted pipeline.
    cache_source = root / "outputs/kaggle_detector_tta"
    if v18.sha256(cache_source / "inference_manifest.json") != v10.SOURCE_PINS["v6"]["manifest"]:
        raise ValueError("Changed frozen V6 source manifest")
    if v18.sha256(cache_source / "inference_image_diagnostics.csv") != v10.BASELINE_DIAGNOSTICS_SHA256:
        raise ValueError("Changed frozen V6 diagnostics")

    hashes = {name: hashlib.sha256(source.encode()).hexdigest() for name, source in modules.items()}
    bundled = "import sys\nimport types\nimport hashlib\n\n"
    bundled += (f"SOURCES = {modules!r}\nMODULE_SHA256 = {hashes!r}\n"
                f"STAGED_BATCH = {effective_batch!r}\nSTAGED_MODE = {mode!r}\n"
                f"STAGED_TRAIN_SEED = {effective_seed!r}\n\n")
    bundled += (
        "if __name__ == '__main__':\n"
        "    for name, source in SOURCES.items():\n"
        "        if hashlib.sha256(source.encode()).hexdigest() != MODULE_SHA256[name]:\n"
        "            raise ValueError('Bundled source hash mismatch: ' + name)\n"
        "        module = types.ModuleType(name)\n"
        "        module.__file__ = __file__\n"
        "        sys.modules[name] = module\n"
        "        exec(compile(source, name + '.py', 'exec'), module.__dict__)\n"
        "    runner = sys.modules['v18_native2048']\n"
        "    runner.MODULE_SHA256 = MODULE_SHA256\n"
        "    argv = ['v18_native2048', '--batch', str(STAGED_BATCH),\n"
        "            '--train-seed', str(STAGED_TRAIN_SEED)]\n"
        "    if STAGED_MODE in ('smoke', 'smoke_timing'):\n"
        "        argv.append('--smoke')\n"
        "    sys.argv = argv\n"
        "    runner.run_v18()\n")
    compile(bundled, "v18_native2048.py", "exec")
    output.mkdir(parents=True, exist_ok=True)
    (output / "v18_native2048.py").write_text(bundled)

    metadata = json.loads((root / "kaggle_inference/kernel-metadata.json").read_text())
    metadata.update(id=f"abhimanyu122/{SLUGS[mode]}", title=TITLES[mode],
                    code_file="v18_native2048.py", language="python", kernel_type="script",
                    is_private="true", enable_gpu="true", enable_internet="true",
                    machine_shape="NvidiaTeslaT4",
                    docker_image="gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461",
                    docker_image_pinning_type="original",
                    dataset_sources=["abhimanyu122/solar-filament-v6-checkpoints"],
                    competition_sources=["filament-segmentation-2026"],
                    kernel_sources=["abhimanyu122/solar-filament-coco-refiner",
                                    "abhimanyu122/solar-filament-detector-tta"])
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    manifest = dict(experiment="v18_native2048", mode=mode, trains=True,
                    arm=v18.ARM, arm_mask_ratio=v18.ARM_MASK_RATIO,
                    control_mask_ratio=v18.CONTROL_MASK_RATIO,
                    train_resolution=v18.TRAIN_RESOLUTION,
                    control_inference_resolution=v18.CONTROL_INFERENCE_RESOLUTION,
                    arm_inference_resolution=v18.ARM_INFERENCE_RESOLUTION,
                    train_seed=effective_seed,
                    declared_candidates=sorted(v18.declared_candidates()),
                    submit_bar=v18.SUBMIT_BAR, promotion_gate=v18.PROMOTION_GATE,
                    global_batch=effective_batch,
                    training_arguments=v18.training_arguments(
                        v18.TRAIN_RESOLUTION, v18.ARM_MASK_RATIO, effective_batch,
                        train_seed=effective_seed),
                    warm_start_sha256=v18.V6_DETECTOR_SHA256,
                    refiner_sha256=v10.REFINER_SHA256,
                    provenance_sha256=v10.PROVENANCE_SHA256,
                    annotation_sha256=provenance["annotation_sha256"],
                    expected_control_tune=v18.V6_REFERENCE["tune"],
                    expected_control_audit=v18.V6_REFERENCE["audit"],
                    dataset={k: partitions[k] for k in ("train_entries", "train_instances")},
                    training_files=len(partitions["train_files"]),
                    validation_files=len(partitions["valid_files"]),
                    module_sha256=hashes,
                    script_sha256=v18.sha256(output / "v18_native2048.py"),
                    submits=False,
                    note="Detector retrain at imgsz=1536 (train AND inference resolution for "
                         "the arm; control stays at its accepted 1024), mask_ratio=1. Native "
                         "images are 2048x2048, but the smoke grid measured that full imgsz=2048 "
                         "OOMs on a single Kaggle T4 even at batch=1; 1536/batch=2 is the largest "
                         "configuration confirmed to fit (peak ~11.9-12.5GB of 14.56GB). Tests "
                         "whether training/inferring at a higher-than-1024 resolution -- every "
                         "prior detector run's downsample -- moves the ceiling_analysis 0.8734 "
                         "raw-proposal IoU cap and end-to-end PQ materially, unlike V16/V17's "
                         "small mask_ratio-only gain at 1024. Export/submission is a separate "
                         "pinned step, gated on SUBMIT_BAR and PROMOTION_GATE.")
    (output / "staging_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mode", choices=sorted(SLUGS), default="full")
    parser.add_argument("--batch", type=int, default=None,
                        help="Defaults to the module's declared GLOBAL_BATCH (2).")
    parser.add_argument("--train-seed", type=int, default=None)
    args = parser.parse_args()
    default = root / f"outputs/v18_{args.mode}_kernel"
    print(stage(root, args.output or default, args.mode, args.batch, args.train_seed))


if __name__ == "__main__":
    main()
