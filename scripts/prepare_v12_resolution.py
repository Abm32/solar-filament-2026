"""Stage the pinned V12 resolution comparison; never launches or submits anything.

Bundles the module sources inline with SHA-256 self-checks and pins the warm-start
checkpoint, the canonical refiner, the annotations, and the frozen V6 cache
manifest, so the notebook cannot silently train from different weights or score
against different targets.
"""
import argparse
import hashlib
import json
from pathlib import Path

from prepare_v11_refiner import load_helpers, without_main


MODULES = ("optimize_inference", "detector_tta", "v10_quality", "v12_resolution")
SLUGS = {"full": "solar-filament-v12-resolution",
         "smoke": "solar-filament-v12-resolution-smoke"}
TITLES = {"full": "Solar Filament V12 Resolution",
          "smoke": "Solar Filament V12 Resolution Smoke"}


def stage(root, output, mode="full", batch=4):
    root, output = Path(root), Path(output)
    if mode not in SLUGS:
        raise ValueError(f"Unknown staging mode {mode}")
    modules = {name: without_main((root / "kaggle_inference" / f"{name}.py").read_text())
               for name in MODULES}
    for name, source in modules.items():
        compile(source, name + ".py", "exec")
    v10 = load_helpers(root, "v10_quality")
    v12 = load_helpers(root, "v12_resolution")
    if batch not in (v12.GLOBAL_BATCH, v12.FALLBACK_BATCH):
        raise ValueError("Batch must be the declared global batch or its audited fallback")

    # Pinned V6 warm-start source and canonical refiner.
    source_root = root / "outputs/v6_inference_source"
    if v12.sha256(source_root / "v6_provenance.json") != v10.PROVENANCE_SHA256:
        raise ValueError("Changed V6 provenance before staging")
    provenance = json.loads((source_root / "v6_provenance.json").read_text())
    for name, digest in provenance["sha256"].items():
        if v12.sha256(source_root / name) != digest:
            raise ValueError(f"Changed V6 artifact before staging: {name}")
    if provenance["sha256"]["yolo_best.pt"] != v12.V6_DETECTOR_SHA256:
        raise ValueError("Warm-start checkpoint is not the pinned V6 detector")

    # Canonical annotations and the split this run must reproduce.
    data_root = root / "data/raw/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"
    annotations = data_root / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    if v12.sha256(annotations) != provenance["annotation_sha256"]:
        raise ValueError("Canonical annotation identity changed")
    coco = json.loads(annotations.read_text())
    images = {image["id"]: image for image in coco["images"]}
    grouped = {}
    for entry in coco["annotations"]:
        grouped.setdefault(entry["image_id"], []).append(entry)
    partitions = v12.dataset_partitions(images, grouped, provenance)

    # Frozen V6 cache manifest: the control must reproduce the accepted pipeline.
    cache_source = root / "outputs/kaggle_detector_tta"
    if v12.sha256(cache_source / "inference_manifest.json") != v10.SOURCE_PINS["v6"]["manifest"]:
        raise ValueError("Changed frozen V6 source manifest")
    if v12.sha256(cache_source / "inference_image_diagnostics.csv") != v10.BASELINE_DIAGNOSTICS_SHA256:
        raise ValueError("Changed frozen V6 diagnostics")

    hashes = {name: hashlib.sha256(source.encode()).hexdigest() for name, source in modules.items()}
    bundled = "import sys\nimport types\nimport hashlib\n\n"
    bundled += (f"SOURCES = {modules!r}\nMODULE_SHA256 = {hashes!r}\n"
                f"STAGED_BATCH = {batch!r}\nSTAGED_MODE = {mode!r}\n\n")
    bundled += (
        "if __name__ == '__main__':\n"
        "    for name, source in SOURCES.items():\n"
        "        if hashlib.sha256(source.encode()).hexdigest() != MODULE_SHA256[name]:\n"
        "            raise ValueError('Bundled source hash mismatch: ' + name)\n"
        "        module = types.ModuleType(name)\n"
        "        module.__file__ = __file__\n"
        "        sys.modules[name] = module\n"
        "        exec(compile(source, name + '.py', 'exec'), module.__dict__)\n"
        "    runner = sys.modules['v12_resolution']\n"
        "    runner.MODULE_SHA256 = MODULE_SHA256\n"
        "    argv = ['v12_resolution', '--batch', str(STAGED_BATCH)]\n"
        "    if STAGED_MODE == 'smoke':\n"
        "        argv.append('--smoke')\n"
        "    sys.argv = argv\n"
        "    runner.run_v12()\n")
    compile(bundled, "v12_resolution.py", "exec")
    output.mkdir(parents=True, exist_ok=True)
    (output / "v12_resolution.py").write_text(bundled)

    metadata = json.loads((root / "kaggle_inference/kernel-metadata.json").read_text())
    metadata.update(id=f"abhimanyu122/{SLUGS[mode]}", title=TITLES[mode],
                    code_file="v12_resolution.py", language="python", kernel_type="script",
                    is_private="true", enable_gpu="true", enable_internet="true",
                    machine_shape="NvidiaTeslaT4",
                    docker_image="gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461",
                    docker_image_pinning_type="original",
                    dataset_sources=["abhimanyu122/solar-filament-v6-checkpoints"],
                    competition_sources=["filament-segmentation-2026"],
                    kernel_sources=["abhimanyu122/solar-filament-coco-refiner",
                                    "abhimanyu122/solar-filament-detector-tta"])
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    manifest = dict(experiment="v12_detector_resolution", mode=mode, trains=True,
                    arms=list(v12.ARMS), arm_resolution=v12.ARM_RESOLUTION,
                    inference_resolutions=list(v12.INFERENCE_RESOLUTIONS),
                    declared_candidates=sorted(v12.declared_candidates()),
                    gate=v12.GATE, global_batch=batch,
                    training_arguments={arm: v12.training_arguments(v12.ARM_RESOLUTION[arm], batch)
                                        for arm in v12.ARMS},
                    warm_start_sha256=v12.V6_DETECTOR_SHA256,
                    refiner_sha256=v10.REFINER_SHA256,
                    provenance_sha256=v10.PROVENANCE_SHA256,
                    annotation_sha256=provenance["annotation_sha256"],
                    expected_control_tune=v12.V6_REFERENCE["tune"],
                    expected_control_audit=v12.V6_REFERENCE["audit"],
                    dataset={k: partitions[k] for k in ("train_entries", "train_instances")},
                    training_files=len(partitions["train_files"]),
                    validation_files=len(partitions["valid_files"]),
                    module_sha256=hashes,
                    script_sha256=v12.sha256(output / "v12_resolution.py"),
                    submits=False,
                    note="Detector-resolution comparison only; postprocessing frozen. "
                         "Export/submission is a separate pinned step.")
    (output / "staging_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    v12.verify_matched_budget(manifest["training_arguments"])
    return output


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mode", choices=sorted(SLUGS), default="full")
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()
    default = root / f"outputs/v12_{args.mode}_kernel"
    print(stage(root, args.output or default, args.mode, args.batch))


if __name__ == "__main__":
    main()
