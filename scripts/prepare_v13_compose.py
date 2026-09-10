"""Stage the pinned V13 composition notebook; never launches or submits anything.

V13 combines E-A's box `iou=0.3` detector flags (E023, gate PASSED) with V11's
`proposal_centered_e4` refiner (E019, public 0.39). This stager pins the V11
checkpoint hash, all 321 frozen raw detector caches, the V6 source artifacts, the
canonical refiner, the canonical annotations, and every bundled module source, so
the notebook cannot drift onto a different checkpoint, detector, or split.

The frozen `iou=0.7` caches are still pinned even though the composition arm
regenerates its own `iou=0.3` proposals: V13's control arm reads them to reproduce
the accepted 0.38 submission byte-for-byte before any composition claim is made.
"""
import argparse
import hashlib
import json
from pathlib import Path

from prepare_v11_refiner import load_helpers, without_main
from prepare_v11_export import verify_completed_run, RUN_IDENTITY, SOURCE_RUN

MODULES = ("optimize_inference", "detector_tta", "v10_quality", "v11_refiner", "v13_compose")
ARM = "proposal_centered"
SLUGS = {"full": "solar-filament-v13-compose", "smoke": "solar-filament-v13-compose-smoke"}
TITLES = {"full": "Solar Filament V13 Compose", "smoke": "Solar Filament V13 Compose Smoke"}


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

    # The V11 parent must be the completed, gate-failed run that scored 0.39.
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

    # The composition cell and both bars must be exactly what the tests pin.
    compose = load_helpers(root, "v13_compose")
    if compose.CELL["iou"] != 0.3 or compose.CONTROL_CELL["iou"] != 0.7:
        raise ValueError("V13 cell/control flags are not the measured E-A pair")
    if compose.CONTROL_REFERENCE["tune"][1:] != [626, 211, 508]:
        raise ValueError("V13 control reference is not the accepted E015 counts")

    hashes = {name: hashlib.sha256(source.encode()).hexdigest() for name, source in modules.items()}
    argv_extra = ["--limit", str(limit)] if mode == "smoke" else []
    bundled = "import sys\nimport types\nimport hashlib\n\n"
    bundled += (f"SOURCES = {modules!r}\nMODULE_SHA256 = {hashes!r}\n"
                f"RAW_CACHE_SHA256 = {inventories!r}\nCHECKPOINT_SHA256 = {checkpoint_sha256!r}\n\n")
    bundled += (
        "if __name__ == '__main__':\n"
        "    for name, source in SOURCES.items():\n"
        "        if hashlib.sha256(source.encode()).hexdigest() != MODULE_SHA256[name]:\n"
        "            raise ValueError('Bundled source hash mismatch: ' + name)\n"
        "        module = types.ModuleType(name)\n"
        "        module.__file__ = __file__\n"
        "        sys.modules[name] = module\n"
        "        exec(compile(source, name + '.py', 'exec'), module.__dict__)\n"
        "    runner = sys.modules['v13_compose']\n"
        "    runner.MODULE_SHA256 = MODULE_SHA256\n"
        "    runner.RAW_CACHE_SHA256 = RAW_CACHE_SHA256\n"
        "    runner.CHECKPOINT_SHA256 = CHECKPOINT_SHA256\n"
        f"    sys.argv = ['v13_compose'] + {argv_extra!r}\n"
        "    runner.run_compose()\n")
    compile(bundled, "v13_compose.py", "exec")
    output.mkdir(parents=True, exist_ok=True)
    (output / "v13_compose.py").write_text(bundled)

    metadata = json.loads((root / "kaggle_inference/kernel-metadata.json").read_text())
    slug = SLUGS[mode] + (f"-{revision}" if revision else "")
    title = TITLES[mode] + (f" {revision.upper()}" if revision else "")
    metadata.update(id=f"abhimanyu122/{slug}", title=title,
                    code_file="v13_compose.py", language="python", kernel_type="script",
                    is_private="true", enable_gpu="true", enable_internet="true",
                    machine_shape="NvidiaTeslaT4",
                    docker_image="gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461",
                    docker_image_pinning_type="original",
                    dataset_sources=["abhimanyu122/solar-filament-v6-checkpoints"],
                    competition_sources=["filament-segmentation-2026"],
                    kernel_sources=["abhimanyu122/solar-filament-coco-refiner",
                                    "abhimanyu122/solar-filament-detector-tta",
                                    SOURCE_RUN])
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    manifest = dict(experiment="v13_ea_flags_x_v11_refiner", mode=mode, revision=revision,
                    trains=False, submits=False, limit=limit, arm=ARM,
                    cell=compose.CELL, control_cell=compose.CONTROL_CELL,
                    v11_source_run=SOURCE_RUN, v11_run_identity=RUN_IDENTITY,
                    v11_checkpoint_sha256=checkpoint_sha256,
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
                    script_sha256=v11.sha256(output / "v13_compose.py"),
                    note="Composes measured E-A iou=0.3 flags with the V11 proposal_centered_e4 "
                         "refiner. Regenerates iou=0.3 proposals because the frozen caches are "
                         "iou=0.7. Control arm must reproduce the accepted 0.38 CSV first. "
                         "Export is gated on SUBMIT_BAR; submission stays manual.")
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
    default = root / f"outputs/v13_{args.mode}{suffix}_kernel"
    print(stage(root, args.output or default, args.mode, args.limit, args.run_output,
                args.revision))


if __name__ == "__main__":
    main()
