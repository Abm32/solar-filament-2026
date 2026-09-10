"""Stage the pinned V14 dual-head experiment; never launches or submits anything.

V14 trains a ResNet34 dual-head (mask + learned-IoU) refiner, warm-started from
V11's completed `proposal_centered_e4` checkpoint, to attack both PQ axes the
ceiling analysis identified: recall (via the learned IoU gate replacing the fixed
0.40 confidence cutoff) and mean matched IoU (via a higher-capacity head with an
FN-weighted Focal-Tversky arm). Design: docs/v14_design_2026-09-07.md. Arithmetic:
docs/ceiling_analysis_2026-09-07.md.

This stager pins the V11 warm-start checkpoint hash, all 321 frozen raw detector
caches, the V6 source artifacts, the canonical refiner, the canonical annotations,
and every bundled module source (v14_dualhead + v14_runtime + their V11/V10/V6
dependencies), so the notebook cannot silently drift onto a different checkpoint,
detector, split, or decision logic than what was tested locally.

V14's frozen control arm reproduces E015's exact tune/audit counts (the same
CONTROL_REFERENCE pinned in v11_refiner/v12_resolution/v13_compose) before any
V14 claim is trusted -- verified here structurally (source constants) and, on
Kaggle, at runtime against the live detector/refiner pipeline.
"""
import argparse
import hashlib
import json
from pathlib import Path

from prepare_v11_refiner import load_helpers, without_main
from prepare_v11_export import verify_completed_run, RUN_IDENTITY, SOURCE_RUN

MODULES = ("optimize_inference", "detector_tta", "v10_quality", "v11_refiner",
           "v14_dualhead", "v14_runtime")
ARM = "proposal_centered"
SLUGS = {"full": "solar-filament-v14-dualhead", "smoke": "solar-filament-v14-dualhead-smoke"}
TITLES = {"full": "Solar Filament V14 Dualhead", "smoke": "Solar Filament V14 Dualhead Smoke"}
SMOKE_LIMIT_CAP = 8


def stage(root, output, mode="full", limit=0, run_output=None, revision=""):
    root, output = Path(root), Path(output)
    if mode not in SLUGS:
        raise ValueError(f"Unknown staging mode {mode}")
    if revision and not revision.replace("-", "").isalnum():
        raise ValueError("Revision must be alphanumeric with dashes")
    if mode == "smoke" and not 0 < limit <= SMOKE_LIMIT_CAP:
        raise ValueError(f"Smoke mode requires --limit in 1..{SMOKE_LIMIT_CAP}")
    if mode == "full" and limit != 0:
        raise ValueError("Full mode must not set --limit (no smoke shortcuts)")
    run_output = Path(run_output) if run_output else root / "outputs/kaggle_v11_full"
    # v14_runtime.py is a pure library with no CLI entry point (by design: it must
    # stay importable on a CPU-only box for tests), so it has no __main__ guard to
    # strip, unlike every other bundled module.
    modules = {}
    for name in MODULES:
        source = (root / "kaggle_inference" / f"{name}.py").read_text()
        modules[name] = source if name == "v14_runtime" else without_main(source)
    for name, source in modules.items():
        compile(source, name + ".py", "exec")
    v10, v11 = load_helpers(root, "v10_quality"), load_helpers(root, "v11_refiner")
    v14 = load_helpers(root, "v14_dualhead")

    # The V11 parent must be the completed, gate-failed run V13/E-A also compose
    # against -- the same warm-start checkpoint every other V1x notebook pins.
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

    # The decision logic (grid, gates, baselines) must be exactly what the tests pin.
    if v14.V11_RUN_IDENTITY != RUN_IDENTITY or v14.V11_SOURCE_RUN != SOURCE_RUN:
        raise ValueError("V14 module does not reference the pinned V11 parent run")
    if v14.verify_grid() is not True or len(v14.declared_candidates()) != 21:
        raise ValueError("V14 predeclared grid is not the tested 3x7 grid")
    if v14.SUBMIT_BAR["min_tune_pq"] != v14.BASELINES["v13_compose"]["tune_pq"]:
        raise ValueError("SUBMIT_BAR no longer anchored to the measured V13 tune PQ")
    if v14.PROMOTION_GATE["required_tune_delta_vs_v13"] != 0.010:
        raise ValueError("PROMOTION_GATE margin changed from the declared 10x V13 delta")
    if v14.CONTROL_REFERENCE["tune"][1:] != [626, 211, 508]:
        raise ValueError("V14 control reference is not the accepted E015 counts")

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
        "    runner = sys.modules['v14_dualhead']\n"
        "    runner.MODULE_SHA256 = MODULE_SHA256\n"
        "    runner.RAW_CACHE_SHA256 = RAW_CACHE_SHA256\n"
        "    runner.CHECKPOINT_SHA256 = CHECKPOINT_SHA256\n"
        f"    sys.argv = ['v14_dualhead'] + {argv_extra!r}\n"
        "    runner.run_experiment()\n")
    compile(bundled, "v14_dualhead.py", "exec")
    output.mkdir(parents=True, exist_ok=True)
    (output / "v14_dualhead.py").write_text(bundled)

    metadata = json.loads((root / "kaggle_inference/kernel-metadata.json").read_text())
    slug = SLUGS[mode] + (f"-{revision}" if revision else "")
    title = TITLES[mode] + (f" {revision.upper()}" if revision else "")
    metadata.update(id=f"abhimanyu122/{slug}", title=title,
                    code_file="v14_dualhead.py", language="python", kernel_type="script",
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

    manifest = dict(experiment="v14_dualhead_iou_head_x_focal_tversky", mode=mode,
                    revision=revision, trains=True, submits=False, limit=limit,
                    arms=list(v14.ARMS), arm_loss=v14.ARM_LOSS,
                    warm_start_arm=ARM,
                    v11_source_run=SOURCE_RUN, v11_run_identity=RUN_IDENTITY,
                    v11_checkpoint_sha256=checkpoint_sha256,
                    detector_sha256=provenance["sha256"]["yolo_best.pt"],
                    refiner_sha256=v10.REFINER_SHA256,
                    provenance_sha256=v10.PROVENANCE_SHA256,
                    annotation_sha256=provenance["annotation_sha256"],
                    expected_control_sha256=v10.CONTROL_SHA256,
                    cached_images=len(inventories),
                    cache_inventory_sha256=v11.digest_json(inventories),
                    declared_candidates=len(v14.declared_candidates()),
                    submit_bar=v14.SUBMIT_BAR, promotion_gate=v14.PROMOTION_GATE,
                    module_sha256=hashes,
                    script_sha256=v11.sha256(output / "v14_dualhead.py"),
                    note="Trains ResNet34 dual-head (mask + learned IoU) arms warm-started "
                         "from V11 proposal_centered_e4, gated on both axes (recall via the "
                         "learned IoU gate, mean IoU via Focal-Tversky arm_b). Frozen control "
                         "must reproduce the accepted 0.38 counts before any claim. Export is "
                         "gated on SUBMIT_BAR; submission stays manual even if authorised.")
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
    default = root / f"outputs/v14_{args.mode}{suffix}_kernel"
    print(stage(root, args.output or default, args.mode, args.limit, args.run_output,
                args.revision))


if __name__ == "__main__":
    main()
