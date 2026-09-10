"""Stage the pinned V15 mask-head experiment; never launches or submits anything.

V15 retests the mask-capacity hypothesis with an encoder that is ACTUALLY pretrained.
V14 (E025) measured three of its own defects instead: ImageNet init was silently
disabled by the warm start, the IoU head trained on positives only and collapsed to a
constant, and the head diagnostic was misaligned. Design: docs/v15_design_2026-09-07.md.

The critical protocol difference from V14: V15 changes ONLY the refiner and keeps
V13's exact measured selection (E-A iou=0.3 proposals + threshold x confidence grid,
refined_constrained). The learned IoU score is reported as a keep-fraction frontier
and never selects, so V15 structurally cannot repeat V14's FP explosion.

This stager pins the V11 warm-start checkpoint hash, all 321 frozen raw detector
caches, the V6 source artifacts, the canonical refiner, the canonical annotations, and
every bundled module source, so the notebook cannot drift onto different weights, a
different split, or different decision logic than what was tested locally. It also
asserts the three defect fixes are present in the staged source.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

from prepare_v11_refiner import load_helpers, without_main
from prepare_v11_export import verify_completed_run, RUN_IDENTITY, SOURCE_RUN

MODULES = ("optimize_inference", "detector_tta", "v10_quality", "v11_refiner",
           "v15_maskhead", "v15_runtime")
ARM = "proposal_centered"
SLUGS = {"full": "solar-filament-v15-maskhead", "smoke": "solar-filament-v15-maskhead-smoke"}
TITLES = {"full": "Solar Filament V15 Maskhead", "smoke": "Solar Filament V15 Maskhead Smoke"}
SMOKE_LIMIT_CAP = 8


def verify_defect_fixes(root, v15):
    """Assert the three V14 defects cannot recur in the staged source.

    Structural checks, so a future edit that reintroduces a defect fails at staging
    rather than after 3,400 GPU-seconds.
    """
    head = ast.parse((root / "kaggle_inference/v15_maskhead.py").read_text())
    runtime = ast.parse((root / "kaggle_inference/v15_runtime.py").read_text())

    # Defect 1: ImageNet weights must be unconditional.
    builder = next(n for n in head.body
                   if isinstance(n, ast.FunctionDef) and n.name == "build_mask_head")
    calls = [n for n in ast.walk(builder) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "resnet34"]
    if len(calls) != 1:
        raise ValueError("Expected exactly one resnet34 construction")
    weights = next(kw.value for kw in calls[0].keywords if kw.arg == "weights")
    if isinstance(weights, ast.IfExp) or ast.unparse(weights) != "ResNet34_Weights.DEFAULT":
        raise ValueError("Defect 1 regression: ImageNet weights are conditional again")

    # Defect 2: the scoring cohort health check must be enforced, not advisory.
    if v15.MIN_NEGATIVE_FRACTION < 0.20:
        raise ValueError("Scoring-cohort negative floor was weakened below 0.20")
    try:
        v15.verify_scoring_cohort([0.6] * 50)
    except ValueError:
        pass
    else:
        raise ValueError("Defect 2 regression: a positives-only cohort was accepted")
    names = {n.func.attr for n in ast.walk(runtime) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute)}
    if "verify_scoring_cohort" not in names:
        raise ValueError("Runtime never calls verify_scoring_cohort")

    # Defect 3: the head diagnostic must be built from surviving indices and
    # permutation-checked.
    if "verify_head_alignment" not in names:
        raise ValueError("Runtime never verifies head alignment")
    if not any(isinstance(n, ast.FunctionDef) and n.name == "surviving_indices"
               for n in runtime.body):
        raise ValueError("Runtime lacks the index mirror that aligns the diagnostic")
    return True


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
    # v15_runtime.py is a pure library with no CLI entry point (by design: it must
    # stay importable on a CPU-only box for tests), so it has no __main__ to strip.
    modules = {}
    for name in MODULES:
        source = (root / "kaggle_inference" / f"{name}.py").read_text()
        modules[name] = source if name == "v15_runtime" else without_main(source)
    for name, source in modules.items():
        compile(source, name + ".py", "exec")
    v10, v11 = load_helpers(root, "v10_quality"), load_helpers(root, "v11_refiner")
    v15 = load_helpers(root, "v15_maskhead")

    # The V11 parent must be the completed, gate-failed run V13/V14 also compose against.
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

    # The decision logic must be exactly what the tests pin.
    if v15.V11_RUN_IDENTITY != RUN_IDENTITY or v15.V11_SOURCE_RUN != SOURCE_RUN:
        raise ValueError("V15 module does not reference the pinned V11 parent run")
    if v15.verify_grid() is not True or len(v15.declared_candidates()) != 42:
        raise ValueError("V15 predeclared grid is not the tested 6x7 V13 grid")
    if v15.SUBMIT_BAR["min_tune_pq"] != v15.BASELINES["v13_compose"]["tune_pq"]:
        raise ValueError("SUBMIT_BAR no longer anchored to the measured V13 tune PQ")
    if v15.PROMOTION_GATE["required_tune_delta_vs_v13"] != 0.010:
        raise ValueError("PROMOTION_GATE margin changed from the declared +0.010")
    if v15.CONTROL_REFERENCE["tune"][1:] != [626, 211, 508]:
        raise ValueError("V15 control reference is not the accepted E015 counts")
    if v15.ACCEPTED_CONTROL_CSV_SHA256 != v10.CONTROL_SHA256:
        raise ValueError("Control CSV pin differs from the accepted 0.38 submission")
    # Selection must be byte-identical in intent to V13's.
    v13 = load_helpers(root, "v13_compose")
    if v15.THRESHOLDS != v13.THRESHOLDS or v15.CONFIDENCES != v13.CONFIDENCES:
        raise ValueError("V15 selection grid drifted from V13's proven grid")
    if v15.CELL != v13.CELL or v15.MODE != v13.MODE:
        raise ValueError("V15 must reuse V13's exact E-A cell and refinement mode")
    verify_defect_fixes(root, v15)

    hashes = {name: hashlib.sha256(source.encode()).hexdigest()
              for name, source in modules.items()}
    argv_extra = ["--limit", str(limit)] if mode == "smoke" else []
    bundled = "import sys\nimport types\nimport hashlib\n\n"
    bundled += (f"SOURCES = {modules!r}\nMODULE_SHA256 = {hashes!r}\n"
                f"RAW_CACHE_SHA256 = {inventories!r}\n"
                f"CHECKPOINT_SHA256 = {checkpoint_sha256!r}\n\n")
    bundled += (
        "if __name__ == '__main__':\n"
        "    for name, source in SOURCES.items():\n"
        "        if hashlib.sha256(source.encode()).hexdigest() != MODULE_SHA256[name]:\n"
        "            raise ValueError('Bundled source hash mismatch: ' + name)\n"
        "        module = types.ModuleType(name)\n"
        "        module.__file__ = __file__\n"
        "        sys.modules[name] = module\n"
        "        exec(compile(source, name + '.py', 'exec'), module.__dict__)\n"
        "    runner = sys.modules['v15_maskhead']\n"
        "    runner.MODULE_SHA256 = MODULE_SHA256\n"
        "    runner.RAW_CACHE_SHA256 = RAW_CACHE_SHA256\n"
        "    runner.CHECKPOINT_SHA256 = CHECKPOINT_SHA256\n"
        f"    sys.argv = ['v15_maskhead'] + {argv_extra!r}\n"
        "    runner.run_experiment()\n")
    compile(bundled, "v15_maskhead.py", "exec")
    output.mkdir(parents=True, exist_ok=True)
    (output / "v15_maskhead.py").write_text(bundled)

    metadata = json.loads((root / "kaggle_inference/kernel-metadata.json").read_text())
    slug = SLUGS[mode] + (f"-{revision}" if revision else "")
    title = TITLES[mode] + (f" {revision.upper()}" if revision else "")
    metadata.update(id=f"abhimanyu122/{slug}", title=title,
                    code_file="v15_maskhead.py", language="python", kernel_type="script",
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

    manifest = dict(experiment="v15_maskhead_imagenet_init", mode=mode,
                    revision=revision, trains=True, submits=False, limit=limit,
                    arm=v15.ARM, warm_start_arm=ARM,
                    v11_source_run=SOURCE_RUN, v11_run_identity=RUN_IDENTITY,
                    v11_checkpoint_sha256=checkpoint_sha256,
                    detector_sha256=provenance["sha256"]["yolo_best.pt"],
                    refiner_sha256=v10.REFINER_SHA256,
                    provenance_sha256=v10.PROVENANCE_SHA256,
                    annotation_sha256=provenance["annotation_sha256"],
                    expected_control_sha256=v10.CONTROL_SHA256,
                    cached_images=len(inventories),
                    cache_inventory_sha256=v11.digest_json(inventories),
                    declared_candidates=len(v15.declared_candidates()),
                    selection_grid=dict(thresholds=list(v15.THRESHOLDS),
                                        confidences=list(v15.CONFIDENCES)),
                    cell=v15.CELL, mode_name=v15.MODE,
                    submit_bar=v15.SUBMIT_BAR, promotion_gate=v15.PROMOTION_GATE,
                    min_negative_fraction=v15.MIN_NEGATIVE_FRACTION,
                    defect_fixes_verified=True,
                    module_sha256=hashes,
                    script_sha256=v11.sha256(output / "v15_maskhead.py"),
                    note="Trains ONE ImageNet-initialised ResNet34 mask head warm-started "
                         "from V11 proposal_centered_e4, then selects with V13's EXACT "
                         "threshold x confidence grid. The learned IoU score is reported "
                         "as a keep-fraction frontier and never selects, so the V14 FP "
                         "explosion cannot recur. Frozen control must reproduce the "
                         "accepted 0.38 counts AND CSV hash before any claim. Export is "
                         "gated on SUBMIT_BAR; submission stays manual.")
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
    default = root / f"outputs/v15_{args.mode}{suffix}_kernel"
    print(stage(root, args.output or default, args.mode, args.limit, args.run_output,
                args.revision))


if __name__ == "__main__":
    main()
