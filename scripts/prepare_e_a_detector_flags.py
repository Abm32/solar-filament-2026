"""Stage the pinned E-A detector-flag ablation; never launches or submits anything.

Bundles detector_flags.py with its two dependencies (optimize_inference,
detector_tta) inline with SHA-256 self-checks, matching the pattern in
prepare_v11_refiner.py / prepare_v12_resolution.py. No training happens here:
E-A reuses the frozen V6 detector + COCO refiner checkpoints and only varies
inference-time YOLO predict() flags (retina_masks, agnostic_nms, box iou).

detector_tta.py imports torch at module scope, which this staging environment
does not have installed, so its REFINER_SHA256 constant is read by AST instead
of executed (matching tests/test_detector_flags.py's own extraction pattern).
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

from prepare_v11_refiner import load_helpers, without_main

MODULES = ("optimize_inference", "detector_tta", "detector_flags")
SLUGS = {"full": "solar-filament-ea-detector-flags",
         "smoke": "solar-filament-ea-detector-flags-smoke"}
TITLES = {"full": "Solar Filament EA Detector Flags",
          "smoke": "Solar Filament EA Detector Flags Smoke"}


def constant(root, module_name, const_name):
    """Extract a module-level string constant by AST without importing torch/cv2."""
    path = root / "kaggle_inference" / f"{module_name}.py"
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == const_name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"Constant {const_name} not found in {module_name}.py")


def stage(root, output, mode="full", limit=0, revision=""):
    root, output = Path(root), Path(output)
    if mode not in SLUGS:
        raise ValueError(f"Unknown staging mode {mode}")
    if revision and not revision.replace("-", "").isalnum():
        raise ValueError("Revision must be alphanumeric with dashes")
    if mode == "smoke" and limit <= 0:
        raise ValueError("Smoke mode requires a positive --limit")
    if mode == "full" and limit != 0:
        raise ValueError("Full mode must not set --limit (no smoke shortcuts)")
    modules = {name: without_main((root / "kaggle_inference" / f"{name}.py").read_text())
               for name in MODULES}
    for name, source in modules.items():
        compile(source, name + ".py", "exec")

    v10 = load_helpers(root, "v10_quality")
    source_root = root / "outputs/v6_inference_source"
    if hashlib.sha256((source_root / "v6_provenance.json").read_bytes()).hexdigest() != v10.PROVENANCE_SHA256:
        raise ValueError("Changed V6 provenance before staging")
    provenance = json.loads((source_root / "v6_provenance.json").read_text())
    for name, digest in provenance["sha256"].items():
        if hashlib.sha256((source_root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Changed V6 artifact before staging: {name}")

    refiner_path = root / "outputs/kaggle_coco_refiner/best_refiner_coco.pt"
    refiner_sha256 = constant(root, "detector_tta", "REFINER_SHA256")
    if hashlib.sha256(refiner_path.read_bytes()).hexdigest() != refiner_sha256:
        raise ValueError("Refiner differs from the accepted checkpoint")

    data_root = root / "data/raw/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"
    annotations = data_root / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    if hashlib.sha256(annotations.read_bytes()).hexdigest() != provenance["annotation_sha256"]:
        raise ValueError("Canonical annotation identity changed")

    hashes = {name: hashlib.sha256(source.encode()).hexdigest() for name, source in modules.items()}
    bundled = "import sys\nimport types\nimport hashlib\n\n"
    bundled += f"SOURCES = {modules!r}\nMODULE_SHA256 = {hashes!r}\n\n"
    argv_extra = ["--limit", str(limit)] if mode == "smoke" else []
    bundled += (
        "if __name__ == '__main__':\n"
        "    for name, source in SOURCES.items():\n"
        "        if hashlib.sha256(source.encode()).hexdigest() != MODULE_SHA256[name]:\n"
        "            raise ValueError('Bundled source hash mismatch: ' + name)\n"
        "        module = types.ModuleType(name)\n"
        "        module.__file__ = __file__\n"
        "        sys.modules[name] = module\n"
        "        exec(compile(source, name + '.py', 'exec'), module.__dict__)\n"
        "    runner = sys.modules['detector_flags']\n"
        f"    sys.argv = ['detector_flags'] + {argv_extra!r}\n"
        "    runner.run_experiment()\n")
    compile(bundled, "detector_flags.py", "exec")
    output.mkdir(parents=True, exist_ok=True)
    (output / "detector_flags.py").write_text(bundled)

    metadata = json.loads((root / "kaggle_inference/kernel-metadata.json").read_text())
    slug = SLUGS[mode] + (f"-{revision}" if revision else "")
    title = TITLES[mode] + (f" {revision.upper()}" if revision else "")
    metadata.update(id=f"abhimanyu122/{slug}", title=title,
                    code_file="detector_flags.py", language="python", kernel_type="script",
                    is_private="true", enable_gpu="true", enable_internet="true",
                    machine_shape="NvidiaTeslaT4",
                    docker_image="gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461",
                    docker_image_pinning_type="original",
                    dataset_sources=["abhimanyu122/solar-filament-v6-checkpoints"],
                    competition_sources=["filament-segmentation-2026"],
                    kernel_sources=["abhimanyu122/solar-filament-coco-refiner",
                                    "abhimanyu122/solar-filament-detector-tta"])
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    manifest = dict(experiment="e_a_detector_flags", mode=mode, revision=revision, trains=False,
                    limit=limit,
                    warm_start_sha256=provenance["sha256"]["yolo_best.pt"],
                    refiner_sha256=refiner_sha256,
                    provenance_sha256=v10.PROVENANCE_SHA256,
                    annotation_sha256=provenance["annotation_sha256"],
                    module_sha256=hashes,
                    script_sha256=hashlib.sha256((output / "detector_flags.py").read_bytes()).hexdigest(),
                    submits=False,
                    note="Inference-flag ablation only; no training, no threshold-grid expansion. "
                         "Export/submission is a separate pinned step gated on the declared "
                         "+0.003 tune / nonnegative-audit bar.")
    (output / "staging_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mode", choices=sorted(SLUGS), default="full")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--revision", default="",
                        help="Slug suffix for a corrected re-run; a changed source may "
                             "never be pushed to an already-launched slug.")
    args = parser.parse_args()
    suffix = f"_{args.revision.replace('-', '_')}" if args.revision else ""
    default = root / f"outputs/e_a_{args.mode}{suffix}_kernel"
    print(stage(root, args.output or default, args.mode, args.limit, args.revision))


if __name__ == "__main__":
    main()
