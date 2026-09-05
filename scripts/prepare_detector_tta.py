"""Bundle the shared evaluator and detector-TTA ablation into one Kaggle script."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v8", action="store_true", help="Use V8 detector with accepted COCO refiner")
    args = parser.parse_args()
    core = (root / "kaggle_inference/optimize_inference.py").read_text()
    guard = '\nif __name__ == "__main__":\n    main()\n'
    if core.count(guard) != 1:
        raise ValueError("Cannot locate shared evaluator entry point")
    experiment = (root / "kaggle_inference/detector_tta.py").read_text()
    replacement = "import optimize_inference as core"
    if experiment.count(replacement) != 1:
        raise ValueError("Cannot locate shared evaluator import")
    bundled = core.replace(guard, "\n") + "\n" + experiment.replace(
        replacement, "core = sys.modules[__name__]")
    if args.v8:
        checkpoint = root / "outputs/kaggle_kernel_v8/yolo_best.pt"
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            raise ValueError("Download the complete V8 detector checkpoint before staging")
        settings = {"DEFAULT_DETECTOR_VERSION = 6": "DEFAULT_DETECTOR_VERSION = 8",
                    "EXPECTED_V8_DETECTOR_SHA256 = None":
                    f"EXPECTED_V8_DETECTOR_SHA256 = {hashlib.sha256(checkpoint.read_bytes()).hexdigest()!r}"}
        for original, replacement in settings.items():
            if bundled.count(original) != 1:
                raise ValueError(f"Cannot locate experiment setting: {original}")
            bundled = bundled.replace(original, replacement)
    compile(bundled, "detector_tta.py", "exec")
    output = root / ("outputs/v8_coco_tta_kernel" if args.v8 else "outputs/detector_tta_kernel")
    output.mkdir(parents=True, exist_ok=True)
    (output / "detector_tta.py").write_text(bundled)
    metadata = json.loads((root / "kaggle_inference/kernel-metadata.json").read_text())
    metadata.update(id="abhimanyu122/solar-filament-detector-tta",
                    title="Solar Filament Detector TTA", code_file="detector_tta.py",
                    kernel_sources=["abhimanyu122/solar-filament-coco-refiner"])
    if args.v8:
        metadata.update(id="abhimanyu122/solar-filament-v8-coco-tta",
                        title="Solar Filament V8 COCO TTA")
        metadata["kernel_sources"].append("abhimanyu122/solar-filament-yolo-refiner")
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
