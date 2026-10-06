"""V19 Mask2Former: export a Kaggle submission.csv from a trained checkpoint.

Loads the checkpoint canonically evaluated by calibrate_v19_threshold.py,
runs inference over every test image with that artifact's selected threshold and
evaluation resolution, and writes filament_id/segmentation_rle rows in this
project's standard format (COCO compressed RLE at 2048x2048, non-overlapping
instances, minimum mask area 20).

Only run this after canonical tune-only calibration. The calibration artifact
must authorize export, clear both frozen V17 bars, describe the exact expected
held-out populations, and match the supplied checkpoint hash. The exporter uses
the selected threshold and evaluated resolution from that artifact so the test
candidate cannot silently differ from the candidate that passed validation.

Mask2Former's post_process_instance_segmentation returns a single non-overlapping
segmentation map, so no separate overlap-resolution pass is needed.

Usage:

    python scripts/export_v19_submission.py \\
        --checkpoint outputs/v19_mask2former_azure_all_annotators/best_model \\
        --result outputs/v19_mask2former_azure_all_annotators/v19_threshold_calibration.json \\
        --data-root data/raw \\
        --out outputs/v19_mask2former_azure_all_annotators/submission.csv
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

SUBMIT_BAR_TUNE = 0.448703
SUBMIT_BAR_AUDIT = 0.470763
MIN_MASK_AREA = 20
EXPECTED_POPULATIONS = {
    "tune": {"files": 94, "annotator_records": 150, "gt_instances": 1134},
    "audit": {"files": 47, "annotator_records": 67, "gt_instances": 459},
}


def checkpoint_sha256(checkpoint_dir: Path) -> str:
    model_path = checkpoint_dir / "model.safetensors"
    if not model_path.is_file():
        raise RuntimeError(f"Missing checkpoint weights: {model_path}")
    digest = hashlib.sha256()
    with model_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_and_check_result(result_path: Path, checkpoint_dir: Path | None = None) -> dict:
    result = json.loads(result_path.read_text())
    tune_pq = float(result.get("best_tune_pq", float("-inf")))
    audit_pq = float(result.get("audit_pq", float("-inf")))
    threshold = float(result.get("selected_threshold", float("nan")))
    infer_size = int(result.get("eval_resolution", 0))
    authorized = (
        result.get("export_authorized") is True
        and result.get("cleared_submit_bar") is True
        and tune_pq >= SUBMIT_BAR_TUNE
        and audit_pq >= SUBMIT_BAR_AUDIT
        and 0.0 <= threshold <= 1.0
        and infer_size > 0
        and result.get("populations") == EXPECTED_POPULATIONS
    )
    if not authorized:
        raise RuntimeError(
            f"V19 calibration is not export-authorized: "
            f"export_authorized={result.get('export_authorized')}, "
            f"cleared_submit_bar={result.get('cleared_submit_bar')}, "
            f"tune={tune_pq} (required {SUBMIT_BAR_TUNE}), "
            f"audit={audit_pq} (required {SUBMIT_BAR_AUDIT}), "
            f"threshold={result.get('selected_threshold')}, "
            f"eval_resolution={result.get('eval_resolution')}, "
            f"populations={result.get('populations')}. "
            "Both canonical bars and candidate-identity checks gate export; refusing CSV."
        )
    if checkpoint_dir is not None:
        expected_sha = result.get("checkpoint_sha256")
        actual_sha = checkpoint_sha256(checkpoint_dir)
        if expected_sha != actual_sha:
            raise RuntimeError(
                f"Checkpoint hash mismatch: calibration={expected_sha}, actual={actual_sha}. "
                "Refusing to export a different model than the calibrated candidate."
            )
    return result


def export_submission(checkpoint_dir: Path, test_dir: Path, out_csv: Path,
                      threshold: float, infer_size: int) -> dict:
    import torch
    from pycocotools import mask as mask_utils
    import cv2
    import numpy as np

    from src.rle import encode_binary_mask

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available -- export must run on the GPU VM")

    from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

    device = torch.device("cuda")
    processor = Mask2FormerImageProcessor(do_resize=True, do_normalize=True, num_labels=1, ignore_index=0)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(str(checkpoint_dir)).to(device)
    model.eval()

    test_images = sorted(test_dir.glob("*.jpeg"))
    if not test_images:
        raise RuntimeError(f"No test images found under {test_dir}")

    rows = []
    populated, empty_images = 0, []
    for image_path in test_images:
        stem = image_path.stem
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"could not read {image_path}")
        height, width = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with torch.no_grad():
            inputs = processor(images=[image_rgb], size={"height": infer_size, "width": infer_size},
                                return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                outputs = model(**inputs)
            result = processor.post_process_instance_segmentation(
                outputs, threshold=threshold, target_sizes=[(infer_size, infer_size)],
            )[0]

        segmentation = result["segmentation"].cpu().numpy()
        if infer_size != height or infer_size != width:
            segmentation = cv2.resize(
                segmentation.astype(np.int32), (width, height), interpolation=cv2.INTER_NEAREST,
            )

        instance_id = 0
        image_had_instance = False
        for segment in result["segments_info"]:
            mask = segmentation == segment["id"]
            area = int(mask.sum())
            if area < MIN_MASK_AREA:
                continue
            if mask.shape != (2048, 2048):
                raise RuntimeError(
                    f"{stem}: expected mask shape (2048, 2048) to match src.rle.MASK_SIZE, "
                    f"got {mask.shape} -- test images must be 2048x2048 native resolution"
                )
            instance_id += 1
            image_had_instance = True
            rows.append({
                "filament_id": f"{stem}_{instance_id}",
                "segmentation_rle": encode_binary_mask(mask),
            })
        if image_had_instance:
            populated += 1
        else:
            empty_images.append(stem)

    import pandas as pd
    frame = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_csv, index=False)

    return {
        "rows": len(frame),
        "test_images": len(test_images),
        "populated_images": populated,
        "empty_images": empty_images,
        "threshold": threshold,
        "infer_size": infer_size,
        "checkpoint_sha256": checkpoint_sha256(checkpoint_dir),
        "out_csv": str(out_csv),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                         help="Path to the best_model/ directory saved by run_full_training")
    parser.add_argument("--result", type=Path, required=True,
                        help="Canonical v19_threshold_calibration.json; checked fail-closed")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "raw")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    result = load_and_check_result(args.result, args.checkpoint)
    threshold = float(result["selected_threshold"])
    infer_size = int(result["eval_resolution"])
    print(f"Canonical V19 export authorized: tune={result['best_tune_pq']:.6f} "
          f">= {SUBMIT_BAR_TUNE}, audit={result['audit_pq']:.6f} "
          f">= {SUBMIT_BAR_AUDIT}, threshold={threshold:.2f}, "
          f"resolution={infer_size}, checkpoint={result['checkpoint_sha256']}." )

    test_dir = (args.data_root / "filament-segmentation-2026" / "MAGFiLO_1.0_Kaggle_2026"
                / "test" / "test_images")
    summary = export_submission(
        args.checkpoint, test_dir, args.out, threshold=threshold, infer_size=infer_size,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
