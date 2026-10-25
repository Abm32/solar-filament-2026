"""V19 Mask2Former: export a Kaggle submission.csv from a trained checkpoint.

Loads the best-tune-PQ checkpoint saved by scripts/train_v19_mask2former.py's
run_full_training, runs inference over every test image, and writes
filament_id/segmentation_rle rows in this project's standard format (same
convention as src/rle.py and scripts/validate_submission.py: COCO compressed
RLE at 2048x2048, non-overlapping instances, minimum mask area 20).

Only ever run this AFTER confirming v19_full_result.json's cleared_submit_bar
is true -- SUBMIT_BAR gates whether a CSV is authorized to be exported at all,
per docs/v19_design_2026-09-09.md's predeclared protocol. This script enforces
that check itself rather than trusting the caller to have looked.

Mask2Former's post_process_instance_segmentation already returns a single
non-overlapping segmentation map (each pixel assigned to at most one segment,
argmax over query logits) -- unlike the YOLO-seg pipeline's V6-V18 scripts,
there is no separate crop/NMS/overlap-resolution step needed here.

Usage:

    python scripts/export_v19_submission.py \\
        --checkpoint outputs/v19_mask2former_azure_full/best_model \\
        --result outputs/v19_mask2former_azure_full/v19_full_result.json \\
        --data-root data/raw \\
        --out outputs/v19_mask2former_azure_full/submission.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

SUBMIT_BAR_TUNE = 0.448703
MIN_MASK_AREA = 20
MASK_THRESHOLD = 0.5
INFER_SIZE = 2048  # native test-image resolution; no train/infer mismatch
# issue here since this is the FINAL export pass, run once, not the per-epoch
# model-selection loop that E029/v19_regression_2026-09-10.md fixed. Unlike
# that loop, a single higher-resolution inference pass at submission time is
# exactly what the original design doc asked for.


def load_and_check_result(result_path: Path) -> dict:
    result = json.loads(result_path.read_text())
    if not result.get("cleared_submit_bar"):
        raise RuntimeError(
            f"v19_full_result.json reports cleared_submit_bar={result.get('cleared_submit_bar')} "
            f"(best_tune_pq={result.get('best_tune_pq')} vs required {SUBMIT_BAR_TUNE}). "
            "SUBMIT_BAR gates export -- refusing to generate a submission CSV."
        )
    return result


def export_submission(checkpoint_dir: Path, test_dir: Path, out_csv: Path,
                       infer_size: int = INFER_SIZE) -> dict:
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
                outputs, threshold=MASK_THRESHOLD, target_sizes=[(infer_size, infer_size)],
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
        "out_csv": str(out_csv),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                         help="Path to the best_model/ directory saved by run_full_training")
    parser.add_argument("--result", type=Path, required=True,
                         help="Path to v19_full_result.json -- checked for cleared_submit_bar")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "raw")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--force", action="store_true",
                         help="Skip the SUBMIT_BAR check (diagnostics only; never use to submit)")
    args = parser.parse_args()

    if args.force:
        print("WARNING: --force set, skipping SUBMIT_BAR check. This CSV must not be submitted "
              "unless the bar is independently confirmed cleared.", file=sys.stderr)
    else:
        result = load_and_check_result(args.result)
        print(f"SUBMIT_BAR cleared: best_tune_pq={result['best_tune_pq']:.6f} "
              f">= {SUBMIT_BAR_TUNE}. Proceeding with export.")

    test_dir = (args.data_root / "filament-segmentation-2026" / "MAGFiLO_1.0_Kaggle_2026"
                / "test" / "test_images")
    summary = export_submission(args.checkpoint, test_dir, args.out)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
