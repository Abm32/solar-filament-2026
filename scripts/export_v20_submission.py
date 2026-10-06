"""Export the exact selected V20 CondInst test candidate for public measurement.

This candidate failed the predeclared local export gates. Export therefore
requires --measurement-override, which records an explicit user-authorized
leaderboard measurement without changing export_authorized in the evaluation.
The script never submits to Kaggle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

BASE = Path(os.environ.get("V20_BASE", "/mnt/solar-v20-condinst"))
sys.path.insert(0, str(BASE))
import train_v20_condinst as v20  # noqa: E402

EXPECTED_CHECKPOINT_SHA256 = "e18bc02c655f52d7eccf8c16ea6c919cbc2d286f42452746a3500b9bfa797a78"
EXPECTED_CONFIDENCE = 0.50
EXPECTED_TEST_IMAGES = 180


def normalize(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    values = gray[gray > 0]
    if values.size < gray.size // 4:
        raise RuntimeError("invalid solar disk")
    lo, hi = np.percentile(values, (2, 98))
    normalized = np.clip(
        (gray.astype(np.float32) - lo) * 255.0 / max(hi - lo, 1.0), 0, 255,
    ).astype(np.uint8)
    return cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)


def encode(mask: np.ndarray) -> str:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    counts = rle["counts"]
    return counts.decode("ascii") if isinstance(counts, bytes) else str(counts)


def check_override(report: dict, checkpoint: Path, override: bool) -> tuple[float, str]:
    if report.get("selected_checkpoint") != checkpoint.name:
        raise RuntimeError("checkpoint filename differs from the selected V20 candidate")
    actual_sha = v20.sha256(checkpoint)
    if actual_sha != report.get("selected_checkpoint_sha256") or actual_sha != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("checkpoint hash differs from the selected V20 candidate")
    confidence = float(report.get("selected_confidence", -1))
    if confidence != EXPECTED_CONFIDENCE:
        raise RuntimeError("selected confidence differs from the frozen V20 candidate")
    if report.get("export_authorized") is not True and not override:
        raise RuntimeError(
            "V20 failed local export gates. Pass --measurement-override only after explicit "
            "user authorization for one public leaderboard measurement."
        )
    return confidence, actual_sha


def export(checkpoint: Path, evaluation: Path, test_dir: Path, output_csv: Path,
           manifest_path: Path, override: bool) -> dict:
    from detectron2.engine import DefaultPredictor

    report = json.loads(evaluation.read_text())
    confidence, checkpoint_sha = check_override(report, checkpoint, override)
    test_images = sorted(test_dir.glob("*.jpeg"))
    if len(test_images) != EXPECTED_TEST_IMAGES:
        raise RuntimeError(f"expected {EXPECTED_TEST_IMAGES} test images, found {len(test_images)}")

    cfg = v20.build_cfg()
    cfg.defrost()
    cfg.MODEL.WEIGHTS = str(checkpoint)
    cfg.freeze()
    predictor = DefaultPredictor(cfg)

    rows = []
    coverage = []
    areas = []
    started = time.monotonic()
    for image_index, image_path in enumerate(test_images, 1):
        original = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if original is None:
            raise RuntimeError(f"could not read {image_path}")
        if original.shape[:2] != (2048, 2048):
            raise RuntimeError(f"{image_path.name}: unexpected shape {original.shape[:2]}")
        prepared = normalize(original)
        outputs = predictor(prepared)["instances"].to("cpu")
        masks = outputs.pred_masks.numpy() if len(outputs) else np.zeros((0, 2048, 2048), bool)
        scores = outputs.scores.numpy() if len(outputs) else np.zeros(0, np.float32)
        accepted = v20.ownership_masks(masks, scores, confidence)
        for instance_index, mask in enumerate(accepted, 1):
            area = int(mask.sum())
            if area < 20:
                raise RuntimeError("ownership cleanup emitted a sub-minimum mask")
            areas.append(area)
            rows.append({
                "filament_id": f"{image_path.stem}_{instance_index}",
                "segmentation_rle": encode(mask),
            })
        coverage.append({
            "file_name": image_path.name,
            "raw_predictions": int(len(outputs)),
            "accepted_predictions": len(accepted),
        })
        if image_index % 20 == 0 or image_index == len(test_images):
            print(f"exported {image_index}/{len(test_images)} images", flush=True)

    frame = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    manifest = {
        "status": "measurement_override_export",
        "explicitly_not_gate_authorized": report.get("export_authorized") is not True,
        "user_override_recorded": override,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": checkpoint_sha,
        "evaluation_sha256": v20.sha256(evaluation),
        "confidence": confidence,
        "test_images": len(test_images),
        "rows": len(frame),
        "populated_images": sum(item["accepted_predictions"] > 0 for item in coverage),
        "empty_images": [item["file_name"] for item in coverage if item["accepted_predictions"] == 0],
        "area_min": min(areas) if areas else None,
        "area_median": float(np.median(areas)) if areas else None,
        "area_max": max(areas) if areas else None,
        "csv_sha256": v20.sha256(output_csv),
        "seconds": time.monotonic() - started,
        "coverage": coverage,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: value for key, value in manifest.items() if key != "coverage"}, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--measurement-override", action="store_true")
    args = parser.parse_args()
    export(
        args.checkpoint, args.evaluation, args.test_dir, args.output,
        args.manifest, args.measurement_override,
    )


if __name__ == "__main__":
    main()
