"""Calibrate V19 Mask2Former confidence on the canonical validation population.

The original V19 runs evaluated one arbitrary annotator record per held-out file,
so their tune population contained 728 GT instances instead of the 1,134 used by
V11-V18. This command predicts each file once, scores that prediction against all
annotator records for the file, selects a confidence threshold on tune only, and
then evaluates only that selected threshold on audit. It never trains or exports a
submission CSV.

Usage:
    python scripts/calibrate_v19_threshold.py \
      --checkpoint outputs/v19_mask2former_azure_full/best_model \
      --out outputs/v19_mask2former_azure_full/v19_threshold_calibration.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.train_v19_mask2former import (  # noqa: E402
    IMG_SIZE_TRAIN,
    PQ_IOU_THRESHOLD,
    PROMOTION_GATE_TUNE,
    SUBMIT_BAR_AUDIT,
    SUBMIT_BAR_TUNE,
    load_split,
)
from src.coco import index_coco, load_coco  # noqa: E402

# The fixed 0.50 result had too many false positives, so this bounded grid tests
# only the evidence-backed direction (stricter confidence), in 0.05 increments.
THRESHOLDS = tuple(round(0.50 + 0.05 * i, 2) for i in range(10))
EXPECTED_POPULATION = {
    "tune": {"files": 94, "annotator_records": 150, "gt_instances": 1134},
    "audit": {"files": 47, "annotator_records": 67, "gt_instances": 459},
}


def build_canonical_records(coco_path: Path, image_dir: Path, file_names: list[str]) -> list[dict]:
    """One inference record per file, retaining every annotator target."""
    coco = load_coco(coco_path)
    images, annotations = index_coco(coco)
    by_file: dict[str, list[int]] = {}
    for image_id, info in images.items():
        by_file.setdefault(info["file_name"], []).append(image_id)

    records = []
    for file_name in file_names:
        image_ids = sorted(by_file.get(file_name, []))
        if not image_ids:
            raise ValueError(f"No COCO image records for {file_name}")
        records.append({
            "file_name": file_name,
            "image_path": str(image_dir / file_name),
            "targets": [
                {"image_id": image_id, "annotations": annotations.get(image_id, [])}
                for image_id in image_ids
            ],
        })
    return records


def population_summary(records: list[dict]) -> dict:
    return {
        "files": len(records),
        "annotator_records": sum(len(record["targets"]) for record in records),
        "gt_instances": sum(
            len(target["annotations"])
            for record in records
            for target in record["targets"]
        ),
    }


def assert_population(records: list[dict], group: str) -> dict:
    actual = population_summary(records)
    expected = EXPECTED_POPULATION[group]
    if actual != expected:
        raise ValueError(f"Canonical {group} population mismatch: {actual} != {expected}")
    return actual


def pq_value(total: list[float]) -> float:
    tp_iou, tp, fp, fn = total
    denominator = tp + 0.5 * (fp + fn)
    return float(tp_iou / denominator) if denominator else 0.0


def add_pq_counts(total: list[float], pred_masks: list, gt_masks: list) -> None:
    """Accumulate project-standard PQ counts for one annotator target."""
    import numpy as np
    from pycocotools import mask as mask_utils

    n_gt, n_pred = len(gt_masks), len(pred_masks)
    if not n_gt or not n_pred:
        total[2] += n_pred
        total[3] += n_gt
        return
    gt_rles = mask_utils.encode(np.asfortranarray(np.stack(gt_masks, axis=-1).astype(np.uint8)))
    pred_rles = mask_utils.encode(
        np.asfortranarray(np.stack(pred_masks, axis=-1).astype(np.uint8))
    )
    iou = np.asarray(mask_utils.iou(pred_rles, gt_rles, [0] * n_gt)).T
    hits = iou > PQ_IOU_THRESHOLD
    total[0] += float(iou[hits].sum())
    total[1] += int(hits.sum())
    total[2] += int((~hits.any(axis=0)).sum())
    total[3] += int((~hits.any(axis=1)).sum())


def select_tune_threshold(results: list[dict]) -> dict:
    if not results:
        raise ValueError("No tune threshold results")
    # Prefer the stricter threshold on an exact PQ tie to avoid gratuitous FPs.
    return max(results, key=lambda row: (row["pq"], row["threshold"]))


def evaluate_thresholds(model, processor, records: list[dict], device, infer_size: int,
                        thresholds: tuple[float, ...]) -> list[dict]:
    """Predict each file once, then post-process/score each requested threshold."""
    import cv2
    import numpy as np
    import torch
    from pycocotools import mask as mask_utils

    totals = {threshold: [0.0, 0, 0, 0] for threshold in thresholds}
    model.eval()
    for index, record in enumerate(records, start=1):
        image_bgr = cv2.imread(record["image_path"])
        if image_bgr is None:
            raise RuntimeError(f"Could not read {record['image_path']}")
        height, width = image_bgr.shape[:2]
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        target_masks = []
        for target in record["targets"]:
            masks = []
            for ann in target["annotations"]:
                rle = mask_utils.merge(mask_utils.frPyObjects(ann["segmentation"], height, width))
                masks.append(mask_utils.decode(rle).astype(bool))
            target_masks.append(masks)

        with torch.no_grad():
            inputs = processor(
                images=[image], size={"height": infer_size, "width": infer_size},
                return_tensors="pt",
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                outputs = model(**inputs)

        for threshold in thresholds:
            result = processor.post_process_instance_segmentation(
                outputs, threshold=threshold, target_sizes=[(infer_size, infer_size)],
            )[0]
            segmentation = result["segmentation"].cpu().numpy()
            pred_masks = [segmentation == segment["id"] for segment in result["segments_info"]]
            if infer_size != height or infer_size != width:
                pred_masks = [
                    cv2.resize(mask.astype(np.uint8), (width, height),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
                    for mask in pred_masks
                ]
            for gt_masks in target_masks:
                add_pq_counts(totals[threshold], pred_masks, gt_masks)

        if index % 10 == 0 or index == len(records):
            print(f"evaluated {index}/{len(records)} files", flush=True)

    return [
        {
            "threshold": threshold,
            "pq": pq_value(total),
            "tp_iou": total[0],
            "tp": int(total[1]),
            "fp": int(total[2]),
            "fn": int(total[3]),
        }
        for threshold, total in totals.items()
    ]


def checkpoint_sha256(checkpoint_dir: Path) -> str:
    return hashlib.sha256((checkpoint_dir / "model.safetensors").read_bytes()).hexdigest()


def run_calibration(checkpoint_dir: Path, coco_path: Path, image_dir: Path,
                    out_path: Path, infer_size: int = IMG_SIZE_TRAIN) -> dict:
    import torch
    from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available -- calibration must run on a GPU")

    split = load_split(coco_path)
    tune_records = build_canonical_records(coco_path, image_dir, split.tune_files)
    audit_records = build_canonical_records(coco_path, image_dir, split.audit_files)
    populations = {
        "tune": assert_population(tune_records, "tune"),
        "audit": assert_population(audit_records, "audit"),
    }

    device = torch.device("cuda")
    processor = Mask2FormerImageProcessor(
        do_resize=True, do_normalize=True, num_labels=1, ignore_index=0,
    )
    model = Mask2FormerForUniversalSegmentation.from_pretrained(str(checkpoint_dir)).to(device)

    tune_results = evaluate_thresholds(
        model, processor, tune_records, device, infer_size, THRESHOLDS,
    )
    selected = select_tune_threshold(tune_results)
    audit = evaluate_thresholds(
        model, processor, audit_records, device, infer_size, (selected["threshold"],),
    )[0]
    cleared_submit_bar = (
        selected["pq"] >= SUBMIT_BAR_TUNE and audit["pq"] >= SUBMIT_BAR_AUDIT
    )
    result = {
        "checkpoint_sha256": checkpoint_sha256(checkpoint_dir),
        "eval_resolution": infer_size,
        "selection": "maximum canonical tune PQ; audit evaluated only after selection",
        "thresholds": list(THRESHOLDS),
        "populations": populations,
        "tune_results": tune_results,
        "selected_threshold": selected["threshold"],
        "best_tune_pq": selected["pq"],
        "selected_tune": selected,
        "audit_pq": audit["pq"],
        "selected_audit": audit,
        "submit_bar_tune": SUBMIT_BAR_TUNE,
        "submit_bar_audit": SUBMIT_BAR_AUDIT,
        "promotion_gate_tune": PROMOTION_GATE_TUNE,
        "cleared_submit_bar": cleared_submit_bar,
        "cleared_promotion_gate": selected["pq"] >= PROMOTION_GATE_TUNE,
        "export_authorized": cleared_submit_bar,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "raw")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--infer-size", type=int, default=IMG_SIZE_TRAIN)
    args = parser.parse_args()
    root = args.data_root / "filament-segmentation-2026" / "MAGFiLO_1.0_Kaggle_2026"
    run_calibration(
        args.checkpoint,
        root / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json",
        root / "train" / "train_images",
        args.out,
        args.infer_size,
    )


if __name__ == "__main__":
    main()
