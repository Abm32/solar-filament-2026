"""Compare two instance-segmentation submissions with one-to-one mask matching.

This is a structural comparison, not a quality estimate: test labels are unavailable.
It is intended to make post-submission regressions reproducible without using the
public leaderboard to tune a new candidate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

COLUMNS = ["filament_id", "segmentation_rle"]
DEFAULT_SHAPE = (2048, 2048)
MATCH_IOU = 0.5


def load_submission(path: Path, shape=DEFAULT_SHAPE) -> dict[str, list[dict]]:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(frame.columns) != COLUMNS:
        raise ValueError(f"Unexpected columns in {path}: {list(frame.columns)}")
    if frame.empty or frame.filament_id.duplicated().any():
        raise ValueError(f"Submission is empty or has duplicate IDs: {path}")

    grouped: dict[str, list[dict]] = {}
    for row in frame.itertuples(index=False):
        stem, separator, suffix = row.filament_id.rpartition("_")
        if not separator or not stem or not suffix.isdigit() or int(suffix) < 1:
            raise ValueError(f"Malformed filament ID: {row.filament_id}")
        rle = {"size": list(shape), "counts": row.segmentation_rle.encode("utf-8")}
        area = int(mask_utils.area(rle))
        if area <= 0:
            raise ValueError(f"Empty mask: {row.filament_id}")
        grouped.setdefault(stem, []).append(
            {"id": row.filament_id, "rle": rle, "area": area}
        )
    return grouped


def greedy_match(iou: np.ndarray, threshold: float = MATCH_IOU):
    """Highest-IoU-first one-to-one matching with the project's strict >0.5 rule."""
    if iou.size == 0:
        return []
    order = np.dstack(np.unravel_index(np.argsort(-iou, axis=None), iou.shape))[0]
    matches = []
    used_left: set[int] = set()
    used_right: set[int] = set()
    for left, right in order:
        value = float(iou[left, right])
        if value <= threshold:
            break
        if int(left) in used_left or int(right) in used_right:
            continue
        used_left.add(int(left))
        used_right.add(int(right))
        matches.append((int(left), int(right), value))
    return matches


def quantiles(values) -> dict[str, float]:
    series = pd.Series(list(values), dtype=float)
    if series.empty:
        return {}
    return {
        str(q): float(series.quantile(q))
        for q in (0, 0.1, 0.25, 0.5, 0.75, 0.9, 1)
    }


def compare_submissions(left_path: Path, right_path: Path, shape=DEFAULT_SHAPE):
    left = load_submission(left_path, shape)
    right = load_submission(right_path, shape)
    matched_rows = []
    left_only_rows = []
    right_only_rows = []
    image_rows = []

    for stem in sorted(set(left) | set(right)):
        left_instances = left.get(stem, [])
        right_instances = right.get(stem, [])
        if left_instances and right_instances:
            iou = mask_utils.iou(
                [item["rle"] for item in left_instances],
                [item["rle"] for item in right_instances],
                [0] * len(right_instances),
            )
        else:
            iou = np.zeros((len(left_instances), len(right_instances)), dtype=float)

        matches = greedy_match(iou)
        matched_left = {left_index for left_index, _, _ in matches}
        matched_right = {right_index for _, right_index, _ in matches}
        for left_index, right_index, value in matches:
            left_item = left_instances[left_index]
            right_item = right_instances[right_index]
            matched_rows.append(
                {
                    "stem": stem,
                    "left_id": left_item["id"],
                    "right_id": right_item["id"],
                    "iou": value,
                    "left_area": left_item["area"],
                    "right_area": right_item["area"],
                    "area_ratio_right_left": right_item["area"] / left_item["area"],
                }
            )
        for index, item in enumerate(left_instances):
            if index not in matched_left:
                best = float(iou[index].max()) if right_instances else 0.0
                left_only_rows.append(
                    {"stem": stem, "id": item["id"], "area": item["area"], "best_iou_right": best}
                )
        for index, item in enumerate(right_instances):
            if index not in matched_right:
                best = float(iou[:, index].max()) if left_instances else 0.0
                right_only_rows.append(
                    {"stem": stem, "id": item["id"], "area": item["area"], "best_iou_left": best}
                )
        image_rows.append(
            {
                "stem": stem,
                "left": len(left_instances),
                "right": len(right_instances),
                "matched": len(matches),
                "left_only": len(left_instances) - len(matches),
                "right_only": len(right_instances) - len(matches),
                "delta": len(right_instances) - len(left_instances),
            }
        )

    matched = pd.DataFrame(matched_rows)
    left_only = pd.DataFrame(left_only_rows)
    right_only = pd.DataFrame(right_only_rows)
    per_image = pd.DataFrame(image_rows)
    summary = {
        "left_rows": sum(map(len, left.values())),
        "right_rows": sum(map(len, right.values())),
        "matched_iou_gt_0_5": len(matched),
        "left_only": len(left_only),
        "right_only": len(right_only),
        "matched_iou_quantiles": quantiles(matched.get("iou", [])),
        "matched_area_ratio_quantiles": quantiles(matched.get("area_ratio_right_left", [])),
        "left_only_area_quantiles": quantiles(left_only.get("area", [])),
        "right_only_area_quantiles": quantiles(right_only.get("area", [])),
        "images_right_fewer": int((per_image.delta < 0).sum()),
        "images_equal": int((per_image.delta == 0).sum()),
        "images_right_more": int((per_image.delta > 0).sum()),
        "sum_negative_delta": int(per_image.loc[per_image.delta < 0, "delta"].sum()),
        "sum_positive_delta": int(per_image.loc[per_image.delta > 0, "delta"].sum()),
        "images_with_left_only": int((per_image.left_only > 0).sum()),
        "images_with_right_only": int((per_image.right_only > 0).sum()),
        "near_unmatched_left_iou_0_3_0_5": int(
            left_only.get("best_iou_right", pd.Series(dtype=float)).between(0.3, 0.5, inclusive="both").sum()
        ),
        "near_unmatched_right_iou_0_3_0_5": int(
            right_only.get("best_iou_left", pd.Series(dtype=float)).between(0.3, 0.5, inclusive="both").sum()
        ),
    }
    return summary, matched, left_only, right_only, per_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    summary, matched, left_only, right_only, per_image = compare_submissions(args.left, args.right)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    matched.to_csv(args.output_dir / "matched.csv", index=False)
    left_only.to_csv(args.output_dir / "left_only.csv", index=False)
    right_only.to_csv(args.output_dir / "right_only.csv", index=False)
    per_image.to_csv(args.output_dir / "per_image.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
