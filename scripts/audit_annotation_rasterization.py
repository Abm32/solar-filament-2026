"""Compare COCO and historical OpenCV raster areas with the supplied annotation areas."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils


def rounded_area(polygons, height, width):
    points = [np.round(np.asarray(p, dtype=np.float32).reshape(-1, 2)).astype(np.int32)
              for p in polygons]
    joined = np.concatenate(points)
    x0, y0 = np.maximum(joined.min(axis=0), [0, 0])
    x1, y1 = np.minimum(joined.max(axis=0) + 1, [width, height])
    if x1 <= x0 or y1 <= y0:
        return 0
    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    for polygon in points:
        cv2.fillPoly(mask, [polygon - [x0, y0]], 1)
    return int(mask.sum())


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--output", type=Path, default=root / "outputs/annotation_rasterization")
    args = parser.parse_args()
    path = args.annotations or next((root / "data/raw").rglob(
        "MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    coco = json.loads(path.read_text())
    images = {image["id"]: image for image in coco["images"]}
    rows = []
    for ann in coco["annotations"]:
        info = images[ann["image_id"]]
        h, w = info["height"], info["width"]
        rle = mask_utils.merge(mask_utils.frPyObjects(ann["segmentation"], h, w))
        rows.append(dict(annotation_id=ann["id"], supplied_area=ann["area"],
                         coco_area=int(mask_utils.area(rle)),
                         rounded_area=rounded_area(ann["segmentation"], h, w)))
    data = pd.DataFrame(rows)
    summary = {"annotations": len(data)}
    for method in ("coco", "rounded"):
        areas = data[f"{method}_area"]
        summary[method] = {
            "exact_matches": int((areas == data.supplied_area).sum()),
            "median_area_ratio": float((areas / data.supplied_area).median()),
            "median_absolute_area_error": float((areas - data.supplied_area).abs().median()),
        }
    args.output.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output / "areas.csv", index=False)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
