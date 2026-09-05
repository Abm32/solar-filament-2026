"""Inspect cached predictions on tuning files without rerunning GPU inference."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from pycocotools import mask as mask_utils


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache_dir", type=Path)
    parser.add_argument("--baseline", type=Path, default=root / "outputs/kaggle_coco_refiner")
    args = parser.parse_args()
    manifest = json.loads((args.baseline / "inference_manifest.json").read_text())
    selected = json.loads((args.baseline / "submission_candidates.json").read_text())[0]
    path = next((root / "data/raw").rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    coco = json.loads(path.read_text())
    images, annotations = defaultdict(list), defaultdict(list)
    for info in coco["images"]:
        images[info["file_name"]].append(info)
    for annotation in coco["annotations"]:
        annotations[annotation["image_id"]].append(annotation)
    rows = []
    for cache in sorted(args.cache_dir.glob("*.json.gz")):
        with gzip.open(cache, "rt") as stream:
            entry = json.load(stream)
        name = entry["file_name"]
        if name not in manifest["tune_files"]:
            raise ValueError(f"Visual diagnosis is restricted to tuning files: {name}")
        variant = next(v for v in entry["variants"] if all(
            v[key] == selected[key] for key in ("mode", "threshold", "tta")))
        predictions = [dict(size=r["size"], counts=r["counts"].encode())
                       for r, score in zip(variant["rles"], variant["scores"])
                       if score >= selected["confidence"]]
        image = cv2.imread(str(path.parent / "train_images" / name))
        for info in images[name]:
            gt = [mask_utils.merge(mask_utils.frPyObjects(a["segmentation"], info["height"], info["width"]))
                  for a in annotations[info["id"]]]
            iou = (mask_utils.iou(predictions, gt, [0] * len(gt)).T if predictions and gt
                   else np.zeros((len(gt), len(predictions))))
            for index, rle in enumerate(gt):
                best_index = int(np.argmax(iou[index])) if predictions else None
                best_iou = float(iou[index, best_index]) if predictions else 0.0
                area = int(mask_utils.area(rle))
                coverage, purity = 0.0, 0.0
                if best_index is not None and best_iou > 0:
                    pred = predictions[best_index]
                    intersection = float(mask_utils.area(mask_utils.merge([rle, pred], intersect=True)))
                    coverage, purity = intersection / area, intersection / float(mask_utils.area(pred))
                reason = ("matched" if best_iou > 0.5 else "no_overlap" if best_iou == 0 else
                          "under_coverage" if coverage < 0.5 else "over_coverage" if purity < 0.5
                          else "boundary_or_mixed")
                rows.append(dict(file_name=name, image_id=info["id"], gt_index=index,
                                 gt_area=area, best_iou=best_iou, gt_coverage=coverage,
                                 prediction_purity=purity, diagnosis=reason))
            panels = []
            for label, masks, color in (("Annotations (cyan)", gt, (255, 255, 0)),
                                        ("Predictions (orange)", predictions, (0, 160, 255))):
                panel = cv2.resize(image, (768, 768))
                for index, rle in enumerate(masks):
                    small = cv2.resize(mask_utils.decode(rle), (768, 768), interpolation=cv2.INTER_NEAREST)
                    contours, _ = cv2.findContours(small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(panel, contours, -1, color, 1)
                    ys, xs = np.where(small)
                    if len(xs):
                        cv2.putText(panel, str(index), (int(xs.mean()), int(ys.mean())),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                panel = Image.fromarray(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
                canvas = Image.new("RGB", (768, 804), "#111111")
                canvas.paste(panel, (0, 36))
                ImageDraw.Draw(canvas).text((8, 10), label, fill="white")
                panels.append(canvas)
            canvas = Image.new("RGB", (1536, 834), "#111111")
            ImageDraw.Draw(canvas).text((8, 10), f"{name} / annotator image {info['id']}", fill="white")
            for index, panel in enumerate(panels):
                canvas.paste(panel, (index * 768, 30))
            canvas.save(args.cache_dir.parent / f"failure_{info['id']}.png")
    if not rows:
        raise ValueError("No cached tuning annotations found")
    frame = pd.DataFrame(rows)
    frame.to_csv(args.cache_dir.parent / "failure_diagnostics.csv", index=False)
    print(frame.groupby(["file_name", "diagnosis"]).size().to_string())


if __name__ == "__main__":
    main()
