"""V44: full-population train of the V42 recipe + held-out tune eval + test export.

V42 proved yolo11s-seg @ 2048 breaks the recall wall (fit recall 0.997, FP 1).
V44 trains that exact recipe on ALL canonical fit files (one max-instance record
per physical image), evaluates honestly on the held-out canonical TUNE split with
the project PQ metric (src/pq_eval semantics), then runs test inference and
exports a competition-format, non-overlapping submission CSV.

Modes:
  --mode train   : train on all fit files -> weights
  --mode tune    : evaluate a checkpoint on the held-out tune split (PQ), sweep conf
  --mode test    : run test inference -> validated submission.csv

Firewall note: tune labels are used ONLY for evaluation in --mode tune, never for
training. Training uses fit files only. Test images have no labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("YOLO_OFFLINE", "True")

ANNOTATION_SHA256 = "5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1"
IMGSZ = 2048
MIN_AREA = 20
CONF_SWEEP = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50)
V17_TUNE_PQ = 0.448703  # bar to beat locally


def sha256(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as s:
        for b in iter(lambda: s.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def annotation_rle(mask_utils, info, ann):
    return mask_utils.merge(
        mask_utils.frPyObjects(ann["segmentation"], int(info["height"]), int(info["width"]))
    )


def canonical_split(document):
    """Return (fit_files, tune_files) using the project's frozen split."""
    by_file = defaultdict(list)
    images = {str(r["id"]): r for r in document["images"]}
    for image_id, info in images.items():
        by_file[info["file_name"]].append(image_id)
    files = sorted(by_file)
    import random

    random.Random(42).shuffle(files)
    tune = sorted(files[:141])
    fit = sorted(files[141:])
    return fit, tune, by_file, images


def records_for(files, by_file, images, annotations):
    """One max-instance annotator record per physical file (project convention)."""
    out = []
    for name in files:
        image_id = max(
            by_file[name],
            key=lambda v: (len(annotations[v]), sum(r.get("area", 0) for r in annotations[v]), v),
        )
        out.append((name, image_id))
    return out


def build_train_dataset(np, cv2, mask_utils, document, image_dir, work: Path):
    images = {str(r["id"]): r for r in document["images"]}
    annotations = defaultdict(list)
    for r in document["annotations"]:
        annotations[str(r["image_id"])].append(r)
    fit, _tune, by_file, _ = canonical_split(document)
    recs = records_for(fit, by_file, images, annotations)
    img_out = work / "images" / "train"
    lbl_out = work / "labels" / "train"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    n_inst = 0
    for name, image_id in recs:
        info = images[image_id]
        gray = cv2.imread(str(image_dir / name), cv2.IMREAD_GRAYSCALE)
        if gray is None or gray.shape != (2048, 2048):
            raise RuntimeError(f"bad fit image {name}")
        stem = Path(name).stem
        cv2.imwrite(str(img_out / f"{stem}.png"), gray)
        lines = []
        for row in annotations[image_id]:
            m = mask_utils.decode(annotation_rle(mask_utils, info, row)).astype(np.uint8)
            cont, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cont:
                continue
            c = max(cont, key=cv2.contourArea)
            if len(c) < 3:
                continue
            poly = c.reshape(-1, 2).astype(np.float64) / 2048.0
            lines.append("0 " + " ".join(f"{v:.6f}" for v in poly.reshape(-1)))
            n_inst += 1
        (lbl_out / f"{stem}.txt").write_text("\n".join(lines) + "\n")
    yaml_path = work / "v44.yaml"
    yaml_path.write_text(
        f"path: {work}\ntrain: images/train\nval: images/train\nnc: 1\nnames: [filament]\n"
    )
    return yaml_path, len(recs), n_inst


def disjoint_instances(np, cv2, mask_utils, result, confidence):
    """Return list of full-res non-overlapping binary RLEs, highest-conf wins pixels."""
    if result.masks is None:
        return []
    data = result.masks.data.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    order = np.argsort(-confs)
    occupied = np.zeros((2048, 2048), dtype=bool)
    out = []
    for i in order:
        if confs[i] < confidence:
            continue
        m = data[i]
        if m.shape != (2048, 2048):
            m = cv2.resize(m.astype(np.float32), (2048, 2048), interpolation=cv2.INTER_LINEAR)
        binary = (m >= 0.5) & (~occupied)
        if int(binary.sum()) < MIN_AREA:
            continue
        occupied |= binary
        out.append(mask_utils.encode(np.asfortranarray(binary.astype(np.uint8))))
    return out


def pq_on_tune(np, cv2, mask_utils, model, document, image_dir, device, confidence):
    """Project PQ over the held-out tune split at a given confidence."""
    images = {str(r["id"]): r for r in document["images"]}
    annotations = defaultdict(list)
    for r in document["annotations"]:
        annotations[str(r["image_id"])].append(r)
    _fit, tune, by_file, _ = canonical_split(document)
    recs = records_for(tune, by_file, images, annotations)
    iou_sum = tp = fp = fn = 0
    matched_all = []
    for name, image_id in recs:
        info = images[image_id]
        gray = cv2.imread(str(image_dir / name), cv2.IMREAD_GRAYSCALE)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        result = model.predict(
            source=rgb, imgsz=IMGSZ, conf=0.01, iou=0.6, max_det=1000,
            device=device, retina_masks=False, verbose=False,
        )[0]
        preds = disjoint_instances(np, cv2, mask_utils, result, confidence)
        targets = [annotation_rle(mask_utils, info, r) for r in annotations[image_id]]
        if preds and targets:
            iou = np.asarray(mask_utils.iou(preds, targets, [0] * len(targets)), dtype=float)
            hits = np.argwhere(iou > 0.5)
            ordered = sorted(hits.tolist(), key=lambda pr: (-iou[pr[0], pr[1]], pr))
            up, ut = set(), set()
            for p, t in ordered:
                if p not in up and t not in ut:
                    up.add(p)
                    ut.add(t)
                    matched_all.append(float(iou[p, t]))
            tp += len(up)
            fp += len(preds) - len(up)
            fn += len(targets) - len(ut)
        else:
            fp += len(preds)
            fn += len(targets)
    iou_sum = sum(matched_all)
    denom = tp + 0.5 * (fp + fn)
    return {
        "confidence": confidence,
        "pq": float(iou_sum / denom) if denom else 0.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "median_matched_iou": float(np.median(matched_all)) if matched_all else 0.0,
        "tp": tp, "fp": fp, "fn": fn, "n_images": len(recs),
    }


def gpu_preflight(torch):
    if not torch.cuda.is_available():
        raise RuntimeError("V44 requires CUDA")
    p = torch.cuda.get_device_properties(0)
    return {"name": p.name, "total_mib": p.total_memory / 2**20,
            "visible": torch.cuda.device_count()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["train", "tune", "test"], required=True)
    ap.add_argument("--annotations", type=Path, required=True)
    ap.add_argument("--images", type=Path)
    ap.add_argument("--test-images", type=Path)
    ap.add_argument("--weights", type=Path)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--confidence", type=float, default=0.30)
    args = ap.parse_args()
    if sha256(args.annotations) != ANNOTATION_SHA256:
        raise RuntimeError("V44 annotation identity changed")

    import cv2
    import numpy as np
    import torch
    from pycocotools import mask as mask_utils
    from ultralytics import YOLO

    document = json.loads(args.annotations.read_text())
    device = 0

    if args.mode == "train":
        gpu_preflight(torch)
        yaml_path, n_files, n_inst = build_train_dataset(
            np, cv2, mask_utils, document, args.images, args.work
        )
        model = YOLO(str(args.weights))
        model.train(
            data=str(yaml_path), epochs=args.epochs, imgsz=IMGSZ, batch=2, device=device,
            workers=2, optimizer="AdamW", lr0=1e-3, seed=20260930, deterministic=True,
            augment=False, mosaic=0.0, mixup=0.0, copy_paste=0.0, erasing=0.0,
            hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, translate=0.0, scale=0.0,
            fliplr=0.0, flipud=0.0, degrees=0.0, plots=False, val=False,
            project=str(args.work / "runs"), name="v44", exist_ok=True, verbose=True,
        )
        best = args.work / "runs" / "v44" / "weights" / "best.pt"
        write_json(args.output, {
            "status": "train_complete", "mode": "train", "n_fit_files": n_files,
            "n_instances": n_inst, "epochs": args.epochs,
            "weights": str(best), "weights_sha256": sha256(best),
            "reads_tune": False, "reads_test": False,
        })
        print(json.dumps({"train_complete": True, "n_fit_files": n_files,
                          "n_instances": n_inst, "weights": str(best)}))

    elif args.mode == "tune":
        model = YOLO(str(args.weights))
        cells = [pq_on_tune(np, cv2, mask_utils, model, document, args.images, device, c)
                 for c in CONF_SWEEP]
        best = max(cells, key=lambda c: c["pq"])
        write_json(args.output, {
            "status": "tune_complete", "mode": "tune", "cells": cells, "best": best,
            "v17_tune_pq": V17_TUNE_PQ, "beats_v17": best["pq"] > V17_TUNE_PQ,
            "weights": str(args.weights), "reads_tune": True, "reads_test": False,
        })
        print(json.dumps({"best": best, "v17_tune_pq": V17_TUNE_PQ,
                          "beats_v17": best["pq"] > V17_TUNE_PQ}, indent=2))

    elif args.mode == "test":
        import pandas as pd

        stems = sorted(p.stem for p in args.test_images.glob("*.jpeg"))
        if len(stems) != 180:
            raise RuntimeError(f"expected 180 test images, found {len(stems)}")
        model = YOLO(str(args.weights))
        rows = []
        populated = 0
        for stem in stems:
            path = next(args.test_images.glob(f"{stem}.jpeg"))
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            result = model.predict(
                source=rgb, imgsz=IMGSZ, conf=0.01, iou=0.6, max_det=1000,
                device=device, retina_masks=False, verbose=False,
            )[0]
            preds = disjoint_instances(np, cv2, mask_utils, result, args.confidence)
            if preds:
                populated += 1
            for i, rle in enumerate(preds, start=1):
                counts = rle["counts"]
                counts = counts.decode("utf-8") if isinstance(counts, bytes) else str(counts)
                rows.append(
                    {"filament_id": f"{stem}_{i}", "segmentation_rle": counts}
                )
        csv_path = args.work / "submission.csv"
        pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]).to_csv(
            csv_path, index=False
        )
        write_json(args.output, {
            "status": "test_complete", "mode": "test", "confidence": args.confidence,
            "rows": len(rows), "populated_images": populated, "total_images": len(stems),
            "csv": str(csv_path), "csv_sha256": sha256(csv_path),
            "weights": str(args.weights), "reads_tune": False, "reads_test": False,
        })
        print(json.dumps({"rows": len(rows), "populated_images": populated,
                          "csv": str(csv_path)}))


if __name__ == "__main__":
    main()
