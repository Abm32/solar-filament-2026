"""V48: full-population train with EdgeAttNet/V23 preprocessing to lift recall.

The V46 decomposition showed the dominant PQ loss is 475 hard false-negatives
(faint filaments the raw-input V44 never detects). EdgeAttNet (SOTA on MAGFiLO)
and V17 both rely on radial-flatten + CLAHE preprocessing to make faint filaments
visible. V48 retrains the full-population yolo11s-seg on preprocessed inputs and
evaluates on preprocessed held-out tune, testing whether preprocessing recovers
the hard-FN recall and raises tune PQ toward / past V17 (0.448703).

Modes: train | tune. Preprocessing applied identically to both. No test/export here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ANNOTATION_SHA256 = "5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1"
IMGSZ = 2048
MIN_AREA = 20
CONF_SWEEP = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50)
V17_TUNE_PQ = 0.448703


def sha256(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as s:
        for b in iter(lambda: s.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def preprocess(np, cv2, gray):
    """EdgeAttNet/V23 pipeline: disk-mask, radial flatten, Gaussian(3x3,0.7), CLAHE.
    Returns a uint8 2048x2048 single-channel image."""
    g = gray.astype(np.float32)
    h, w = g.shape
    cy, cx = h / 2.0, w / 2.0
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    radius = min(h, w) / 2.0 * 0.98
    disk = r <= radius
    nb = 100
    rb = (r / radius * nb).astype(int).clip(0, nb - 1)
    prof = np.array(
        [g[disk & (rb == i)].mean() if np.any(disk & (rb == i)) else 0.0 for i in range(nb)]
    )
    prof_s = cv2.GaussianBlur(prof.reshape(-1, 1), (1, 9), 0).ravel()
    bg = np.where(disk, prof_s[rb], 1.0)
    bg = np.maximum(bg, 1.0)
    flat = np.where(disk, g / bg, 0.0)
    flat = np.clip(flat / max(float(flat.max()), 1e-6), 0.0, 1.0)
    sm = cv2.GaussianBlur(flat, (3, 3), 0.7)
    u8 = np.rint(sm * 255).astype(np.uint8)
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(u8)
    cl[~disk] = 0
    return cl


def annotation_rle(mask_utils, info, ann):
    return mask_utils.merge(
        mask_utils.frPyObjects(ann["segmentation"], int(info["height"]), int(info["width"]))
    )


def canonical_split(document):
    by_file = defaultdict(list)
    images = {str(r["id"]): r for r in document["images"]}
    for image_id, info in images.items():
        by_file[info["file_name"]].append(image_id)
    files = sorted(by_file)
    import random

    random.Random(42).shuffle(files)
    tune = sorted(files[:141])
    fit = sorted(files[141:])
    annotations = defaultdict(list)
    for r in document["annotations"]:
        annotations[str(r["image_id"])].append(r)
    return fit, tune, by_file, images, annotations


def records_for(files, by_file, annotations):
    out = []
    for name in files:
        iid = max(
            by_file[name],
            key=lambda v: (len(annotations[v]), sum(r.get("area", 0) for r in annotations[v]), v),
        )
        out.append((name, iid))
    return out


def disjoint(np, cv2, mask_utils, result, confidence):
    if result.masks is None:
        return []
    data = result.masks.data.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    order = np.argsort(-confs)
    occ = np.zeros((2048, 2048), dtype=bool)
    out = []
    for i in order:
        if confs[i] < confidence:
            continue
        m = data[i]
        if m.shape != (2048, 2048):
            m = cv2.resize(m.astype(np.float32), (2048, 2048), interpolation=cv2.INTER_LINEAR)
        b = (m >= 0.5) & (~occ)
        if int(b.sum()) < MIN_AREA:
            continue
        occ |= b
        out.append(mask_utils.encode(np.asfortranarray(b.astype(np.uint8))))
    return out


def pq(np, mask_utils, preds, targets):
    if not preds or not targets:
        return 0.0, 0, len(preds), len(targets)
    iou = np.asarray(mask_utils.iou(preds, targets, [0] * len(targets)), dtype=float)
    hits = np.argwhere(iou > 0.5)
    ordered = sorted(hits.tolist(), key=lambda pr: (-iou[pr[0], pr[1]], pr))
    up, ut, s = set(), set(), 0.0
    for p, t in ordered:
        if p not in up and t not in ut:
            up.add(p)
            ut.add(t)
            s += float(iou[p, t])
    return s, len(up), len(preds) - len(up), len(targets) - len(ut)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["train", "tune"], required=True)
    ap.add_argument("--annotations", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=120)
    args = ap.parse_args()
    if sha256(args.annotations) != ANNOTATION_SHA256:
        raise RuntimeError("annotation identity changed")

    import cv2
    import numpy as np
    from pycocotools import mask as mask_utils
    from ultralytics import YOLO

    document = json.loads(args.annotations.read_text())
    fit, tune, by_file, images, annotations = canonical_split(document)

    if args.mode == "train":
        recs = records_for(fit, by_file, annotations)
        img_out = args.work / "images" / "train"
        lbl_out = args.work / "labels" / "train"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        n_inst = 0
        for name, iid in recs:
            info = images[iid]
            gray = cv2.imread(str(args.images / name), cv2.IMREAD_GRAYSCALE)
            if gray is None or gray.shape != (2048, 2048):
                raise RuntimeError(f"bad fit image {name}")
            pre = preprocess(np, cv2, gray)
            stem = Path(name).stem
            cv2.imwrite(str(img_out / f"{stem}.png"), pre)
            lines = []
            for row in annotations[iid]:
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
        yaml_path = args.work / "v48.yaml"
        yaml_path.write_text(
            f"path: {args.work}\ntrain: images/train\nval: images/train\nnc: 1\nnames: [filament]\n"
        )
        model = YOLO(str(args.weights))
        model.train(
            data=str(yaml_path), epochs=args.epochs, imgsz=IMGSZ, batch=2, device=0,
            workers=2, optimizer="AdamW", lr0=1e-3, seed=20260930, deterministic=True,
            augment=False, mosaic=0.0, mixup=0.0, copy_paste=0.0, erasing=0.0,
            hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, translate=0.0, scale=0.0,
            fliplr=0.0, flipud=0.0, degrees=0.0, plots=False, val=False,
            project=str(args.work / "runs"), name="v48", exist_ok=True, verbose=True,
        )
        best = args.work / "runs" / "v48" / "weights" / "best.pt"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "status": "train_complete", "n_fit_files": len(recs), "n_instances": n_inst,
            "epochs": args.epochs, "preprocessing": "radial_flatten_clahe",
            "weights": str(best), "weights_sha256": sha256(best),
        }, indent=2) + "\n")
        print(json.dumps({"train_complete": True, "n_instances": n_inst, "weights": str(best)}))

    else:  # tune
        recs = records_for(tune, by_file, annotations)
        model = YOLO(str(args.weights))
        cache = {}
        targets_by = {}
        for name, iid in recs:
            info = images[iid]
            gray = cv2.imread(str(args.images / name), cv2.IMREAD_GRAYSCALE)
            pre = preprocess(np, cv2, gray)
            rgb = cv2.cvtColor(pre, cv2.COLOR_GRAY2BGR)
            cache[name] = model.predict(
                source=rgb, imgsz=IMGSZ, conf=0.01, iou=0.6, max_det=1000,
                device=0, retina_masks=False, verbose=False,
            )[0]
            targets_by[name] = [annotation_rle(mask_utils, info, r) for r in annotations[iid]]
        cells = []
        for conf in CONF_SWEEP:
            s = tp = fp = fn = 0
            for name, _ in recs:
                preds = disjoint(np, cv2, mask_utils, cache[name], conf)
                ss, t, f, n = pq(np, mask_utils, preds, targets_by[name])
                s += ss
                tp += t
                fp += f
                fn += n
            denom = tp + 0.5 * (fp + fn)
            cells.append({"confidence": conf, "pq": float(s / denom) if denom else 0.0,
                          "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
                          "tp": tp, "fp": fp, "fn": fn})
        best = max(cells, key=lambda c: c["pq"])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "status": "tune_complete", "cells": cells, "best": best,
            "v17_tune_pq": V17_TUNE_PQ, "beats_v17": best["pq"] > V17_TUNE_PQ,
            "weights": str(args.weights), "preprocessing": "radial_flatten_clahe",
        }, indent=2) + "\n")
        print(json.dumps({"best": best, "v17_tune_pq": V17_TUNE_PQ,
                          "beats_v17": best["pq"] > V17_TUNE_PQ}, indent=2))


if __name__ == "__main__":
    main()
