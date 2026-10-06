"""V58 step 1: build an OOF (out-of-fold) hard-negative CANDIDATE dataset for the
filament verifier.

Rationale (docs/swefil_correction_and_revised_strategy_2026-10-04.md): our bottleneck is
RECOGNITION (faint real filament vs filament-looking dark junk / sunspot / network), not
mask quality. A high-recall detector proposes many candidates; a separate learned
VERIFIER decides which are real. The verifier must train on the detector's OWN mistakes,
sampled OUT-OF-FOLD so the labels are honest (a model that trained on an image produces
optimistically-clean proposals there).

This script runs a Mask R-CNN detector at LOW confidence (high recall) over a set of
files and, for every proposal, records:
  - crop of the candidate region + context (2x the box, so the verifier sees surroundings)
  - label: TP (best GT IoU >= 0.5) / FP (IoU < 0.1) / NEARMISS (0.1<=IoU<0.5, uncertain)
  - features: detector score, box area, log_area, aspect/elongation, disk position (r,theta)
    normalised by solar-disk radius, mean/contrast of the crop.
Outputs a manifest .jsonl (one row per candidate) + crop .npy stack per shard.

Two OOF modes:
  --oof-mode kfold : the proper OOF — pass --fold-weights mapping each held-out fold to a
                     detector trained ONLY on the other folds (built by train_v58_fold.py).
  --oof-mode tune  : cheap honest OOF — use an existing full-FIT-trained detector but ONLY
                     generate candidates on the 141 TUNE files (never in its training set).

Firewall: FIT/tune labels used only to LABEL candidates (as in any eval); no test, no
external data here. Context crops come from the competition images only.

  CUDA_VISIBLE_DEVICES=N python scripts/build_v58_oof_candidates.py \
      --weights <mrcnn.pt> --annotations <ann.json> --images <train_images> \
      --files fold3 --conf 0.10 --out outputs/v58_oof/fold3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import train_v52_cldice_semantic as v52  # noqa: E402
import train_v56_maskrcnn as v56  # noqa: E402

FULL = 2048
CROP = 128          # verifier input crop size (resized)
CONTEXT = 2.0       # context multiple of the candidate box
TP_IOU, FP_IOU = 0.5, 0.1


def disk_params(gray, np_):
    """Estimate solar disk centre + radius from the Ha image (bright disk on dark bg)."""
    ys, xs = np_.nonzero(gray > max(10, int(gray.mean() * 0.3)))
    if len(xs) < 100:
        return FULL / 2, FULL / 2, FULL / 2
    cx, cy = float(xs.mean()), float(ys.mean())
    r = float(np_.sqrt(((xs - cx) ** 2 + (ys - cy) ** 2).mean()) * 1.4)
    return cx, cy, max(r, 1.0)


def candidate_features(np_, box, score, mask, gray, disk):
    x0, y0, x1, y1 = box
    cx, cy, r = disk
    bw, bh = max(x1 - x0, 1), max(y1 - y0, 1)
    mcx, mcy = (x0 + x1) / 2, (y0 + y1) / 2
    area = float(mask.sum())
    rr = float(np_.hypot(mcx - cx, mcy - cy) / r)                 # normalised disk radius
    theta = float(np_.arctan2(mcy - cy, mcx - cx))                 # disk angle
    reg = gray[int(y0):int(y1), int(x0):int(x1)]
    mean_i = float(reg.mean()) / 255.0 if reg.size else 0.0
    contrast = float(reg.std()) / 255.0 if reg.size else 0.0
    return {
        "score": float(score), "box_area": float(bw * bh), "mask_area": area,
        "log_area": float(np_.log1p(area)), "elongation": float(max(bw, bh) / min(bw, bh)),
        "disk_r": rr, "disk_theta": theta, "mean_int": mean_i, "contrast": contrast,
    }


def extract(args):
    import cv2
    import torch
    from pycocotools import mask as mask_utils

    device = torch.device("cuda:0")
    document = json.loads(args.annotations.read_text())
    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    imgsz = ckpt.get("imgsz", 1024)
    model = v56.build_model(torch, imgsz).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    images = {str(r["id"]): r for r in document["images"]}
    annotations = v52.load_annotations(document)
    by_file = v52.canonical_split(document)[2]
    recs = v52.records_for(sorted(args.files_list), by_file, images, annotations)

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = (args.out / "candidates.jsonl").open("w")
    crops, n_tp, n_fp, n_nm = [], 0, 0, 0
    with torch.no_grad():
        for idx, (name, image_id) in enumerate(recs):
            info = images[image_id]
            gray = cv2.imread(str(args.images / name), cv2.IMREAD_GRAYSCALE)
            disk = disk_params(gray, np)
            x = (cv2.resize(gray, (imgsz, imgsz), interpolation=cv2.INTER_AREA)
                 if imgsz != FULL else gray)
            xt = torch.from_numpy(x.astype(np.float32) / 255.0)[None].repeat(3, 1, 1).to(device)
            with torch.cuda.amp.autocast():
                out = model([xt])[0]
            keep = out["scores"].cpu().numpy() >= args.conf
            boxes = out["boxes"].cpu().numpy()[keep]
            scores = out["scores"].cpu().numpy()[keep]
            masks = (out["masks"][:, 0].cpu().numpy()[keep] >= 0.5).astype(np.uint8)
            # upscale boxes/masks to FULL
            sc = FULL / imgsz
            gt = [v52.annotation_rle(mask_utils, info, e) for e in annotations[image_id]]
            gt = [r for r in gt if mask_utils.area(r) > 0]
            for b, s, m in zip(boxes * sc, scores, masks, strict=True):
                mfull = cv2.resize(m.astype(np.float32), (FULL, FULL),
                                   interpolation=cv2.INTER_LINEAR) >= 0.5
                mfull = mfull.astype(np.uint8)
                if int(mfull.sum()) < v56.MIN_AREA:
                    continue
                rle = mask_utils.encode(np.asfortranarray(mfull))
                iou = (float(np.asarray(mask_utils.iou([rle], gt, [0] * len(gt))).max())
                       if gt else 0.0)
                label = "tp" if iou >= TP_IOU else ("fp" if iou < FP_IOU else "nearmiss")
                n_tp += int(label == "tp")
                n_fp += int(label == "fp")
                n_nm += int(label == "nearmiss")
                feat = candidate_features(np, b, s, mfull, gray, disk)
                # context crop
                x0, y0, x1, y1 = b
                cxb, cyb = (x0 + x1) / 2, (y0 + y1) / 2
                half = max(x1 - x0, y1 - y0) * CONTEXT / 2
                a, bb = int(max(0, cxb - half)), int(max(0, cyb - half))
                c, d = int(min(FULL, cxb + half)), int(min(FULL, cyb + half))
                crop = gray[bb:d, a:c]
                crop = (cv2.resize(crop, (CROP, CROP), interpolation=cv2.INTER_AREA)
                        if crop.size else np.zeros((CROP, CROP), np.uint8))
                crops.append(crop.astype(np.uint8))
                row = {"file": name, "iou": iou, "label": label, "crop_idx": len(crops) - 1,
                       **feat}
                manifest.write(json.dumps(row) + "\n")
            print(f"{idx + 1}/{len(recs)} {name} cands+={keep.sum()}", flush=True)
    manifest.close()
    stack = np.stack(crops) if crops else np.zeros((0, CROP, CROP), np.uint8)
    np.save(args.out / "crops.npy", stack)
    summary = {"n_candidates": len(crops), "tp": n_tp, "fp": n_fp, "nearmiss": n_nm,
               "conf": args.conf, "weights": str(args.weights), "n_files": len(recs),
               "reads_test": False}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


def resolve_files(document, spec):
    """spec: 'tune' | 'fit' | 'foldK' (K in 0..3, OOF quarter of fit)."""
    import random
    fit, tune, _by, _ = v52.canonical_split(document)
    if spec == "tune":
        return tune
    if spec == "fit":
        return fit
    if spec.startswith("fold"):
        k = int(spec[4:])
        files = sorted(fit)
        random.Random(1234).shuffle(files)
        return sorted(files[k::4])  # held-out quarter k
    raise SystemExit(f"unknown --files {spec}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--annotations", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--files", required=True, help="tune | fit | fold0..fold3")
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if v52.sha256(args.annotations) != v56.ANNOTATION_SHA256:
        raise RuntimeError("annotation identity changed")
    document = json.loads(args.annotations.read_text())
    args.files_list = resolve_files(document, args.files)
    extract(args)


if __name__ == "__main__":
    main()
