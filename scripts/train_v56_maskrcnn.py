"""V56: Mask R-CNN (torchvision maskrcnn_resnet50_fpn_v2, COCO-pretrained) instance
segmentation — a genuine architecture PIVOT, not another V17/V52 fusion tweak.

Why (2026-10-03 external research + internal learnings):
- LEADERBOARD: top 0.63, dense cluster 0.55-0.63. Our V17 (public 0.39) is far down the
  board. The project's 0.4487-tune-PQ "ceiling" (E027/E028 hard-FN wall) describes V17's
  YOLO-family pipeline, NOT the competition. +0.01 fusion tweaks cannot reach 0.55+.
- The only REAL reproducible public kernel at this task uses this exact architecture
  (nomannic19). The "0.70"/"0.55" kernels are fake embedded-payload CSVs.
- V55 learning: the deficiency is INSTANCE-level (pixel losses can't fix instance FP). A
  proposal-based detector scores/penalises INSTANCES directly — the right tool.
- lamhuy analysis: high scorers win on RECALL (more valid instances across more images).
  V17 chronically under-recalls. Mask R-CNN + COCO objectness prior targets exactly this.

Pipeline: images grayscale->3ch; targets = per-instance binary masks + boxes + class-1
labels from the project annotations; maskrcnn_resnet50_fpn_v2 (COCO weights), transform
min/max size = IMAGE_SIZE; train FIT-only (566 files), evaluate on 141 held-out tune via
the PROJECT PQ (instance IoU>0.5 greedy one-to-one) with a score/mask-threshold sweep.

Firewall: ANNOTATION_SHA256 checked; FIT-only training (tune labels used ONLY to score
--mode tune); canonical Random(42) split; no test/CSV/submission here. Checkpoints are
self-describing. Reuses V52's split/annotation helpers (V52 untouched).

  CUDA_VISIBLE_DEVICES=N python scripts/train_v56_maskrcnn.py --mode train \
      --annotations <ann.json> --images <train_images> --work outputs/v56_mrcnn \
      --output outputs/v56_mrcnn/train.json --epochs 20 --batch 2 --imgsz 1024 --lr 5e-3
  CUDA_VISIBLE_DEVICES=N python scripts/train_v56_maskrcnn.py --mode tune \
      --annotations <ann.json> --images <train_images> \
      --weights outputs/v56_mrcnn/weights/v56_mrcnn_best.pt \
      --work outputs/v56_mrcnn --output outputs/v56_mrcnn/tune.json
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

ANNOTATION_SHA256 = v52.ANNOTATION_SHA256
FULL = 2048
MIN_AREA = 20
V17_TUNE_PQ = 0.448703
SCORE_SWEEP = (0.30, 0.40, 0.50, 0.60, 0.70)
MASK_THRESH = 0.50


# ------------------------------------------------------------- targets / dataset
def instance_masks(mask_utils, info, anns, np_):
    """Per-instance full-res binary masks for one image (project RLE -> uint8)."""
    masks = []
    for ann in anns:
        rle = v52.annotation_rle(mask_utils, info, ann)
        m = mask_utils.decode(rle)
        if m.ndim == 3:
            m = m[..., 0]
        if int(m.sum()) >= MIN_AREA:
            masks.append(m.astype(np_.uint8))
    return masks


def masks_to_boxes(np_, masks):
    boxes = []
    for m in masks:
        ys, xs = np_.nonzero(m)
        boxes.append([float(xs.min()), float(ys.min()), float(xs.max()) + 1, float(ys.max()) + 1])
    return boxes


# ------------------------------------------------------------- union of annotators
# MAGFiLO is multi-annotator and annotations are SPARSE: each annotator marks only a
# subset of the real filaments (measured min/max coverage ratio ~0.677, union/single
# ~1.35x). Training + scoring against one annotator punishes the model for detecting
# real-but-unannotated-by-that-annotator filaments. The union of all annotators per
# physical file is a far more complete label set (transfer of the BioHub sparse-GT
# insight). See docs/annotation_sparsity_finding_2026-10-04.md.
def union_rles(mask_utils, np_, document_images, by_file, annotations, file_name,
               match_iou=0.5, min_agree=1):
    """Cluster every annotator's instances for one physical file by IoU, keeping each
    cluster only if >= min_agree DISTINCT annotators contributed to it.

    min_agree=1 -> full union (every instance any annotator marked, deduped).
    min_agree=2 -> consensus: only instances >=2 annotators independently marked
                   (strips single-annotator noise; higher precision).
    Returns the representative (largest-area) RLE of each surviving cluster.
    """
    # collect (rle, annotator_id) for every instance, largest-area first
    items = []
    for image_id in by_file[file_name]:
        info = document_images[image_id]
        for ann in annotations.get(image_id, []):
            r = v52.annotation_rle(mask_utils, info, ann)
            if mask_utils.area(r) > 0:
                items.append((r, image_id))
    items.sort(key=lambda t: -float(mask_utils.area(t[0])))
    # greedy clustering: each new instance joins the first existing cluster it
    # overlaps (IoU>=match_iou), else starts a new cluster
    clusters = []  # list of dict(rep=rle, annset=set, members=[rle])
    for r, aid in items:
        placed = False
        for c in clusters:
            iou = float(np_.asarray(mask_utils.iou([c["rep"]], [r], [0]), dtype=float).max())
            if iou >= match_iou:
                c["annset"].add(aid)
                placed = True
                break
        if not placed:
            clusters.append({"rep": r, "annset": {aid}})
    return [c["rep"] for c in clusters if len(c["annset"]) >= min_agree]


def union_instance_masks(mask_utils, np_, document_images, by_file, annotations, file_name,
                         min_agree=1):
    """Consensus/union RLEs -> list of full-res binary masks (training targets)."""
    masks = []
    for r in union_rles(mask_utils, np_, document_images, by_file, annotations, file_name,
                        min_agree=min_agree):
        m = mask_utils.decode(r)
        if m.ndim == 3:
            m = m[..., 0]
        if int(m.sum()) >= MIN_AREA:
            masks.append(m.astype(np_.uint8))
    return masks


def _target_min_agree(target):
    """Map a target/eval-gt name to the union min_agree level ('union'->1, 'consensus2'->2)."""
    return {"union": 1, "consensus2": 2}.get(target, 1)


def make_dataset(torch, cv2, mask_utils, document, image_dir, files, imgsz, target="single"):
    from torch.utils.data import Dataset

    images = {str(r["id"]): r for r in document["images"]}
    annotations = v52.load_annotations(document)
    by_file = v52.canonical_split(document)[2]
    recs = v52.records_for(files, by_file, images, annotations)
    use_union = target in ("union", "consensus2")
    min_agree = _target_min_agree(target)

    class MRCNNSet(Dataset):
        def __len__(self):
            return len(recs)

        def __getitem__(self, i):
            name, image_id = recs[i]
            info = images[image_id]
            gray = cv2.imread(str(image_dir / name), cv2.IMREAD_GRAYSCALE)
            if imgsz != FULL:
                img = cv2.resize(gray, (imgsz, imgsz), interpolation=cv2.INTER_AREA)
            else:
                img = gray
            x = torch.from_numpy(img.astype(np.float32) / 255.0)[None].repeat(3, 1, 1)

            if use_union:
                masks_full = union_instance_masks(mask_utils, np, images, by_file,
                                                  annotations, name, min_agree=min_agree)
            else:
                masks_full = instance_masks(mask_utils, info, annotations[image_id], np)
            if imgsz != FULL:
                masks_r = [cv2.resize(m, (imgsz, imgsz), interpolation=cv2.INTER_NEAREST)
                           for m in masks_full]
            else:
                masks_r = masks_full
            masks_r = [m for m in masks_r if int(m.sum()) > 0]
            if not masks_r:  # degenerate guard; Mask R-CNN needs >=1 box
                masks_r = [np.ones((imgsz, imgsz), np.uint8)]
            boxes = masks_to_boxes(np, masks_r)
            target = {
                "boxes": torch.as_tensor(boxes, dtype=torch.float32),
                "labels": torch.ones((len(masks_r),), dtype=torch.int64),
                "masks": torch.as_tensor(np.stack(masks_r), dtype=torch.uint8),
                "image_id": torch.tensor([i]),
            }
            return x, target

    return MRCNNSet(), len(recs)


def build_model(torch, imgsz):
    from torchvision.models.detection import (
        MaskRCNN_ResNet50_FPN_V2_Weights,
        maskrcnn_resnet50_fpn_v2,
    )
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1)
    # 2 classes: background + filament
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, 2)
    in_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_mask, 256, 2)
    model.transform.min_size = (imgsz,)
    model.transform.max_size = imgsz
    # allow many detections — filaments can be numerous (recall matters)
    model.roi_heads.detections_per_img = 300
    return model


# ------------------------------------------------------------- train
def collate(batch):
    return tuple(zip(*batch, strict=True))


def train(args):
    import cv2
    import torch
    from pycocotools import mask as mask_utils
    from torch.utils.data import DataLoader

    if not torch.cuda.is_available():
        raise RuntimeError("V56 requires CUDA")
    torch.manual_seed(20261003)
    device = torch.device("cuda:0")
    document = json.loads(args.annotations.read_text())
    fit, _tune, _by, _ = v52.canonical_split(document)

    dataset, n_fit = make_dataset(torch, cv2, mask_utils, document, args.images, fit, args.imgsz,
                                  target=args.target)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                        collate_fn=collate, pin_memory=True)
    model = build_model(torch, args.imgsz).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler()
    best_path = args.work / "weights" / "v56_mrcnn_best.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        agg, n = 0.0, 0
        for images, targets in loader:
            images = [im.to(device, non_blocking=True) for im in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                losses = model(images, targets)
                loss = sum(losses.values())
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            agg += float(loss)
            n += 1
        sched.step()
        row = {"epoch": epoch, "loss": agg / max(n, 1), "lr": opt.param_groups[0]["lr"]}
        history.append(row)
        print(json.dumps(row), flush=True)
        torch.save({"model": model.state_dict(), "imgsz": args.imgsz, "epoch": epoch}, best_path)

    v52.write_json(args.output, {
        "status": "train_complete", "mode": "train", "experiment": "v56_maskrcnn",
        "target": args.target,
        "n_fit_files": n_fit, "epochs": args.epochs, "imgsz": args.imgsz, "lr": args.lr,
        "weights": str(best_path), "weights_sha256": v52.sha256(best_path),
        "history": history, "reads_tune": False, "reads_test": False,
    })
    print(json.dumps({"train_complete": True, "n_fit_files": n_fit, "weights": str(best_path)}))


# ------------------------------------------------------------- tune (project PQ)
def pq_counts_for_image(np_, mask_utils, pred_rles, gt_rles):
    if not pred_rles and not gt_rles:
        return 0.0, 0, 0, 0
    if not pred_rles:
        return 0.0, 0, 0, len(gt_rles)
    if not gt_rles:
        return 0.0, 0, len(pred_rles), 0
    iou = np_.asarray(mask_utils.iou(pred_rles, gt_rles, [0] * len(gt_rles)), dtype=float)
    hits = np_.argwhere(iou > 0.5)
    ordered = sorted(hits.tolist(), key=lambda pg: (-iou[pg[0], pg[1]], pg))
    up, ug, matched = set(), set(), []
    for p, g in ordered:
        if p not in up and g not in ug:
            up.add(p)
            ug.add(g)
            matched.append(float(iou[p, g]))
    tp = len(up)
    return float(sum(matched)), tp, len(pred_rles) - tp, len(gt_rles) - len(ug)


def disjoint_rles(np_, cv2, mask_utils, masks, scores, imgsz):
    """Upscale masks to FULL, enforce disjoint pixel ownership by descending score."""
    order = np_.argsort(scores)[::-1]
    owned = np_.zeros((FULL, FULL), np_.uint8)
    rles = []
    for idx in order:
        m = masks[idx]
        if imgsz != FULL:
            m = cv2.resize(m.astype(np_.float32), (FULL, FULL), interpolation=cv2.INTER_LINEAR)
            m = (m >= 0.5).astype(np_.uint8)
        m = m & (~owned & 1)
        if int(m.sum()) < MIN_AREA:
            continue
        owned |= m
        rles.append(mask_utils.encode(np_.asfortranarray(m)))
    return rles


def tune(args):
    import cv2
    import torch
    from pycocotools import mask as mask_utils

    device = torch.device("cuda:0")
    document = json.loads(args.annotations.read_text())
    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    imgsz = ckpt.get("imgsz", args.imgsz)
    model = build_model(torch, imgsz).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    images = {str(r["id"]): r for r in document["images"]}
    annotations = v52.load_annotations(document)
    _fit, tune_files, by_file, _ = v52.canonical_split(document)
    recs = v52.records_for(sorted(tune_files), by_file, images, annotations)

    # cache raw predictions once, then sweep score threshold cheaply
    per_image = []
    with torch.no_grad():
        for name, image_id in recs:
            info = images[image_id]
            gray = cv2.imread(str(args.images / name), cv2.IMREAD_GRAYSCALE)
            img = (cv2.resize(gray, (imgsz, imgsz), interpolation=cv2.INTER_AREA)
                   if imgsz != FULL else gray)
            x = torch.from_numpy(img.astype(np.float32) / 255.0)[None].repeat(3, 1, 1).to(device)
            with torch.cuda.amp.autocast():
                out = model([x])[0]
            scores = out["scores"].cpu().numpy()
            masks = (out["masks"][:, 0].cpu().numpy() >= MASK_THRESH).astype(np.uint8)
            if args.eval_gt in ("union", "consensus2"):
                gt = union_rles(mask_utils, np, images, by_file, annotations, name,
                                min_agree=_target_min_agree(args.eval_gt))
            else:
                gt = [v52.annotation_rle(mask_utils, info, e) for e in annotations[image_id]]
            gt = [r for r in gt if mask_utils.area(r) > 0]
            per_image.append((scores, masks, gt))

    cells = []
    for st in SCORE_SWEEP:
        iou_sum = tp = fp = fn = 0
        for scores, masks, gt in per_image:
            keep = scores >= st
            km, ks = masks[keep], scores[keep]
            rles = disjoint_rles(np, cv2, mask_utils, km, ks, imgsz) if len(km) else []
            s, i_tp, i_fp, i_fn = pq_counts_for_image(np, mask_utils, rles, gt)
            iou_sum += s
            tp += i_tp
            fp += i_fp
            fn += i_fn
        denom = tp + 0.5 * (fp + fn)
        cells.append({"score_thresh": st, "pq": float(iou_sum / denom) if denom else 0.0,
                      "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
                      "tp": tp, "fp": fp, "fn": fn})
    best = max(cells, key=lambda c: c["pq"])
    v52.write_json(args.output, {
        "status": "tune_complete", "mode": "tune", "experiment": "v56_maskrcnn",
        "eval_gt": args.eval_gt,
        "cells": cells, "best": best, "v17_tune_pq": V17_TUNE_PQ,
        "beats_v17": best["pq"] > V17_TUNE_PQ, "imgsz": imgsz, "mask_thresh": MASK_THRESH,
        "weights": str(args.weights), "reads_tune": True, "reads_test": False,
    })
    print(json.dumps({"best": best, "v17_tune_pq": V17_TUNE_PQ,
                      "beats_v17": best["pq"] > V17_TUNE_PQ}, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["train", "tune"], required=True)
    ap.add_argument("--annotations", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--weights", type=Path)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--target", choices=["single", "union", "consensus2"], default="single",
                    help="train GT: single, union(>=1 annotator), or consensus2(>=2)")
    ap.add_argument("--eval-gt", choices=["single", "union", "consensus2"], default="single",
                    dest="eval_gt",
                    help="tune GT to score against: single max-annotator or union of annotators")
    args = ap.parse_args()

    if v52.sha256(args.annotations) != ANNOTATION_SHA256:
        raise RuntimeError("V56 annotation identity changed")
    if args.mode == "train":
        train(args)
    else:
        if not args.weights:
            raise SystemExit("--weights required for --mode tune")
        tune(args)


if __name__ == "__main__":
    main()

