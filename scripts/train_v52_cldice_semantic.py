"""V52: clDice-trained semantic U-Net as a NEW orthogonal source.

Motivation (SESSION_RESUME_2026-10-01 sec.4 + V51 addendum): every YOLO-family
source (V42/V44/V45/V49) and the V17 ensemble share CORRELATED false positives,
so NMS/agreement fusion never beats V17's 0.448703 (V51 gate failed: V44 is nested
inside V17). The one untried lever with a real mechanism is a *semantic* model with
a topology-preserving loss (clDice, arXiv 2003.07311): a different architecture AND
a different objective should produce FPs that are NOT correlated with the detector
family, which is the only way an ensemble can gain (E028).

This script trains a compact pure-torch U-Net (no new deps) with BCE + soft-Dice +
soft-clDice on the 566 canonical FIT files only, then evaluates standalone project
PQ on the held-out 141 TUNE files, and can dump per-image instance RLEs+scores so
the V51 fusion harness can test it as an extra source.

Modes:
  --mode train : train on fit files -> weights (reads_tune=False)
  --mode tune  : evaluate a checkpoint on the held-out tune split (project PQ),
                 sweep probability threshold; optionally dump proposal cache.

Firewall: tune labels are used ONLY in --mode tune for evaluation, NEVER in train.
ANNOTATION_SHA256 identity is checked before any data is touched. No test inference,
no CSV, no submission here (gating happens in the V51 fusion harness).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")

ANNOTATION_SHA256 = "5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1"
FULL = 2048          # native resolution / RLE canvas
TRAIN_RES = 1024     # U-Net training/inference resolution (T4-friendly, thin structures)
MIN_AREA = 20        # project convention for instance acceptance (full-res pixels)
PROB_SWEEP = (0.30, 0.40, 0.50, 0.60, 0.70)
V17_TUNE_PQ = 0.448703  # local bar to beat


# ---------------------------------------------------------------- utilities
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
    """Frozen project split: shuffle(sorted(files), Random(42)); tune=first 141."""
    import random

    by_file = defaultdict(list)
    images = {str(r["id"]): r for r in document["images"]}
    for image_id, info in images.items():
        by_file[info["file_name"]].append(image_id)
    files = sorted(by_file)
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


def load_annotations(document):
    annotations = defaultdict(list)
    for r in document["annotations"]:
        annotations[str(r["image_id"])].append(r)
    return annotations


# ---------------------------------------------------------------- model (pure torch)
def build_unet(torch_nn):
    nn = torch_nn

    class DoubleConv(nn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    class UNet(nn.Module):
        """Compact 4-level U-Net, 1->1 channel, base width 32."""

        def __init__(self, base=32):
            super().__init__()
            self.d1 = DoubleConv(1, base)
            self.d2 = DoubleConv(base, base * 2)
            self.d3 = DoubleConv(base * 2, base * 4)
            self.d4 = DoubleConv(base * 4, base * 8)
            self.bott = DoubleConv(base * 8, base * 16)
            self.pool = nn.MaxPool2d(2)
            self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
            self.u4 = DoubleConv(base * 16, base * 8)
            self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
            self.u3 = DoubleConv(base * 8, base * 4)
            self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
            self.u2 = DoubleConv(base * 4, base * 2)
            self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
            self.u1 = DoubleConv(base * 2, base)
            self.head = nn.Conv2d(base, 1, 1)

        def forward(self, x):
            import torch as _t

            c1 = self.d1(x)
            c2 = self.d2(self.pool(c1))
            c3 = self.d3(self.pool(c2))
            c4 = self.d4(self.pool(c3))
            b = self.bott(self.pool(c4))
            x = self.u4(_t.cat([self.up4(b), c4], 1))
            x = self.u3(_t.cat([self.up3(x), c3], 1))
            x = self.u2(_t.cat([self.up2(x), c2], 1))
            x = self.u1(_t.cat([self.up1(x), c1], 1))
            return self.head(x)

    return UNet


# ---------------------------------------------------------------- clDice loss
def soft_erode(torch, img):
    # min-filter via -maxpool(-x), separable 3x1 and 1x3
    import torch.nn.functional as functional

    p1 = -functional.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
    p2 = -functional.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
    return torch.min(p1, p2)


def soft_dilate(torch, img):
    import torch.nn.functional as functional

    return functional.max_pool2d(img, (3, 3), (1, 1), (1, 1))


def soft_open(torch, img):
    return soft_dilate(torch, soft_erode(torch, img))


def soft_skeleton(torch, img, iters=10):
    """Differentiable soft-skeleton (Shit et al. 2020, arXiv 2003.07311)."""
    img1 = soft_open(torch, img)
    skel = torch.relu(img - img1)
    for _ in range(iters):
        img = soft_erode(torch, img)
        img1 = soft_open(torch, img)
        delta = torch.relu(img - img1)
        skel = skel + torch.relu(delta - skel * delta)
    return skel


def cldice_loss(torch, prob, target, iters=10, smooth=1.0):
    """soft-clDice = 1 - 2 * (Tprec*Tsens)/(Tprec+Tsens)."""
    sk_pred = soft_skeleton(torch, prob, iters)
    sk_true = soft_skeleton(torch, target, iters)
    tprec = (torch.sum(sk_pred * target) + smooth) / (torch.sum(sk_pred) + smooth)
    tsens = (torch.sum(sk_true * prob) + smooth) / (torch.sum(sk_true) + smooth)
    return 1.0 - 2.0 * (tprec * tsens) / (tprec + tsens)


def dice_loss(torch, prob, target, smooth=1.0):
    num = 2.0 * torch.sum(prob * target) + smooth
    den = torch.sum(prob) + torch.sum(target) + smooth
    return 1.0 - num / den


def combined_loss(torch, functional, logits, target, cldice_w=0.5):
    prob = torch.sigmoid(logits)
    bce = functional.binary_cross_entropy_with_logits(logits, target)
    d = dice_loss(torch, prob, target)
    cl = cldice_loss(torch, prob, target)
    return bce + d + cldice_w * cl, {"bce": float(bce), "dice": float(d), "cldice": float(cl)}


# ---------------------------------------------------------------- data
def semantic_mask(np, cv2, mask_utils, info, anns, res):
    """Union of all instance masks for an image, rendered at res x res (float32 0/1)."""
    canvas = np.zeros((FULL, FULL), dtype=np.uint8)
    for ann in anns:
        m = mask_utils.decode(annotation_rle(mask_utils, info, ann))
        canvas |= m.astype(np.uint8)
    if res != FULL:
        canvas = cv2.resize(canvas, (res, res), interpolation=cv2.INTER_NEAREST)
    return canvas.astype(np.float32)


def read_gray(np, cv2, path, res):
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError(f"missing image {path}")
    if gray.shape != (FULL, FULL):
        raise RuntimeError(f"unexpected shape {gray.shape} for {path}")
    if res != FULL:
        gray = cv2.resize(gray, (res, res), interpolation=cv2.INTER_AREA)
    return gray.astype(np.float32) / 255.0


def make_dataset(torch, np, cv2, mask_utils, document, image_dir, files, res):
    from torch.utils.data import Dataset

    images = {str(r["id"]): r for r in document["images"]}
    annotations = load_annotations(document)
    recs = records_for(files, document_by_file(document), images, annotations)

    class FilamentSem(Dataset):
        def __len__(self):
            return len(recs)

        def __getitem__(self, i):
            name, image_id = recs[i]
            info = images[image_id]
            x = read_gray(np, cv2, image_dir / name, res)
            y = semantic_mask(np, cv2, mask_utils, info, annotations[image_id], res)
            return (
                torch.from_numpy(x)[None],
                torch.from_numpy(y)[None],
            )

    return FilamentSem(), len(recs)


def document_by_file(document):
    by_file = defaultdict(list)
    for r in document["images"]:
        by_file[r["file_name"]].append(str(r["id"]))
    return by_file


# ------------------------------------------------- semantic -> instances + PQ
def mask_to_instances(np, cv2, measure, prob_full, thresh):
    """Threshold the full-res prob map, connected-components -> disjoint instances.

    Returns (list_of_binary_uint8_masks, list_of_mean_prob_scores), already
    non-overlapping (connected components are disjoint by construction).
    """
    binary = prob_full >= thresh
    labels = measure.label(binary, connectivity=2)
    masks, scores = [], []
    for lab in range(1, int(labels.max()) + 1):
        comp = labels == lab
        area = int(comp.sum())
        if area < MIN_AREA:
            continue
        masks.append(comp.astype(np.uint8))
        scores.append(float(prob_full[comp].mean()))
    return masks, scores


def predict_prob_full(torch, np, cv2, model, device, gray_full, res, tta=True):
    """Return a FULL x FULL probability map (optionally 4-view flip-averaged)."""
    x = cv2.resize(gray_full, (res, res), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    axes_sets = ((), (1,), (0,), (0, 1)) if tta else ((),)
    acc = np.zeros((res, res), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for axes in axes_sets:
            v = np.flip(x, axes).copy() if axes else x
            t = torch.from_numpy(v)[None, None].to(device)
            p = torch.sigmoid(model(t))[0, 0].float().cpu().numpy()
            acc += np.flip(p, axes).copy() if axes else p
    acc /= len(axes_sets)
    if res != FULL:
        acc = cv2.resize(acc, (FULL, FULL), interpolation=cv2.INTER_LINEAR)
    return acc


def pq_counts_for_image(np, mask_utils, pred_masks, pred_scores, gt_rles):
    """Greedy one-to-one IoU>0.5 matching (project pq_eval semantics)."""
    if not pred_masks and not gt_rles:
        return 0.0, 0, 0, 0
    if not pred_masks:
        return 0.0, 0, 0, len(gt_rles)
    if not gt_rles:
        return 0.0, 0, len(pred_masks), 0
    pred_rles = [mask_utils.encode(np.asfortranarray(m)) for m in pred_masks]
    iou = np.asarray(mask_utils.iou(pred_rles, gt_rles, [0] * len(gt_rles)), dtype=float)
    hits = np.argwhere(iou > 0.5)
    ordered = sorted(hits.tolist(), key=lambda pr: (-iou[pr[0], pr[1]], pr))
    up, ut, matched = set(), set(), []
    for p, t in ordered:
        if p not in up and t not in ut:
            up.add(p)
            ut.add(t)
            matched.append(float(iou[p, t]))
    tp = len(up)
    fp = len(pred_masks) - tp
    fn = len(gt_rles) - len(ut)
    return float(sum(matched)), tp, fp, fn


def evaluate_tune(torch, np, cv2, measure, mask_utils, model, document, image_dir,
                  device, res, thresh, tta, dump_dir=None):
    images = {str(r["id"]): r for r in document["images"]}
    annotations = load_annotations(document)
    _fit, tune, by_file, _ = canonical_split(document)
    recs = records_for(tune, by_file, images, annotations)
    iou_sum = tp = fp = fn = 0
    matched_all = []
    for name, image_id in recs:
        info = images[image_id]
        gray = cv2.imread(str(image_dir / name), cv2.IMREAD_GRAYSCALE)
        prob_full = predict_prob_full(torch, np, cv2, model, device, gray.astype(np.float32),
                                      res, tta)
        pred_masks, pred_scores = mask_to_instances(np, cv2, measure, prob_full, thresh)
        gt_rles = [annotation_rle(mask_utils, info, r) for r in annotations[image_id]]
        s, i_tp, i_fp, i_fn = pq_counts_for_image(np, mask_utils, pred_masks, pred_scores, gt_rles)
        iou_sum += s
        tp += i_tp
        fp += i_fp
        fn += i_fn
        matched_all.append(s)
        if dump_dir is not None:
            _dump_proposals(np, mask_utils, dump_dir, name, pred_masks, pred_scores, thresh)
    denom = tp + 0.5 * (fp + fn)
    return {
        "threshold": thresh,
        "pq": float(iou_sum / denom) if denom else 0.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "tp": tp, "fp": fp, "fn": fn, "n_images": len(recs),
    }


def _dump_proposals(np, mask_utils, dump_dir, name, masks, scores, thresh):
    """Persist instance RLEs + scores for the V51 fusion harness (gzip json)."""
    rles = []
    for m in masks:
        rle = mask_utils.encode(np.asfortranarray(m))
        rles.append({"size": list(rle["size"]), "counts": rle["counts"].decode()})
    path = Path(dump_dir) / (name + ".json.gz")
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as stream:
        json.dump({"file": name, "threshold": thresh, "rles": rles, "scores": scores}, stream)


# ---------------------------------------------------------------- train loop
def train(args):
    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as functional
    from pycocotools import mask as mask_utils
    from torch.utils.data import DataLoader

    if not torch.cuda.is_available():
        raise RuntimeError("V52 requires CUDA")
    torch.manual_seed(20261002)
    device = torch.device("cuda:0")
    document = json.loads(args.annotations.read_text())
    fit, _tune, _, _ = canonical_split(document)

    dataset, n_fit = make_dataset(torch, np, cv2, mask_utils, document, args.images,
                                  fit, args.res)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                        drop_last=False, pin_memory=True)
    UNet = build_unet(torch.nn)
    model = UNet(base=args.base).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    best_path = args.work / "weights" / "v52_cldice_best.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        agg = {"total": 0.0, "bce": 0.0, "dice": 0.0, "cldice": 0.0, "n": 0}
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if args.fliplr and torch.rand(1).item() < 0.5:
                x = torch.flip(x, [3])
                y = torch.flip(y, [3])
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss, parts = combined_loss(torch, functional, logits, y, args.cldice_w)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            agg["total"] += float(loss)
            for k in ("bce", "dice", "cldice"):
                agg[k] += parts[k]
            agg["n"] += 1
        row = {"epoch": epoch, **{k: agg[k] / agg["n"] for k in ("total", "bce", "dice", "cldice")}}
        history.append(row)
        print(json.dumps(row), flush=True)
        torch.save({"model": model.state_dict(), "base": args.base, "res": args.res,
                    "epoch": epoch}, best_path)
    write_json(args.output, {
        "status": "train_complete", "mode": "train", "n_fit_files": n_fit,
        "epochs": args.epochs, "res": args.res, "base": args.base, "lr": args.lr,
        "cldice_w": args.cldice_w, "weights": str(best_path),
        "weights_sha256": sha256(best_path), "history": history,
        "reads_tune": False, "reads_test": False,
    })
    print(json.dumps({"train_complete": True, "n_fit_files": n_fit, "weights": str(best_path)}))


def tune(args):
    import cv2
    import numpy as np
    import torch
    from pycocotools import mask as mask_utils
    from skimage import measure

    device = torch.device("cuda:0")
    document = json.loads(args.annotations.read_text())
    ckpt = torch.load(args.weights, map_location=device)
    UNet = build_unet(torch.nn)
    model = UNet(base=ckpt.get("base", args.base)).to(device)
    model.load_state_dict(ckpt["model"])
    res = ckpt.get("res", args.res)

    cells = []
    for thr in PROB_SWEEP:
        dump = (args.dump_dir if (args.dump_dir and abs(thr - args.dump_thresh) < 1e-9) else None)
        cells.append(evaluate_tune(torch, np, cv2, measure, mask_utils, model, document,
                                   args.images, device, res, thr, not args.no_tta, dump))
    best = max(cells, key=lambda c: c["pq"])
    write_json(args.output, {
        "status": "tune_complete", "mode": "tune", "cells": cells, "best": best,
        "v17_tune_pq": V17_TUNE_PQ, "beats_v17_standalone": best["pq"] > V17_TUNE_PQ,
        "weights": str(args.weights), "res": res, "tta": not args.no_tta,
        "reads_tune": True, "reads_test": False,
    })
    print(json.dumps({"best": best, "v17_tune_pq": V17_TUNE_PQ,
                      "beats_v17_standalone": best["pq"] > V17_TUNE_PQ}, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["train", "tune"], required=True)
    ap.add_argument("--annotations", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--weights", type=Path)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--res", type=int, default=TRAIN_RES)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--cldice-w", type=float, default=0.5, dest="cldice_w")
    ap.add_argument("--fliplr", action="store_true")
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--dump-dir", type=Path, default=None)
    ap.add_argument("--dump-thresh", type=float, default=0.5, dest="dump_thresh")
    args = ap.parse_args()

    if sha256(args.annotations) != ANNOTATION_SHA256:
        raise RuntimeError("V52 annotation identity changed")

    if args.mode == "train":
        train(args)
    else:
        if not args.weights:
            raise SystemExit("--weights required for --mode tune")
        tune(args)


if __name__ == "__main__":
    main()
