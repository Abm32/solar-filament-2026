"""V60D: train the native-res TILE detector to emit ONE strong proposal per filament.

Evidence (V60B/C): tiling lifts oracle recall 0.728 -> 0.86, but the detector emits a cloud of
near-duplicates per filament and every post-hoc selection (confidence NMS, consensus rank,
consensus mask) failed. V60D changes the TRAINING signal instead.

Mechanism (Mask R-CNN, torchvision 0.19 RoIHeads; box + mask supervision left UNTOUCHED):
  * The baseline makes EVERY proposal with IoU>=0.5 to a GT a classification positive, so the
    classifier is explicitly trained to score all ~N overlapping proposals of one filament high.
  * Here, per GT, the best-IoU RPN proposal (plus the exact GT box that torchvision appends) is
    the CANONICAL positive; the other IoU>=0.5 proposals of the SAME GT are DUPLICATES.
  * Duplicates keep their box-regression and mask targets (segmentation objective intact) but
    their CLASSIFICATION target becomes background, scaled by --dup-w:
        dup_w = 1.0  full one-to-one negatives              (D1)
        dup_w = 0.3  conservative duplicate-aware penalty   (D2)
        dup_w = 0.0  duplicates ignored by the cls loss     (D3)
    no dup handling at all                                   (D0 control)
  * Proposals matched to a DIFFERENT GT are canonical for that GT -> adjacent filaments are
    preserved. Low overlap with every GT stays an ordinary negative.

Also fixes a baseline artefact: v60_tiling.make_tile_dataset turned EMPTY tiles (28% of fold-0
tiles) into a full-tile "filament" box+mask. Here empty tiles are true empty targets. D0 isolates
this fix so it cannot be confused with the one-to-one effect.

Honest limitation: the classifier scores each ROI independently (no inter-proposal
communication, unlike DETR), so it can only express "one-to-one" as an IoU-sharpened score.
The census (scripts/census_v60d.py) measures whether that is enough.

Firewall: FIT-only (fold.fold_train_files); tune/test never read.

  CUDA_VISIBLE_DEVICES=N python scripts/train_v60d_onetoone.py --fold 0 --dup-w 1.0 \
      --annotations <ann> --images <img> --work outputs/v60d_d1 --output outputs/v60d_d1/train.json
"""
from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import train_v52_cldice_semantic as v52  # noqa: E402
import train_v56_maskrcnn as v56  # noqa: E402
import train_v58_fold as fold  # noqa: E402
import v60_tiling as T  # noqa: E402

# side channel: select_training_samples -> patched fastrcnn_loss (one process per GPU)
_SIDE: dict = {}


def duplicate_mask(torch, box_iou, gt_boxes, proposals, matched_idxs, labels, margin=0.0):
    """Bool (P,) marking duplicate foreground proposals for ONE image.

    proposals: (P,4) = RPN proposals followed by the G appended GT boxes.
    For each GT g: among fg RPN proposals matched to g (labels==1), the highest-IoU one is
    canonical (ties/within `margin` of the best also canonical); every other is a duplicate.
    Appended GT boxes are never duplicates.
    """
    p = proposals.shape[0]
    g = gt_boxes.shape[0]
    dup = torch.zeros(p, dtype=torch.bool, device=proposals.device)
    if g == 0:
        return dup
    n_rpn = p - g
    iou = box_iou(gt_boxes, proposals)                               # (G, P)
    m_iou = iou[matched_idxs, torch.arange(p, device=proposals.device)]
    fg = labels == 1
    for gi in range(g):
        elig = fg & (matched_idxs == gi)
        elig[n_rpn:] = False
        idx = torch.where(elig)[0]
        if idx.numel() <= 1:
            continue
        vals = m_iou[idx]
        canon = vals >= vals.max() - margin
        canon[vals.argmax()] = True
        dup[idx[~canon]] = True
    return dup


def make_select_training_samples(torch, box_iou, dup_w, margin):
    """Replacement for RoIHeads.select_training_samples (torchvision 0.19.1 logic + flags)."""

    def select_training_samples(self, proposals, targets):
        self.check_targets(targets)
        dtype = proposals[0].dtype
        device = proposals[0].device
        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]
        proposals = self.add_gt_proposals(proposals, gt_boxes)
        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)
        dups = [duplicate_mask(torch, box_iou, gt_boxes[i], proposals[i],
                               matched_idxs[i], labels[i], margin)
                for i in range(len(proposals))]
        sampled_inds = self.subsample(labels)
        matched_gt_boxes, cls_lab, cls_w = [], [], []
        n_pos = n_dup = 0
        for i in range(len(proposals)):
            s = sampled_inds[i]
            proposals[i] = proposals[i][s]
            labels[i] = labels[i][s]
            matched_idxs[i] = matched_idxs[i][s]
            d = dups[i][s]
            gt_in = gt_boxes[i]
            if gt_in.numel() == 0:
                gt_in = torch.zeros((1, 4), dtype=dtype, device=device)
            matched_gt_boxes.append(gt_in[matched_idxs[i]])
            cl = labels[i].clone()
            cw = torch.ones_like(cl, dtype=torch.float32)
            cl[d] = 0                                   # duplicate -> background for cls only
            cw[d] = dup_w
            cls_lab.append(cl)
            cls_w.append(cw)
            n_pos += int((labels[i] > 0).sum())
            n_dup += int(d.sum())
        _SIDE["cls"], _SIDE["w"] = cls_lab, cls_w
        _SIDE["n_pos"] = _SIDE.get("n_pos", 0) + n_pos
        _SIDE["n_dup"] = _SIDE.get("n_dup", 0) + n_dup
        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, matched_idxs, labels, regression_targets

    return select_training_samples


def make_patched_fastrcnn_loss(torch, functional, original):
    """fastrcnn_loss with per-proposal cls targets/weights from the side channel.

    With all weights 1 and cls targets == labels this equals the original (mean over N).
    Box loss is identical to the original (uses the UNMODIFIED labels).
    """

    def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
        if "cls" not in _SIDE:
            return original(class_logits, box_regression, labels, regression_targets)
        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)
        cls_lab = torch.cat(_SIDE["cls"], dim=0)
        cls_w = torch.cat(_SIDE["w"], dim=0)
        ce = functional.cross_entropy(class_logits, cls_lab, reduction="none")
        classification_loss = (ce * cls_w).sum() / labels.numel()
        pos = torch.where(labels > 0)[0]
        labels_pos = labels[pos]
        n, _ = class_logits.shape
        box_regression = box_regression.reshape(n, box_regression.size(-1) // 4, 4)
        box_loss = functional.smooth_l1_loss(box_regression[pos, labels_pos],
                                             regression_targets[pos], beta=1 / 9,
                                             reduction="sum") / labels.numel()
        return classification_loss, box_loss

    return fastrcnn_loss


def install_one_to_one(torch, model, dup_w, margin):
    """Patch a torchvision MaskRCNN in place. Returns nothing; idempotent per process."""
    import torch.nn.functional as functional
    from torchvision.models.detection import roi_heads as rh
    from torchvision.ops import box_iou
    if not hasattr(rh, "_v60d_orig_fastrcnn_loss"):
        rh._v60d_orig_fastrcnn_loss = rh.fastrcnn_loss
    rh.fastrcnn_loss = make_patched_fastrcnn_loss(torch, functional, rh._v60d_orig_fastrcnn_loss)
    fn = make_select_training_samples(torch, box_iou, dup_w, margin)
    model.roi_heads.select_training_samples = types.MethodType(fn, model.roi_heads)


def make_dataset(torch, cv2, mask_utils, document, image_dir, files, tile, overlap,
                 legacy_empty=False):
    from torch.utils.data import Dataset

    images = {str(r["id"]): r for r in document["images"]}
    annotations = v52.load_annotations(document)
    by_file = v52.canonical_split(document)[2]
    recs = v52.records_for(files, by_file, images, annotations)
    origins = T.tile_origins(tile, overlap)
    index = [(n, iid, ox, oy) for (n, iid) in recs for ox in origins for oy in origins]

    class TileSet(Dataset):
        def __len__(self):
            return len(index)

        def __getitem__(self, i):
            name, image_id, ox, oy = index[i]
            info = images[image_id]
            gray = cv2.imread(str(image_dir / name), cv2.IMREAD_GRAYSCALE)
            crop = gray[oy:oy + tile, ox:ox + tile]
            x = torch.from_numpy(crop.astype(np.float32) / 255.0)[None].repeat(3, 1, 1)
            masks_t, boxes = [], []
            for ann in annotations[image_id]:
                m = mask_utils.decode(v52.annotation_rle(mask_utils, info, ann))
                if m.ndim == 3:
                    m = m[..., 0]
                sub = m[oy:oy + tile, ox:ox + tile]
                if int(sub.sum()) < T.MIN_INST_AREA:
                    continue
                ys, xs = np.nonzero(sub)
                masks_t.append(sub.astype(np.uint8))
                boxes.append([float(xs.min()), float(ys.min()),
                              float(xs.max()) + 1, float(ys.max()) + 1])
            if not masks_t and legacy_empty:                 # baseline artefact, kept opt-in
                masks_t = [np.ones((tile, tile), np.uint8)]
                boxes = [[0.0, 0.0, float(tile), float(tile)]]
            if masks_t:
                target = {"boxes": torch.as_tensor(boxes, dtype=torch.float32),
                          "labels": torch.ones((len(masks_t),), dtype=torch.int64),
                          "masks": torch.as_tensor(np.stack(masks_t), dtype=torch.uint8),
                          "image_id": torch.tensor([i])}
            else:                                            # TRUE empty tile
                target = {"boxes": torch.zeros((0, 4), dtype=torch.float32),
                          "labels": torch.zeros((0,), dtype=torch.int64),
                          "masks": torch.zeros((0, tile, tile), dtype=torch.uint8),
                          "image_id": torch.tensor([i])}
            return x, target

    return TileSet(), len(index)


def train(args):
    import cv2
    import torch
    from pycocotools import mask as mask_utils
    from torch.utils.data import DataLoader

    torch.manual_seed(20261004 + args.fold)
    device = torch.device("cuda:0")
    document = json.loads(args.annotations.read_text())
    train_files, _ = fold.fold_train_files(document, args.fold)
    ds, n = make_dataset(torch, cv2, mask_utils, document, args.images, train_files,
                         args.tile, args.overlap, legacy_empty=args.legacy_empty)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                        collate_fn=v56.collate, pin_memory=True)
    model = v56.build_model(torch, args.tile).to(device)
    if args.dup_w is not None:
        install_one_to_one(torch, model, args.dup_w, args.margin)
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=args.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler()
    best = args.work / "weights" / "v60_tile_best.pt"
    best.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        agg = nb = 0
        _SIDE["n_pos"] = _SIDE["n_dup"] = 0
        for imgs, tgts in loader:
            imgs = [im.to(device, non_blocking=True) for im in imgs]
            tgts = [{k: v.to(device) for k, v in t.items()} for t in tgts]
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                loss = sum(model(imgs, tgts).values())
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            agg += float(loss)
            nb += 1
        sched.step()
        dup_frac = _SIDE["n_dup"] / max(_SIDE["n_pos"], 1)
        print(json.dumps({"fold": args.fold, "epoch": epoch, "loss": agg / max(nb, 1),
                          "dup_w": args.dup_w, "sampled_pos_dup_frac": round(dup_frac, 4)}),
              flush=True)
        torch.save({"model": model.state_dict(), "tile": args.tile, "epoch": epoch,
                    "fold": args.fold, "dup_w": args.dup_w, "margin": args.margin}, best)
    v52.write_json(args.output, {"status": "train_complete", "fold": args.fold, "n_tiles": n,
                                 "tile": args.tile, "overlap": args.overlap,
                                 "dup_w": args.dup_w, "margin": args.margin,
                                 "legacy_empty": args.legacy_empty, "weights": str(best),
                                 "reads_tune": False, "reads_test": False})
    print(json.dumps({"v60d_train_complete": args.fold, "n_tiles": n, "dup_w": args.dup_w}))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold", type=int, required=True, choices=range(4))
    ap.add_argument("--annotations", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--dup-w", type=float, default=None, dest="dup_w",
                    help="None = D0 control (no duplicate handling); 1.0/0.3/0.0 = D1/D2/D3")
    ap.add_argument("--margin", type=float, default=0.0,
                    help="IoU slack: proposals within margin of the best stay canonical")
    ap.add_argument("--legacy-empty", action="store_true", dest="legacy_empty")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tile", type=int, default=768)
    ap.add_argument("--overlap", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=5e-3)
    args = ap.parse_args()
    if v52.sha256(args.annotations) != v56.ANNOTATION_SHA256:
        raise RuntimeError("annotation identity changed")
    train(args)


if __name__ == "__main__":
    main()
