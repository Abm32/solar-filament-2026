"""V60A: native-resolution TILED detector + proposal-oracle recall test.

Diagnostic motive (diagnose_v59_0_recall): ~310/1134 tune GT are NEVER proposed by V17 at
any confidence (IoU>0.5) -> an upstream REPRESENTATION/proposal bottleneck, not score/
topology/label-noise. Small/thin filaments lose usable pixels inside the full 2048 disk.
TILING crops the native image into overlapping 768 windows so a small filament occupies
many more network pixels (SAHI-style sliced detection).

STRICT GATE (prove new proposals BEFORE building fusion): run the tile detector on tune,
map detections to global coords, compute proposal-ORACLE recall (ignore confidence) for
V17-pool / tile-pool / UNION at IoU 0.5/0.4/0.3.
  union@0.5 <=0.75 kill · 0.75-0.78 modest · 0.78-0.82 real · >0.82 very strong -> fusion.
Also classify tile-ONLY newly-found GT (area/elongation/contrast/disk-r).

OOF: fold k detector trained on FIT minus quarter k (train_v58_fold fold split reused),
tiles generated from those images; evaluated on tune (never in any fold) -> honest.
Firewall: FIT-only training, tune labels for scoring only, no test, no submission.

  CUDA_VISIBLE_DEVICES=N python scripts/v60_tiling.py train --fold K --annotations <ann> \
      --images <img> --work outputs/v60_tile_K --epochs 20 --tile 768 --overlap 0.3
  CUDA_VISIBLE_DEVICES=N python scripts/v60_tiling.py oracle --weights <...> \
      --annotations <ann> --images <img> --out outputs/v60_oracle.json --tile 768 --overlap 0.3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import reproduce_v17_from_cache as v50  # noqa: E402
import train_v52_cldice_semantic as v52  # noqa: E402
import train_v56_maskrcnn as v56  # noqa: E402
import train_v58_fold as fold  # noqa: E402

FULL = 2048
SMALL = 512
MIN_INST_AREA = 20


def tile_origins(tile, overlap):
    stride = int(tile * (1 - overlap))
    xs = list(range(0, FULL - tile + 1, stride))
    if xs[-1] != FULL - tile:
        xs.append(FULL - tile)
    return xs


# ------------------------------------------------------------- training
def make_tile_dataset(torch, cv2, mask_utils, document, image_dir, files, tile, overlap):
    from torch.utils.data import Dataset

    images = {str(r["id"]): r for r in document["images"]}
    annotations = v52.load_annotations(document)
    by_file = v52.canonical_split(document)[2]
    recs = v52.records_for(files, by_file, images, annotations)
    origins = tile_origins(tile, overlap)
    # enumerate (file, image_id, ox, oy)
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
                r = v52.annotation_rle(mask_utils, info, ann)
                m = mask_utils.decode(r)
                if m.ndim == 3:
                    m = m[..., 0]
                sub = m[oy:oy + tile, ox:ox + tile]
                if int(sub.sum()) < MIN_INST_AREA:
                    continue
                ys, xs = np.nonzero(sub)
                masks_t.append(sub.astype(np.uint8))
                boxes.append([float(xs.min()), float(ys.min()),
                              float(xs.max()) + 1, float(ys.max()) + 1])
            if not masks_t:
                masks_t = [np.ones((tile, tile), np.uint8)]
                boxes = [[0.0, 0.0, float(tile), float(tile)]]
            target = {"boxes": torch.as_tensor(boxes, dtype=torch.float32),
                      "labels": torch.ones((len(masks_t),), dtype=torch.int64),
                      "masks": torch.as_tensor(np.stack(masks_t), dtype=torch.uint8),
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
    ds, n = make_tile_dataset(torch, cv2, mask_utils, document, args.images, train_files,
                              args.tile, args.overlap)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                        collate_fn=v56.collate, pin_memory=True)
    model = v56.build_model(torch, args.tile).to(device)
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=args.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler()
    best = args.work / "weights" / "v60_tile_best.pt"
    best.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        agg = nb = 0
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
        print(json.dumps({"fold": args.fold, "epoch": epoch, "loss": agg / max(nb, 1)}),
              flush=True)
        torch.save({"model": model.state_dict(), "tile": args.tile, "epoch": epoch,
                    "fold": args.fold}, best)
    v52.write_json(args.output, {"status": "train_complete", "fold": args.fold,
                                 "n_tiles": n, "tile": args.tile, "overlap": args.overlap,
                                 "weights": str(best), "reads_tune": False, "reads_test": False})
    print(json.dumps({"v60_tile_train_complete": args.fold, "n_tiles": n}))


# ------------------------------------------------------------- oracle recall
def to_small(cv2, m):
    if m.ndim == 3:
        m = m[..., 0]
    return cv2.resize(m, (SMALL, SMALL), interpolation=cv2.INTER_NEAREST).astype(bool)


def tile_detect_global(torch, cv2, np_, model, device, gray, tile, overlap, conf=0.05):
    """Run model on overlapping tiles; return list of global-res small bool masks."""
    out_masks = []
    for ox in tile_origins(tile, overlap):
        for oy in tile_origins(tile, overlap):
            crop = gray[oy:oy + tile, ox:ox + tile]
            xt = torch.from_numpy(crop.astype(np.float32) / 255.0)[None].repeat(3, 1, 1).to(device)
            with torch.no_grad(), torch.cuda.amp.autocast():
                o = model([xt])[0]
            keep = o["scores"].cpu().numpy() >= conf
            tm = (o["masks"][:, 0].cpu().numpy()[keep] >= 0.5).astype(np.uint8)
            for m in tm:
                full = np.zeros((FULL, FULL), np.uint8)
                full[oy:oy + tile, ox:ox + tile] = m
                if int(full.sum()) >= MIN_INST_AREA:
                    out_masks.append(cv2.resize(full, (SMALL, SMALL),
                                                interpolation=cv2.INTER_NEAREST).astype(bool))
    return out_masks


def _best_iou(np_, pool, gf, n_gt):
    if not pool:
        return np_.zeros(n_gt)
    pf = np_.stack([p.ravel() for p in pool]).astype(np_.float32)
    inter = pf @ gf.T
    iou = inter / (pf.sum(1)[:, None] + gf.sum(1)[None, :] - inter + 1e-9)
    return iou.max(0)


def oracle(args):
    import cv2
    import torch
    from pycocotools import mask as mask_utils

    device = torch.device("cuda:0")
    rt = v50.runtime_state()
    core, tta, base = rt["core"], rt["tta"], rt["base"]
    _, tune, _, _, by_file = rt["split"]

    tile_models = []
    for wk in args.weights:
        ck = torch.load(wk, map_location=device, weights_only=False)
        m = v56.build_model(torch, ck.get("tile", args.tile)).to(device)
        m.load_state_dict(ck["model"])
        m.eval()
        tile_models.append(m)

    thrs = (0.5, 0.4, 0.3)
    acc = {t: {"v17": [0, 0], "tile": [0, 0], "union": [0, 0]} for t in thrs}
    tile_only = []
    for name in sorted(tune):
        gray = cv2.imread(str(base.CFG.train_dir / name), cv2.IMREAD_GRAYSCALE)
        # V17 pool (small masks)
        views = v50.read_signed_views(name, "train", "mr1_iou030", v50.DETECTORS["seed1"])
        props, scores = tta.cluster_views(views, v50.POLICY, v50.MATCH_IOU)
        v17 = []
        if props:
            masks = [mask_utils.decode(r) for r in props]
            bounds, _, probs = core.crop_probabilities(base, rt["model"], gray, masks, True)
            rles, _sc = core.clean_refined_predictions(masks, scores, bounds, probs, 0.45, v50.MODE)
            v17 = [to_small(cv2, mask_utils.decode(r)) for r in rles]
        # tile pool (union of all fold detectors on tiles)
        tilep = []
        for m in tile_models:
            tilep += tile_detect_global(torch, cv2, np, m, device, gray, args.tile, args.overlap)
        for image_id in sorted(by_file[name]):
            info = rt["images"][image_id]
            gts, feats = [], []
            for e in rt["annotations"].get(image_id, []):
                r = core.annotation_rle(info, e)
                if mask_utils.area(r) <= 0:
                    continue
                gm = to_small(cv2, mask_utils.decode(r))
                gts.append(gm)
                ys, xs = np.nonzero(gm)
                w = (xs.max() - xs.min() + 1) if len(xs) else 1
                h = (ys.max() - ys.min() + 1) if len(ys) else 1
                feats.append({"area": int(gm.sum()), "elong": max(w, h) / max(min(w, h), 1)})
            if not gts:
                continue
            gf = np.stack([g.ravel() for g in gts]).astype(np.float32)
            bv = _best_iou(np, v17, gf, len(gts))
            bt = _best_iou(np, tilep, gf, len(gts))
            bu = np.maximum(bv, bt)
            for t in thrs:
                acc[t]["v17"][0] += int((bv >= t).sum())
                acc[t]["v17"][1] += len(gts)
                acc[t]["tile"][0] += int((bt >= t).sum())
                acc[t]["tile"][1] += len(gts)
                acc[t]["union"][0] += int((bu >= t).sum())
                acc[t]["union"][1] += len(gts)
            # tile-only newly found @0.5
            for gi in range(len(gts)):
                if bt[gi] >= 0.5 and bv[gi] < 0.5:
                    tile_only.append(feats[gi])
        print(f"oracle {name} v17={len(v17)} tile={len(tilep)}", flush=True)

    def r(p):
        return round(p[0] / p[1], 4) if p[1] else 0.0
    out = {f"iou{t}": {k: r(acc[t][k]) for k in ("v17", "tile", "union")} for t in thrs}
    u05 = out["iou0.5"]["union"]
    gate = ("kill" if u05 <= 0.75 else "modest" if u05 <= 0.78 else "real"
            if u05 <= 0.82 else "very_strong")
    to_area = [t["area"] for t in tile_only]
    to_elong = [t["elong"] for t in tile_only]
    report = {"oracle_recall": out, "union_iou0.5": u05, "gate": gate,
              "n_tile_only_found": len(tile_only),
              "tile_only_mean_area": float(np.mean(to_area)) if to_area else 0,
              "tile_only_mean_elong": float(np.mean(to_elong)) if to_elong else 0,
              "reads_test": False}
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--fold", type=int, required=True, choices=range(4))
    tr.add_argument("--annotations", type=Path, required=True)
    tr.add_argument("--images", type=Path, required=True)
    tr.add_argument("--work", type=Path, required=True)
    tr.add_argument("--output", type=Path, required=True)
    tr.add_argument("--epochs", type=int, default=20)
    tr.add_argument("--batch", type=int, default=2)
    tr.add_argument("--workers", type=int, default=4)
    tr.add_argument("--tile", type=int, default=768)
    tr.add_argument("--overlap", type=float, default=0.3)
    tr.add_argument("--lr", type=float, default=5e-3)
    oc = sub.add_parser("oracle")
    oc.add_argument("--weights", type=Path, nargs="+", required=True)
    oc.add_argument("--annotations", type=Path, required=True)
    oc.add_argument("--images", type=Path, required=True)
    oc.add_argument("--out", type=Path, required=True)
    oc.add_argument("--tile", type=int, default=768)
    oc.add_argument("--overlap", type=float, default=0.3)
    args = ap.parse_args()
    if v52.sha256(args.annotations) != v56.ANNOTATION_SHA256:
        raise RuntimeError("annotation identity changed")
    if args.cmd == "train":
        train(args)
    else:
        oracle(args)


if __name__ == "__main__":
    main()
