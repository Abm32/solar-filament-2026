"""V60A: candidate-conditioned high-res MASK REFINER (segmentation-first, FP-safe).

Target (diagnose_v60_maskquality): 189 GT in IoU 0.3-0.5 are under-segmented (median
pred/GT area 0.686) — V17 localises them but the mask misses barbs/extensions and fails the
0.5 threshold. The refiner takes a candidate mask + a high-res 1-channel crop and outputs the
FULL filament mask, WITHOUT generating new standalone instances (FP-safe).

v1 result (S1/S2, under-segmented seeds only): learned "expand the seed", damaged 172/607
already-good V17 masks, best replacement rule = off. Root cause: the training distribution
never contained a seed that needed NO change.

v2 (this file) — MIXED seed distribution + explicit "correction needed" head:
  seed kind      prob   seed/GT area        meaning at inference
  identity       0.50   1.00 (exact GT)     good mask  -> preserve
  mild           0.15   0.85-0.95           slightly bad -> (label 0, leave alone)
  regime         0.30   0.59-0.79 (~0.69)   measured V17 under-seg -> correct/expand
  severe         0.05   0.30-0.50           bad -> expand/correct
  correction label = 1 iff IoU(seed, GT) < CORR_IOU_THR (0.8)  (measured, not by category)
Model: compact 2-ch U-Net [1-ch grayscale crop, seed] -> mask head (+ optional spine head)
+ a correction-needed logit from pooled bottleneck features (sees image AND seed).
Loss = 1.0*Dice + 0.5*Focal(mask) + spine_w*BCE(spine) + corr_w*BCE(correction).

Firewall: FIT-only (fold.fold_train_files), tune/test never read.

  CUDA_VISIBLE_DEVICES=N python scripts/train_v60a_refiner.py --fold 0 \
      --annotations <ann> --images <img> --work outputs/v60a_mix_f0 \
      --output outputs/v60a_mix_f0/train.json --spine-w 0 --corr-w 0.5 --crop 1024 --epochs 25
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
import train_v58_fold as fold  # noqa: E402

FULL = 2048
# refiner works at 256 (local crop resized to this); enough for thin detail in a local window
NET = 256

# seed-kind mixture (sums to 1 with severe = remainder)
P_IDENTITY = 0.50
P_MILD = 0.15
P_REGIME = 0.30
CORR_IOU_THR = 0.8          # label 1 ("needs correction") iff IoU(seed, GT) < this


def rasterize_spine(np_, cv2, spine_flat, ox, oy, scale, net):
    """Flat [x,y,...] GT spine -> soft heatmap in the crop frame (3px line, then blur)."""
    hm = np_.zeros((net, net), np_.uint8)
    pts = np_.array(spine_flat, dtype=np_.float32).reshape(-1, 2)
    pts = (pts - np_.array([ox, oy])) * scale
    pts = pts.astype(np_.int32)
    for i in range(len(pts) - 1):
        cv2.line(hm, tuple(pts[i]), tuple(pts[i + 1]), 1, 2)
    hm = cv2.GaussianBlur(hm.astype(np_.float32), (5, 5), 0)
    return (hm / (hm.max() + 1e-6)).astype(np_.float32)


def _undersegment_seed(np_, cv2, mt, target_frac, rng, spread=0.1):
    """Seed = under-segmented mask mimicking V17 (~target_frac area), trimming EXTENSIONS first.

    V17's under-segmentation misses barbs/extensions (the far, thin parts), not the core. So we
    rank the mask's pixels by distance from the mask centroid and keep the nearest ~target_frac
    fraction. This hits any target area precisely (unlike discrete morphological erosion, which
    overshoots on thin filaments) and reproduces the real failure mode (lost extensions). Then a
    small closing re-connects the kept core. Guarantees a non-empty strict subset for area>=3.
    """
    area0 = int(mt.sum())
    if area0 <= 2:
        return mt.copy()
    frac = float(rng.uniform(target_frac - spread, target_frac + spread))
    frac = min(0.95, max(0.3, frac))
    ys, xs = np_.nonzero(mt)
    cy, cx = ys.mean(), xs.mean()
    d2 = (ys - cy) ** 2 + (xs - cx) ** 2
    keep = max(1, int(round(area0 * frac)))
    order = np_.argsort(d2)[:keep]                 # nearest-to-centroid pixels (drop far ends)
    seed = np_.zeros_like(mt)
    seed[ys[order], xs[order]] = 1
    k = np_.ones((3, 3), np_.uint8)
    seed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, k)   # reconnect the retained core
    seed = seed & mt                                # stay a strict subset of the true mask
    return seed if int(seed.sum()) >= 1 else mt.copy()


def mask_iou(np_, a, b):
    inter = float((a.astype(bool) & b.astype(bool)).sum())
    union = float((a.astype(bool) | b.astype(bool)).sum())
    return inter / union if union else 1.0


def sample_seed(np_, cv2, mt, rng):
    """Draw a seed from the identity/mild/regime/severe mixture. Returns (seed, kind, label).

    label = 1.0 iff IoU(seed, mt) < CORR_IOU_THR (measured, so it is consistent by construction
    even when the closing step nudges a trimmed seed's area).
    """
    u = float(rng.random())
    if u < P_IDENTITY:
        seed, kind = mt.copy(), "identity"
    elif u < P_IDENTITY + P_MILD:
        seed, kind = _undersegment_seed(np_, cv2, mt, 0.90, rng, spread=0.05), "mild"
    elif u < P_IDENTITY + P_MILD + P_REGIME:
        seed, kind = _undersegment_seed(np_, cv2, mt, 0.69, rng, spread=0.10), "regime"
    else:
        seed, kind = _undersegment_seed(np_, cv2, mt, 0.40, rng, spread=0.10), "severe"
    label = 1.0 if mask_iou(np_, seed, mt) < CORR_IOU_THR else 0.0
    return seed, kind, label


def build_unet(torch, corr_head=True):
    import torch.nn as nn

    def block(cin, cout):
        return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout),
                             nn.ReLU(inplace=True), nn.Conv2d(cout, cout, 3, padding=1),
                             nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    class RefUNet(nn.Module):
        def __init__(self, cin=2, base=32):
            super().__init__()
            self.d1 = block(cin, base)
            self.d2 = block(base, base * 2)
            self.d3 = block(base * 2, base * 4)
            self.d4 = block(base * 4, base * 8)
            self.pool = nn.MaxPool2d(2)
            self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2)
            self.u3 = block(base * 8, base * 4)
            self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
            self.u2 = block(base * 4, base * 2)
            self.up1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
            self.u1 = block(base * 2, base)
            self.mask_head = nn.Conv2d(base, 1, 1)
            self.spine_head = nn.Conv2d(base, 1, 1)
            self.corr_head = (nn.Sequential(nn.Linear(base * 8 * 2, 64), nn.ReLU(inplace=True),
                                            nn.Linear(64, 1)) if corr_head else None)

        def forward(self, x):
            import torch as t
            c1 = self.d1(x)
            c2 = self.d2(self.pool(c1))
            c3 = self.d3(self.pool(c2))
            b = self.d4(self.pool(c3))
            if self.corr_head is not None:
                corr = self.corr_head(t.cat([b.mean((2, 3)), b.amax((2, 3))], 1))[:, 0]
            else:
                corr = t.zeros(x.shape[0], device=x.device, dtype=x.dtype)
            x = self.u3(t.cat([self.up3(b), c3], 1))
            x = self.u2(t.cat([self.up2(x), c2], 1))
            x = self.u1(t.cat([self.up1(x), c1], 1))
            return self.mask_head(x), self.spine_head(x), corr

    return RefUNet()


def losses(torch, functional, mlogit, slogit, mtar, star, spine_w,
           corr_logit=None, corr_tar=None, corr_w=0.0):
    prob = torch.sigmoid(mlogit)
    num = 2 * (prob * mtar).sum() + 1
    den = prob.sum() + mtar.sum() + 1
    dice = 1 - num / den
    ce = functional.binary_cross_entropy_with_logits(mlogit, mtar, reduction="none")
    pt = prob * mtar + (1 - prob) * (1 - mtar)
    focal = ((1 - pt) ** 2 * ce).mean()
    total = dice + 0.5 * focal
    if spine_w > 0:
        total = total + spine_w * functional.binary_cross_entropy_with_logits(slogit, star)
    if corr_w > 0 and corr_logit is not None and corr_tar is not None:
        total = total + corr_w * functional.binary_cross_entropy_with_logits(corr_logit,
                                                                            corr_tar)
    return total, float(dice), float(focal)


def make_dataset(torch, cv2, mask_utils, document, image_dir, files, crop):
    from torch.utils.data import Dataset
    images = {str(r["id"]): r for r in document["images"]}
    annotations = v52.load_annotations(document)
    by_file = v52.canonical_split(document)[2]
    recs = v52.records_for(files, by_file, images, annotations)
    index = [(n, iid, j) for (n, iid) in recs for j in range(len(annotations[iid]))]

    class RefSet(Dataset):
        def __len__(self):
            return len(index)

        def __getitem__(self, i):
            # per-sample RNG from the DataLoader worker seed (changes every epoch) + index.
            # (A single shared default_rng(0) forked into workers replays the same stream
            # every epoch.)
            rng = np.random.default_rng([int(torch.initial_seed()), int(i)])
            name, image_id, j = index[i]
            info = images[image_id]
            ann = annotations[image_id][j]
            gray = cv2.imread(str(image_dir / name), cv2.IMREAD_GRAYSCALE)
            m = mask_utils.decode(v52.annotation_rle(mask_utils, info, ann))
            if m.ndim == 3:
                m = m[..., 0]
            ys, xs = np.nonzero(m)
            if len(xs) == 0:
                return self.__getitem__((i + 1) % len(index))
            cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
            # jitter center so crop placement varies
            half = crop / 2
            cx += rng.integers(-crop // 8, crop // 8)
            cy += rng.integers(-crop // 8, crop // 8)
            ox = int(np.clip(cx - half, 0, FULL - crop))
            oy = int(np.clip(cy - half, 0, FULL - crop))
            g = gray[oy:oy + crop, ox:ox + crop]
            mt = m[oy:oy + crop, ox:ox + crop]
            g = cv2.resize(g, (NET, NET), interpolation=cv2.INTER_AREA)
            mt = cv2.resize(mt, (NET, NET), interpolation=cv2.INTER_NEAREST)
            seed, _kind, label = sample_seed(np, cv2, mt, rng)
            spine = rasterize_spine(np, cv2, ann["spine"], ox, oy, NET / crop, NET)
            x = np.stack([g.astype(np.float32) / 255.0, seed.astype(np.float32)])
            return (torch.from_numpy(x), torch.from_numpy(mt.astype(np.float32))[None],
                    torch.from_numpy(spine)[None], torch.tensor(label, dtype=torch.float32))

    return RefSet(), len(index)


def train(args):
    import cv2
    import torch
    import torch.nn.functional as functional
    from pycocotools import mask as mask_utils
    from torch.utils.data import DataLoader

    torch.manual_seed(20261004 + args.fold + 1000 * args.seed_offset)
    device = torch.device("cuda:0")
    document = json.loads(args.annotations.read_text())
    train_files, _ = fold.fold_train_files(document, args.fold)
    ds, n = make_dataset(torch, cv2, mask_utils, document, args.images, train_files, args.crop)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                        pin_memory=True, drop_last=True)
    model = build_unet(torch, corr_head=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler()
    best = args.work / "weights" / "v60a_refiner.pt"
    best.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        agg = acc = nb = 0
        for x, mt, st, ct in loader:
            x, mt, st, ct = x.to(device), mt.to(device), st.to(device), ct.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                ml, sl, cl = model(x)
                loss, _d, _f = losses(torch, functional, ml, sl, mt, st, args.spine_w,
                                      corr_logit=cl.float(), corr_tar=ct, corr_w=args.corr_w)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            agg += float(loss)
            acc += float(((cl.detach() > 0).float() == ct).float().mean())
            nb += 1
        sched.step()
        print(json.dumps({"fold": args.fold, "epoch": epoch, "loss": agg / max(nb, 1),
                          "corr_acc": acc / max(nb, 1)}), flush=True)
        torch.save({"model": model.state_dict(), "crop": args.crop, "net": NET,
                    "spine_w": args.spine_w, "corr_w": args.corr_w, "corr_head": True,
                    "mix": {"identity": P_IDENTITY, "mild": P_MILD, "regime": P_REGIME},
                    "epoch": epoch}, best)
    v52.write_json(args.output, {"status": "train_complete", "fold": args.fold, "n": n,
                                 "crop": args.crop, "spine_w": args.spine_w,
                                 "corr_w": args.corr_w, "seed_offset": args.seed_offset,
                                 "weights": str(best), "reads_tune": False, "reads_test": False})
    print(json.dumps({"v60a_train_complete": args.fold, "n": n, "corr_w": args.corr_w}))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold", type=int, required=True, choices=range(4))
    ap.add_argument("--annotations", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--crop", type=int, default=1024)
    ap.add_argument("--spine-w", type=float, default=0.0, dest="spine_w")
    ap.add_argument("--corr-w", type=float, default=0.5, dest="corr_w")
    ap.add_argument("--seed-offset", type=int, default=0, dest="seed_offset")
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    if v52.sha256(args.annotations) != v56.ANNOTATION_SHA256:
        raise RuntimeError("annotation identity changed")
    train(args)


if __name__ == "__main__":
    main()
