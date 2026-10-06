"""V58 step 5: pretrain the verifier backbone on SWEFil (filament vs sunspot vs background).

Why (docs/swefil_correction..., confirmed by the MAGFiLO-only verifier result AUC 0.828 ~=
detector score 0.822, +0.006): the discriminative signal for "real filament vs
filament-looking dark junk" is NOT present in MAGFiLO (which never labels sunspots). SWEFil
DOES label 804 sunspots alongside filaments. Pretraining the verifier's EfficientNet-B0
backbone on SWEFil to tell filament / sunspot / background apart injects exactly the
dark-feature discrimination the MAGFiLO verifier lacks. We then load these weights into
train_v58_verifier (via --init-backbone) and fine-tune on MAGFiLO OOF candidates.

SWEFil (CC-BY-4.0, external public data; used here ONLY to pretrain a recognition backbone,
not as competition ground truth). Firewall-verified non-overlapping with Kaggle tune/test.

Crops: same geometry as the verifier (128px, 2x context box) so the backbone transfers.
3-class head: filament (QRF/IRF/ARF -> 1 class), SUNSPOT, BACKGROUND (random disk patches).

  CUDA_VISIBLE_DEVICES=N python scripts/pretrain_v58_swefil.py \
      --swefil /mnt/swefil --out outputs/v58_swefil_pretrain --epochs 40
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

CROP = 128
CONTEXT = 2.0
FULL = 2048
# category_id -> class: 0/1/2 filament types -> FILAMENT(0); 3 -> SUNSPOT(1); plus BACKGROUND(2)
FILAMENT, SUNSPOT, BACKGROUND = 0, 1, 2


def context_crop(cv2, np_, gray, box):
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    half = max(w, h) * CONTEXT / 2
    a, b = int(max(0, cx - half)), int(max(0, cy - half))
    c, d = int(min(FULL, cx + half)), int(min(FULL, cy + half))
    crop = gray[b:d, a:c]
    if crop.size == 0:
        return np_.zeros((CROP, CROP), np_.uint8)
    return cv2.resize(crop, (CROP, CROP), interpolation=cv2.INTER_AREA).astype(np_.uint8)


def build_crops(cv2, swefil, split_json):
    doc = json.loads((Path(swefil) / split_json).read_text())
    images = {im["id"]: im for im in doc["images"]}
    anns = doc["annotations"]
    crops, labels = [], []
    cache = {}
    # positives: filament + sunspot
    per_img_boxes = {}
    for a in anns:
        iid = a["image_id"]
        fn = images[iid]["file_name"].split("/")[-1]
        if iid not in cache:
            g = cv2.imread(str(Path(swefil) / "images" / fn), cv2.IMREAD_GRAYSCALE)
            cache[iid] = g
        g = cache[iid]
        if g is None:
            continue
        cls = SUNSPOT if a["category_id"] == 3 else FILAMENT
        crops.append(context_crop(cv2, np, g, a["bbox"]))
        labels.append(cls)
        per_img_boxes.setdefault(iid, []).append(a["bbox"])
    # negatives: random on-disk patches not overlapping any annotation
    rng = random.Random(0)
    for iid, g in cache.items():
        if g is None:
            continue
        boxes = per_img_boxes.get(iid, [])
        n_bg = max(2, len(boxes) // 2)
        got = 0
        for _ in range(n_bg * 10):
            if got >= n_bg:
                break
            s = rng.randint(80, 300)
            cx = rng.randint(FULL // 4, 3 * FULL // 4)
            cy = rng.randint(FULL // 4, 3 * FULL // 4)
            if int(g[cy, cx]) < 30:   # must be on bright disk
                continue
            overlap = any(abs(cx - (bx + bw / 2)) < (bw + s) / 2 and
                          abs(cy - (by + bh / 2)) < (bh + s) / 2
                          for bx, by, bw, bh in boxes)
            if overlap:
                continue
            crops.append(context_crop(cv2, np, g, [cx - s / 2, cy - s / 2, s, s]))
            labels.append(BACKGROUND)
            got += 1
    return np.stack(crops), np.asarray(labels, np.int64)


def build_backbone(torch):
    import torch.nn as nn
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            self.feat_dim = self.cnn.classifier[1].in_features
            self.cnn.classifier = nn.Identity()
            self.head = nn.Linear(self.feat_dim, 3)

        def forward(self, x):
            return self.head(self.cnn(x))

    return Net()


def train(args):
    import cv2
    import torch
    import torch.nn.functional as functional
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("cuda:0")
    xtr, ytr = build_crops(cv2, args.swefil, "train.json")
    xva, yva = build_crops(cv2, args.swefil, "test.json")
    print("SWEFil crops: train", len(ytr), dict(zip(*np.unique(ytr, return_counts=True),
                                                    strict=True)),
          "| val", len(yva), dict(zip(*np.unique(yva, return_counts=True), strict=True)))

    def ten(a):
        return torch.tensor(a, dtype=torch.float32)

    dtr = TensorDataset(ten(xtr)[:, None], torch.tensor(ytr))
    dva = TensorDataset(ten(xva)[:, None], torch.tensor(yva))
    ltr = DataLoader(dtr, batch_size=64, shuffle=True, num_workers=4)
    lva = DataLoader(dva, batch_size=128)
    model = build_backbone(torch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    best_acc, best = -1.0, args.out / "weights" / "swefil_backbone.pt"
    best.parent.mkdir(parents=True, exist_ok=True)
    hist = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in ltr:
            x = (x / 255.0).repeat(1, 3, 1, 1).to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            functional.cross_entropy(model(x), y).backward()
            opt.step()
        model.eval()
        correct = tot = 0
        fil_sun_correct = fil_sun_tot = 0
        with torch.no_grad():
            for x, y in lva:
                x = (x / 255.0).repeat(1, 3, 1, 1).to(device)
                pred = model(x).argmax(1).cpu()
                correct += int((pred == y).sum())
                tot += len(y)
                m = y < 2  # filament vs sunspot subset (the signal we care about)
                fil_sun_correct += int((pred[m] == y[m]).sum())
                fil_sun_tot += int(m.sum())
        acc = correct / tot
        fs = fil_sun_correct / max(fil_sun_tot, 1)
        hist.append({"epoch": epoch, "val_acc": acc, "fil_vs_sun_acc": fs})
        print(json.dumps(hist[-1]), flush=True)
        if fs > best_acc:
            best_acc = fs
            torch.save({"backbone": model.cnn.state_dict(), "val_acc": acc,
                        "fil_vs_sun_acc": fs}, best)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "train.json").write_text(json.dumps(
        {"status": "train_complete", "best_fil_vs_sun_acc": best_acc, "history": hist,
         "backbone": str(best), "reads_test": False}, indent=2) + "\n")
    print(json.dumps({"swefil_pretrain_complete": True, "best_fil_vs_sun_acc": best_acc}))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--swefil", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
