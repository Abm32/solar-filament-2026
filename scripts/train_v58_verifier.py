"""V58 step 2: train the FILAMENT VERIFIER on the OOF candidate dataset.

The verifier is a binary classifier P(real filament | candidate crop + context + features).
It is the recognition fix our bottleneck needs (docs/swefil_correction_and_revised_strategy):
a high-recall detector over-proposes; the verifier rejects filament-looking junk/sunspots
that the detector's own confidence can't separate (V54 proved score-calibration alone fails
because a segmentation model confidently segments whatever seed you give it; a classifier
trained on the detector's REAL OOF false-positives can judge whether the object should exist).

Inputs: one or more OOF candidate dirs from build_v58_oof_candidates.py, each with
crops.npy (N,128,128 uint8) + candidates.jsonl (label tp/fp/nearmiss + scalar features).
Label: tp=1, fp=0, nearmiss excluded from the loss by default (uncertain bucket).

Model: EfficientNet-B0 (ImageNet) on the 1-ch crop replicated to 3-ch, concatenated with
standardised scalar features -> MLP -> P(real). Grouped-by-file split for honest val.

Reports classification AUC. (PQ-calibrated acceptance threshold is applied downstream in
the verifier-in-the-loop eval harness, eval_v58_verifier.py.)

Firewall: trains only on OOF candidates derived from FIT images. No test, no SWEFil here
(SWEFil pretraining is a separate, rules-gated step).

  CUDA_VISIBLE_DEVICES=N python scripts/train_v58_verifier.py \
      --cand outputs/v58_oof/fold0 outputs/v58_oof/fold1 ... \
      --out outputs/v58_verifier --epochs 30
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FEATURE_KEYS = ["score", "log_area", "elongation", "disk_r", "disk_theta",
                "mean_int", "contrast"]


def load_candidates(cand_dirs):
    crops, feats, labels, files = [], [], [], []
    for d in cand_dirs:
        d = Path(d)
        arr = np.load(d / "crops.npy")
        rows = [json.loads(line) for line in (d / "candidates.jsonl").read_text().splitlines()]
        assert len(rows) == len(arr), f"{d}: {len(rows)} rows vs {len(arr)} crops"
        for r in rows:
            crops.append(arr[r["crop_idx"]])
            feats.append([float(r[k]) for k in FEATURE_KEYS])
            labels.append(r["label"])
            files.append(r["file"])
    return np.stack(crops), np.asarray(feats, np.float32), labels, files


def build_verifier(torch, n_feat, init_backbone=None):
    import torch.nn as nn
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

    class Verifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            feat_dim = self.cnn.classifier[1].in_features
            self.cnn.classifier = nn.Identity()
            self.mlp = nn.Sequential(
                nn.Linear(feat_dim + n_feat, 256), nn.ReLU(inplace=True), nn.Dropout(0.3),
                nn.Linear(256, 1),
            )

        def forward(self, img, feat):
            z = self.cnn(img)
            return self.mlp(torch.cat([z, feat], 1)).squeeze(1)

    v = Verifier()
    if init_backbone is not None:
        ck = torch.load(init_backbone, map_location="cpu", weights_only=False)
        missing, unexpected = v.cnn.load_state_dict(ck["backbone"], strict=False)
        print(f"loaded SWEFil backbone (missing {len(missing)}, unexpected {len(unexpected)})")
    return v


def auc(y, s):
    order = np.argsort(s)
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    npos = int(y.sum())
    nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def train(args):
    import torch
    import torch.nn.functional as functional
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("cuda:0")
    crops, feats, labels, files = load_candidates(args.cand)
    keep = np.array([lab in ("tp", "fp") for lab in labels])
    crops, feats = crops[keep], feats[keep]
    y = np.array([1.0 if lab == "tp" else 0.0 for lab, k in zip(labels, keep, strict=True) if k])
    files = [f for f, k in zip(files, keep, strict=True) if k]
    print(f"candidates: {len(y)}  tp={int(y.sum())}  fp={int((y == 0).sum())}")

    # grouped split by file (honest val)
    import random
    uf = sorted(set(files))
    random.Random(0).shuffle(uf)
    val_files = set(uf[: max(1, len(uf) // 5)])
    va = np.array([f in val_files for f in files])
    tr = ~va

    mean, std = feats[tr].mean(0), feats[tr].std(0) + 1e-6
    fz = (feats - mean) / std

    def ten(a):
        return torch.tensor(a, dtype=torch.float32)

    ds_tr = TensorDataset(ten(crops[tr])[:, None], ten(fz[tr]), ten(y[tr]))
    ds_va = TensorDataset(ten(crops[va])[:, None], ten(fz[va]), ten(y[va]))
    dl_tr = DataLoader(ds_tr, batch_size=64, shuffle=True, num_workers=4)
    dl_va = DataLoader(ds_va, batch_size=128)

    model = build_verifier(torch, len(FEATURE_KEYS)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    pos_w = torch.tensor([(y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)], device=device)
    best_auc, best_path = -1.0, args.out / "weights" / "verifier_best.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)
    hist = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        for img, feat, lab in dl_tr:
            img = (img / 255.0).repeat(1, 3, 1, 1).to(device)
            feat, lab = feat.to(device), lab.to(device)
            opt.zero_grad(set_to_none=True)
            logit = model(img, feat)
            loss = functional.binary_cross_entropy_with_logits(logit, lab, pos_weight=pos_w)
            loss.backward()
            opt.step()
        # val AUC
        model.eval()
        ss, yy = [], []
        with torch.no_grad():
            for img, feat, lab in dl_va:
                img = (img / 255.0).repeat(1, 3, 1, 1).to(device)
                ss.append(torch.sigmoid(model(img, feat.to(device))).cpu().numpy())
                yy.append(lab.numpy())
        va_auc = auc(np.concatenate(yy), np.concatenate(ss))
        hist.append({"epoch": epoch, "val_auc": va_auc})
        print(json.dumps(hist[-1]), flush=True)
        if va_auc > best_auc:
            best_auc = va_auc
            torch.save({"model": model.state_dict(), "mean": mean.tolist(), "std": std.tolist(),
                        "feature_keys": FEATURE_KEYS, "val_auc": va_auc}, best_path)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "train.json").write_text(json.dumps(
        {"status": "train_complete", "n": int(len(y)), "best_val_auc": best_auc,
         "history": hist, "weights": str(best_path), "reads_test": False}, indent=2) + "\n")
    print(json.dumps({"verifier_train_complete": True, "best_val_auc": best_auc}))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cand", nargs="+", required=True, help="OOF candidate dirs")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--init-backbone", type=Path, default=None, dest="init_backbone",
                    help="SWEFil-pretrained backbone weights to initialise the CNN")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
