"""V55: retrain the V52 clDice semantic U-Net with a PRECISION-FOCUSED objective.

Motivation (docs/v54_result_2026-10-03.md): V52 is orthogonal to V17 (V53) but its
false positives are NOT post-hoc separable from its true positives by cheap features
(V54: a FIT-fit quality gate matched the raw mean-prob score, AUC 0.822). The only
remaining way to make V52 a useful fusion source is to change WHAT THE MODEL IS
CONFIDENT ABOUT — i.e. cut the FP rate at training time.

Same architecture/data/eval as V52 (imported, so V52 stays intact and the checkpoint
format is identical -> reuse V52 `--mode tune` for evaluation + proposal dumps). Only
the LOSS changes:

  loss = focal_bce
       + tversky(alpha_fp, beta_fn)        # alpha_fp > beta_fn penalises FP harder
       + cldice_w * soft_clDice            # topology (V52's lever)

Tversky generalises Dice (alpha=beta=0.5 == Dice). Setting alpha>beta (more weight on
the FP term, which is (1-target)*prob) directly trades recall for precision, attacking
V52's 519/364 FP. Focal BCE down-weights easy background so the gradient focuses on the
hard thin-structure boundaries. clДice keeps the tubular topology.

Firewall: FIT-only training; ANNOTATION_SHA256 checked; no tune labels, no test/CSV/
submission here (evaluation + dumping go through V52's tune mode, tune-labels-for-eval
only, exactly as V52).

  CUDA_VISIBLE_DEVICES=N python scripts/train_v55_precision_semantic.py \
      --annotations <ann.json> --images <train_images> --work outputs/v55_tv \
      --output outputs/v55_tv/train.json --epochs 140 --batch 4 --res 1024 --base 32 \
      --lr 1e-3 --cldice-w 1.0 --tversky-alpha 0.7 --tversky-beta 0.3 \
      --focal-gamma 2.0 --fliplr --workers 4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
import sys  # noqa: E402

sys.path.insert(0, str(SCRIPTS))
import train_v52_cldice_semantic as v52  # noqa: E402

ANNOTATION_SHA256 = v52.ANNOTATION_SHA256
TRAIN_RES = v52.TRAIN_RES


def tversky_loss(torch, prob, target, alpha_fp, beta_fn, smooth=1.0):
    """Tversky index loss. alpha_fp weights FP = prob*(1-target);
    beta_fn weights FN = (1-prob)*target. alpha_fp>beta_fn => precision-biased."""
    tp = torch.sum(prob * target)
    fp = torch.sum(prob * (1 - target))
    fn = torch.sum((1 - prob) * target)
    tv = (tp + smooth) / (tp + alpha_fp * fp + beta_fn * fn + smooth)
    return 1.0 - tv


def focal_bce(torch, functional, logits, target, gamma, alpha=0.75):
    """Focal binary cross-entropy. alpha weights the positive (filament) class;
    gamma down-weights easy examples so hard thin boundaries dominate the gradient."""
    prob = torch.sigmoid(logits)
    ce = functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p_t = prob * target + (1 - prob) * (1 - target)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    loss = alpha_t * (1 - p_t).pow(gamma) * ce
    return loss.mean()


def precision_loss(torch, functional, logits, target, cldice_w, alpha_fp, beta_fn, gamma):
    prob = torch.sigmoid(logits)
    fb = focal_bce(torch, functional, logits, target, gamma)
    tv = tversky_loss(torch, prob, target, alpha_fp, beta_fn)
    cl = v52.cldice_loss(torch, prob, target)
    total = fb + tv + cldice_w * cl
    return total, {"focal_bce": float(fb), "tversky": float(tv), "cldice": float(cl)}


def train(args):
    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as functional
    from pycocotools import mask as mask_utils
    from torch.utils.data import DataLoader

    if not torch.cuda.is_available():
        raise RuntimeError("V55 requires CUDA")
    torch.manual_seed(20261003)
    device = torch.device("cuda:0")
    document = json.loads(args.annotations.read_text())
    fit, _tune, _by_file, _ = v52.canonical_split(document)

    dataset, n_fit = v52.make_dataset(torch, np, cv2, mask_utils, document, args.images,
                                      fit, args.res)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                        drop_last=False, pin_memory=True)
    unet = v52.build_unet(torch.nn)
    model = unet(base=args.base).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    best_path = args.work / "weights" / "v55_precision_best.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        agg = {"total": 0.0, "focal_bce": 0.0, "tversky": 0.0, "cldice": 0.0, "n": 0}
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if args.fliplr and torch.rand(1).item() < 0.5:
                x = torch.flip(x, [3])
                y = torch.flip(y, [3])
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss, parts = precision_loss(torch, functional, logits, y, args.cldice_w,
                                             args.tversky_alpha, args.tversky_beta,
                                             args.focal_gamma)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            agg["total"] += float(loss)
            for k in ("focal_bce", "tversky", "cldice"):
                agg[k] += parts[k]
            agg["n"] += 1
        row = {"epoch": epoch,
               **{k: agg[k] / agg["n"] for k in ("total", "focal_bce", "tversky", "cldice")}}
        history.append(row)
        print(json.dumps(row), flush=True)
        # V52-compatible checkpoint so V52 `--mode tune` can load it unchanged
        torch.save({"model": model.state_dict(), "base": args.base, "res": args.res,
                    "epoch": epoch}, best_path)

    v52.write_json(args.output, {
        "status": "train_complete", "mode": "train", "experiment": "v55_precision",
        "n_fit_files": n_fit, "epochs": args.epochs, "res": args.res, "base": args.base,
        "lr": args.lr, "cldice_w": args.cldice_w, "tversky_alpha": args.tversky_alpha,
        "tversky_beta": args.tversky_beta, "focal_gamma": args.focal_gamma,
        "weights": str(best_path), "weights_sha256": v52.sha256(best_path),
        "history": history, "reads_tune": False, "reads_test": False,
    })
    print(json.dumps({"train_complete": True, "n_fit_files": n_fit, "weights": str(best_path)}))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annotations", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=140)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--res", type=int, default=TRAIN_RES)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--cldice-w", type=float, default=1.0, dest="cldice_w")
    ap.add_argument("--tversky-alpha", type=float, default=0.7, dest="tversky_alpha",
                    help="FP penalty weight (>beta => precision-biased)")
    ap.add_argument("--tversky-beta", type=float, default=0.3, dest="tversky_beta",
                    help="FN penalty weight")
    ap.add_argument("--focal-gamma", type=float, default=2.0, dest="focal_gamma")
    ap.add_argument("--fliplr", action="store_true")
    args = ap.parse_args()

    if v52.sha256(args.annotations) != ANNOTATION_SHA256:
        raise RuntimeError("V55 annotation identity changed")
    train(args)


if __name__ == "__main__":
    main()
