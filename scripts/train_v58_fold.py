"""V58 fold trainer: train a Mask R-CNN on 3/4 of the FIT files (holding out quarter K),
so build_v58_oof_candidates.py can generate OUT-OF-FOLD proposals on the held-out quarter.

Reuses V56's model + dataset. The fold split MUST match resolve_files() in
build_v58_oof_candidates.py: files=sorted(fit); Random(1234).shuffle; quarter k = files[k::4].
This trains on fit MINUS quarter k. Short schedule (OOF proposals don't need a perfect
detector, just a representative-recall one trained without seeing the held-out images).

Firewall: FIT-only, never tune/test. 141 tune files are never in any fold.

  CUDA_VISIBLE_DEVICES=K python scripts/train_v58_fold.py --fold K \
      --annotations <ann> --images <train_images> --work outputs/v58_fold_K \
      --epochs 16 --imgsz 1536
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import train_v52_cldice_semantic as v52  # noqa: E402
import train_v56_maskrcnn as v56  # noqa: E402


def fold_train_files(document, k):
    """FIT minus held-out quarter k (matches build_v58_oof_candidates.resolve_files)."""
    fit, _tune, _by, _ = v52.canonical_split(document)
    files = sorted(fit)
    random.Random(1234).shuffle(files)
    heldout = set(files[k::4])
    return sorted(f for f in files if f not in heldout), sorted(heldout)


def train(args):
    import cv2
    import torch
    from pycocotools import mask as mask_utils
    from torch.utils.data import DataLoader

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.manual_seed(20261004 + args.fold)
    device = torch.device("cuda:0")
    document = json.loads(args.annotations.read_text())
    train_files, heldout = fold_train_files(document, args.fold)

    # build a dataset over an explicit file list (reuse V56 make_dataset with files=)
    dataset, n = v56.make_dataset(torch, cv2, mask_utils, document, args.images,
                                  train_files, args.imgsz, target="single")
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                        collate_fn=v56.collate, pin_memory=True)
    model = v56.build_model(torch, args.imgsz).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler()
    best = args.work / "weights" / "v56_mrcnn_best.pt"
    best.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        agg, nb = 0.0, 0
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
        print(json.dumps({"fold": args.fold, "epoch": epoch, "loss": agg / max(nb, 1)}), flush=True)
        torch.save({"model": model.state_dict(), "imgsz": args.imgsz, "epoch": epoch,
                    "fold": args.fold}, best)
    v52.write_json(args.output, {"status": "train_complete", "fold": args.fold,
                                 "n_train": n, "n_heldout": len(heldout), "imgsz": args.imgsz,
                                 "weights": str(best), "reads_tune": False, "reads_test": False})
    print(json.dumps({"fold_train_complete": args.fold, "n_train": n, "n_heldout": len(heldout)}))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold", type=int, required=True, choices=range(4))
    ap.add_argument("--annotations", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--imgsz", type=int, default=1536)
    ap.add_argument("--lr", type=float, default=4e-3)
    args = ap.parse_args()
    if v52.sha256(args.annotations) != v56.ANNOTATION_SHA256:
        raise RuntimeError("annotation identity changed")
    train(args)


if __name__ == "__main__":
    main()
