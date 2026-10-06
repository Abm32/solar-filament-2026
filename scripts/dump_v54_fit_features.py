"""V54 step 1: dump per-proposal GEOMETRIC features + TP/FP labels for V52 proposals
on the 566 FIT files, to train a proposal-quality gate.

Why: V53 proved V52 is orthogonal to V17 (FP-corr 19-24%, recovers 93/317 of V17's
FNs) but its mean-probability score cannot separate the recovered TPs from its FPs,
so the fusion recall path (beta>0) was strictly harmful. A better per-proposal score
could make beta>0 net-positive. This script produces the (features, label) training
set for that score — FIT ONLY (firewall: tune/test never touched here).

For each FIT file it runs the SAME V52 inference as the tune dump (predict_prob_full
-> mask_to_instances at a fixed threshold), then for every proposal computes cheap
geometric features from the binary mask + prob map, and labels it TP (1) iff it
greedily matches a FIT GT instance at IoU>0.5 (project semantics).

Features (all cheap, no model): area, log_area, mean_prob, max_prob, prob_x_area,
compactness (4*pi*area / perimeter^2, 1=disc), extent (area/bbox_area), elongation
(major/minor axis), solidity (area/convex_area), skeleton_length, skel_per_area.

  CUDA_VISIBLE_DEVICES=N python scripts/dump_v54_fit_features.py \
      --weights outputs/v52_cldice_b48/weights/v52_cldice_best.pt \
      --annotations <ann.json> --images <train_images> --thresh 0.40 \
      --out outputs/v54_gate/fit_features_base32.npz --shard N --shards 4
Merge shards: python scripts/dump_v54_fit_features.py merge --out <final.npz> --parts ...
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

FEATURE_NAMES = [
    "log_area", "mean_prob", "max_prob", "prob_x_area_norm",
    "compactness", "extent", "elongation", "solidity",
    "skel_len_norm", "skel_per_area",
]


def proposal_features(np_, cv2, measure, morphology, comp_mask, prob_full):
    """Cheap geometric features for one binary proposal mask + its prob map."""
    area = float(comp_mask.sum())
    ys, xs = np_.nonzero(comp_mask)
    probs = prob_full[comp_mask.astype(bool)]
    mean_prob = float(probs.mean()) if probs.size else 0.0
    max_prob = float(probs.max()) if probs.size else 0.0

    # region props on a tight crop (fast)
    r0, r1, c0, c1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = comp_mask[r0:r1, c0:c1].astype(np_.uint8)
    props = measure.regionprops(crop)[0]
    perim = float(props.perimeter) if props.perimeter > 0 else 1.0
    compactness = float(4.0 * np_.pi * area / (perim * perim))
    extent = float(props.extent)
    major = float(props.axis_major_length) or 1.0
    minor = float(props.axis_minor_length) or 1.0
    elongation = float(major / minor) if minor > 0 else float(major)
    solidity = float(props.solidity) if props.solidity > 0 else 0.0

    skel = morphology.skeletonize(crop > 0)
    skel_len = float(skel.sum())
    bbox_diag = float(np_.hypot(r1 - r0, c1 - c0)) or 1.0

    return {
        "log_area": float(np_.log1p(area)),
        "mean_prob": mean_prob,
        "max_prob": max_prob,
        "prob_x_area_norm": float(mean_prob * np_.log1p(area)),
        "compactness": compactness,
        "extent": extent,
        "elongation": elongation,
        "solidity": solidity,
        "skel_len_norm": float(skel_len / bbox_diag),
        "skel_per_area": float(skel_len / area) if area > 0 else 0.0,
    }


def dump(args):
    import cv2
    import torch
    from pycocotools import mask as mask_utils
    from skimage import measure, morphology

    device = torch.device("cuda:0")
    document = json.loads(args.annotations.read_text())
    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    unet = v52.build_unet(torch.nn)
    model = unet(base=ckpt.get("base", 32)).to(device)
    model.load_state_dict(ckpt["model"])
    res = ckpt.get("res", 1024)

    images = {str(r["id"]): r for r in document["images"]}
    annotations = v52.load_annotations(document)
    fit, _tune, by_file, _ = v52.canonical_split(document)
    recs = v52.records_for(fit, by_file, images, annotations)
    recs = recs[args.shard::args.shards]

    rows, labels = [], []
    for n, (name, image_id) in enumerate(recs, 1):
        info = images[image_id]
        gray = cv2.imread(str(args.images / name), cv2.IMREAD_GRAYSCALE)
        prob_full = v52.predict_prob_full(torch, np, cv2, model, device,
                                          gray.astype(np.float32), res, tta=not args.no_tta)
        masks, _scores = v52.mask_to_instances(np, cv2, measure, prob_full, args.thresh)
        gt_rles = [v52.annotation_rle(mask_utils, info, r) for r in annotations[image_id]]
        gt_rles = [r for r in gt_rles if mask_utils.area(r) > 0]

        # TP/FP labels: greedy IoU>0.5 one-to-one (project semantics)
        if masks and gt_rles:
            pred_rles = [mask_utils.encode(np.asfortranarray(m)) for m in masks]
            iou = np.asarray(mask_utils.iou(pred_rles, gt_rles, [0] * len(gt_rles)), dtype=float)
            hits = np.argwhere(iou > 0.5)
            ordered = sorted(hits.tolist(), key=lambda pg: (-iou[pg[0], pg[1]], pg))
            up, ug = set(), set()
            for p, g in ordered:
                if p not in up and g not in ug:
                    up.add(p)
                    ug.add(g)
        else:
            up = set()

        for i, m in enumerate(masks):
            feat = proposal_features(np, cv2, measure, morphology, m, prob_full)
            rows.append([feat[k] for k in FEATURE_NAMES])
            labels.append(1 if i in up else 0)
        print(f"shard={args.shard} {n}/{len(recs)} {name} "
              f"props={len(masks)} tp={sum(1 for i in range(len(masks)) if i in up)}",
              flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, X=np.asarray(rows, dtype=np.float64),
             y=np.asarray(labels, dtype=np.int64), feature_names=np.asarray(FEATURE_NAMES))
    print(f"wrote {args.out}: {len(labels)} proposals, {sum(labels)} TP")


def merge(args):
    parts = [np.load(p, allow_pickle=True) for p in args.parts]
    X = np.concatenate([p["X"] for p in parts], axis=0)
    y = np.concatenate([p["y"] for p in parts], axis=0)
    np.savez(args.out, X=X, y=y, feature_names=np.asarray(FEATURE_NAMES))
    print(f"merged -> {args.out}: {len(y)} proposals, {int(y.sum())} TP, "
          f"{len(y) - int(y.sum())} FP")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")
    d = sub.add_parser("dump")
    d.add_argument("--weights", type=Path, required=True)
    d.add_argument("--annotations", type=Path, required=True)
    d.add_argument("--images", type=Path, required=True)
    d.add_argument("--thresh", type=float, required=True)
    d.add_argument("--out", type=Path, required=True)
    d.add_argument("--shard", type=int, default=0)
    d.add_argument("--shards", type=int, default=1)
    d.add_argument("--no-tta", action="store_true")
    m = sub.add_parser("merge")
    m.add_argument("--out", type=Path, required=True)
    m.add_argument("--parts", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    if args.command == "merge":
        merge(args)
    else:
        if v52.sha256(args.annotations) != v52.ANNOTATION_SHA256:
            raise RuntimeError("annotation identity changed")
        dump(args)


if __name__ == "__main__":
    main()
