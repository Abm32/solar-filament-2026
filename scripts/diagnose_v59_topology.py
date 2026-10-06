"""V59 pre-diagnostic: does V17 actually lose PQ to TOPOLOGY errors (fragmentation / merge)?

Spine supervision (V59A/B) can only help if V17 is losing meaningful PQ to:
  * one-to-many (FRAGMENTATION): one GT filament covered by >=2 predictions
    -> 1 TP + extra FP(s). PQ loses 0.5 per extra FP.
  * many-to-one (MERGE): one prediction covering >=2 GT filaments
    -> 1 TP + extra FN(s). PQ loses 0.5 per extra FN.
If these rates are low, V59 has no headroom -> kill before training. This measures them
on V17's exact tune proposals (reproduced from signed caches, operating cell c0.30), using
IoU>0.1 overlap analysis (looser than the 0.5 PQ-match, to catch partial frag/merge), per
annotator GT set (V17-identical). Also reports PQ on long/thin filaments (elongation>=4).

Firewall: V17 from its own caches; tune labels for scoring only. No test, no submission.

  CUDA_VISIBLE_DEVICES=N python scripts/diagnose_v59_topology.py --out outputs/v59_diag.json
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

FULL = 2048
SMALL = 512
T_REFINE, C_OP = 0.45, 0.30
V17_SEED, V17_CELL = "seed1", "mr1_iou030"
OVL = 0.1          # loose overlap to detect partial fragmentation/merge
MATCH = 0.5        # PQ match threshold


def small_mask(cv2, np_, m):
    if m.ndim == 3:
        m = m[..., 0]
    return cv2.resize(m, (SMALL, SMALL), interpolation=cv2.INTER_NEAREST).astype(bool)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import cv2
    from pycocotools import mask as mask_utils

    rt = v50.runtime_state()
    core, tta, base = rt["core"], rt["tta"], rt["base"]
    _, tune, _, _, by_file = rt["split"]

    agg = {"tp": 0, "fp": 0, "fn": 0, "iou_sum": 0.0,
           "gt_total": 0, "gt_fragmented": 0, "frag_extra_preds": 0,
           "pred_total": 0, "pred_merged": 0, "merge_extra_gts": 0,
           "thin_gt": 0, "thin_tp": 0}
    for name in sorted(tune):
        gray = cv2.imread(str(base.CFG.train_dir / name), cv2.IMREAD_GRAYSCALE)
        views = v50.read_signed_views(name, "train", V17_CELL, v50.DETECTORS[V17_SEED])
        props, scores = tta.cluster_views(views, v50.POLICY, v50.MATCH_IOU)
        if props:
            masks = [mask_utils.decode(r) for r in props]
            bounds, _, probs = core.crop_probabilities(base, rt["model"], gray, masks, True)
            rles, sc = core.clean_refined_predictions(masks, scores, bounds, probs,
                                                      T_REFINE, v50.MODE)
            keep = np.asarray(sc) >= C_OP if len(sc) else np.zeros(0, bool)
            pmasks = [small_mask(cv2, np, mask_utils.decode(r))
                      for r, f in zip(rles, keep, strict=True) if f]
        else:
            pmasks = []
        pflat = (np.stack([m.ravel() for m in pmasks]).astype(np.float32)
                 if pmasks else np.zeros((0, 1), np.float32))
        for image_id in sorted(by_file[name]):
            info = rt["images"][image_id]
            gmasks, thin = [], []
            for e in rt["annotations"].get(image_id, []):
                r = core.annotation_rle(info, e)
                if mask_utils.area(r) <= 0:
                    continue
                gm = small_mask(cv2, np, mask_utils.decode(r))
                gmasks.append(gm)
                ys, xs = np.nonzero(gm)
                if len(xs):
                    w = xs.max() - xs.min() + 1
                    h = ys.max() - ys.min() + 1
                    elong = max(w, h) / max(min(w, h), 1)
                else:
                    elong = 1
                thin.append(elong >= 4)
            if not gmasks:
                agg["fp"] += len(pmasks)
                continue
            gflat = np.stack([g.ravel() for g in gmasks]).astype(np.float32)
            agg["gt_total"] += len(gmasks)
            agg["pred_total"] += len(pmasks)
            agg["thin_gt"] += int(sum(thin))
            if not pmasks:
                agg["fn"] += len(gmasks)
                continue
            inter = pflat @ gflat.T
            iou = inter / (pflat.sum(1)[:, None] + gflat.sum(1)[None, :] - inter + 1e-9)
            # structural overlap analysis (loose OVL)
            ovl = iou >= OVL
            preds_per_gt = ovl.sum(0)           # how many preds hit each GT
            gts_per_pred = ovl.sum(1)           # how many GT each pred hits
            frag = preds_per_gt >= 2
            agg["gt_fragmented"] += int(frag.sum())
            agg["frag_extra_preds"] += int((preds_per_gt[frag] - 1).sum())
            merged = gts_per_pred >= 2
            agg["pred_merged"] += int(merged.sum())
            agg["merge_extra_gts"] += int((gts_per_pred[merged] - 1).sum())
            # PQ one-to-one (match 0.5)
            hits = sorted(np.argwhere(iou > MATCH).tolist(), key=lambda pg: -iou[pg[0], pg[1]])
            up, ug = set(), set()
            for p, g in hits:
                if p not in up and g not in ug:
                    up.add(p)
                    ug.add(g)
                    agg["iou_sum"] += float(iou[p, g])
                    if thin[g]:
                        agg["thin_tp"] += 1
            agg["tp"] += len(up)
            agg["fp"] += len(pmasks) - len(up)
            agg["fn"] += len(gmasks) - len(ug)
        print(f"diag {name}", flush=True)

    d = agg["tp"] + 0.5 * (agg["fp"] + agg["fn"])
    pq = agg["iou_sum"] / d if d else 0.0
    report = {
        "pq": pq, **{k: agg[k] for k in ("tp", "fp", "fn")},
        "gt_total": agg["gt_total"],
        "gt_fragmented": agg["gt_fragmented"],
        "frag_rate": agg["gt_fragmented"] / max(agg["gt_total"], 1),
        "frag_extra_preds": agg["frag_extra_preds"],
        "pred_total": agg["pred_total"],
        "pred_merged": agg["pred_merged"],
        "merge_rate": agg["pred_merged"] / max(agg["pred_total"], 1),
        "merge_extra_gts": agg["merge_extra_gts"],
        "thin_gt": agg["thin_gt"], "thin_tp": agg["thin_tp"],
        "thin_recall": agg["thin_tp"] / max(agg["thin_gt"], 1),
        # PQ cost estimate: extra FP from frag + extra FN from merge, each worth 0.5 in denom
        "est_pq_cost_frag": 0.5 * agg["frag_extra_preds"] / d if d else 0.0,
        "est_pq_cost_merge": 0.5 * agg["merge_extra_gts"] / d if d else 0.0,
        "reads_test": False,
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
