"""V59a: does the SWEFil verifier reranking TRANSFER onto V17's strong proposals?

The OOF test showed +0.0282 PQ from reranking the weak fold detectors (0.354 base). The
only question that matters: does it still help on V17-level proposals (0.4487 base)?

Pipeline: reproduce V17's EXACT per-tune-file proposal set (seed1/mr1_iou030 signed caches
-> flip_consensus cluster -> V11 refiner clean_refined_predictions at t=0.45), WITHOUT the
c=0.30 confidence cutoff — so we have refined RLEs + refiner scores. Score each proposal
with the FROZEN SWEFil-pretrained verifier (same crop+context+disk-pos feature extraction
as training). Then:
  * baseline = V17's own confidence sweep c in V17 grid (the untouched V17 policy)
  * reranked = S = logit(det) + alpha*logit(ver); sweep alpha {1,1.5,2,2.5,3} x cutoff.
Score project PQ (greedy IoU>0.5) at full 2048 res on the 94-file V50 tune split.

Verdict (user bar, on V17 base): <=+0.003 OOF-artifact · +0.003-0.008 polish ·
+0.008-0.015 real · >+0.015 main line (submission test) · >=+0.025 ceiling-breaker.

Firewall: V17 reproduced from its own signed caches (no detector run), verifier backbone
pretrained on licensed SWEFil + fine-tuned on FIT OOF. Tune labels used only to score PQ
(as any tune eval). No test, no submission.

  CUDA_VISIBLE_DEVICES=N python scripts/eval_v59_v17_verifier.py \
      --verifier outputs/v58_verifier_swefil/weights/verifier_best.pt \
      --out outputs/v59_v17_verifier.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import build_v58_oof_candidates as B  # noqa: E402
import reproduce_v17_from_cache as v50  # noqa: E402
import train_v58_verifier as V  # noqa: E402

FULL = 2048
T_REFINE = 0.45                 # V17's selected t
V17_C = v50.CONFIDENCES         # V17's own confidence grid (baseline sweep)
ALPHAS = (1.0, 1.5, 2.0, 2.5, 3.0)
CUTOFFS = (-0.5, 0.0, 0.5, 1.0, 1.5, 2.0)
V17_SEED, V17_CELL = "seed1", "mr1_iou030"


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def pq(iou_sum, tp, fp, fn):
    d = tp + 0.5 * (fp + fn)
    return float(iou_sum / d) if d else 0.0


def collect(args):
    """Per tune file: V17 refined proposals (rles, det_scores) + verifier scores + GT (small)."""
    import cv2
    import torch
    from pycocotools import mask as mask_utils

    rt = v50.runtime_state()
    core, tta, base = rt["core"], rt["tta"], rt["base"]
    _, tune, _, _, by_file = rt["split"]
    device = torch.device("cuda:0")

    vck = torch.load(args.verifier, map_location=device, weights_only=False)
    vmodel = V.build_verifier(torch, len(vck["feature_keys"])).to(device)
    vmodel.load_state_dict(vck["model"])
    vmodel.eval()
    vmean, vstd = np.asarray(vck["mean"]), np.asarray(vck["std"])
    vkeys = vck["feature_keys"]
    SMALL = 512

    per_image = []
    for name in sorted(tune):
        gray = cv2.imread(str(base.CFG.train_dir / name), cv2.IMREAD_GRAYSCALE)
        disk = B.disk_params(gray, np)
        views = v50.read_signed_views(name, "train", V17_CELL, v50.DETECTORS[V17_SEED])
        proposals, scores = tta.cluster_views(views, v50.POLICY, v50.MATCH_IOU)
        if proposals:
            masks = [mask_utils.decode(r) for r in proposals]
            bounds, _, probs = core.crop_probabilities(base, rt["model"], gray, masks, True)
            rles, dsc = core.clean_refined_predictions(masks, scores, bounds, probs,
                                                       T_REFINE, v50.MODE)
        else:
            rles, dsc = [], []
        pmasks, dscores, crops, feats = [], [], [], []
        for r, s in zip(rles, dsc, strict=True):
            m = mask_utils.decode(r)
            if m.ndim == 3:
                m = m[..., 0]
            msmall = cv2.resize(m, (SMALL, SMALL), interpolation=cv2.INTER_NEAREST).astype(bool)
            if int(msmall.sum()) < 2:
                continue
            ys, xs = np.nonzero(m)
            if len(xs) == 0:
                continue
            box = [float(xs.min()), float(ys.min()), float(xs.max()) + 1, float(ys.max()) + 1]
            pmasks.append(msmall)
            dscores.append(float(s))
            ft = B.candidate_features(np, box, s, m, gray, disk)
            feats.append([ft[kk] for kk in vkeys])
            x0, y0, x1, y1 = box
            cxb, cyb = (x0 + x1) / 2, (y0 + y1) / 2
            half = max(x1 - x0, y1 - y0) * B.CONTEXT / 2
            a0, b0 = int(max(0, cxb - half)), int(max(0, cyb - half))
            c0, d0 = int(min(FULL, cxb + half)), int(min(FULL, cyb + half))
            cr = gray[b0:d0, a0:c0]
            cr = (cv2.resize(cr, (B.CROP, B.CROP), interpolation=cv2.INTER_AREA)
                  if cr.size else np.zeros((B.CROP, B.CROP), np.uint8))
            crops.append(cr.astype(np.uint8))
        if crops:
            ci = torch.tensor(np.stack(crops), dtype=torch.float32)[:, None]
            ci = (ci / 255.0).repeat(1, 3, 1, 1).to(device)
            fz = torch.tensor((np.asarray(feats) - vmean) / vstd, dtype=torch.float32).to(device)
            with torch.no_grad():
                vs = torch.sigmoid(vmodel(ci, fz)).cpu().numpy()
        else:
            vs = np.zeros(0)
        gt_sets = []
        for image_id in sorted(by_file[name]):
            info = rt["images"][image_id]
            gm = []
            for e in rt["annotations"].get(image_id, []):
                r = core.annotation_rle(info, e)
                if mask_utils.area(r) > 0:
                    d = mask_utils.decode(r)
                    if d.ndim == 3:
                        d = d[..., 0]
                    gm.append(cv2.resize(d, (SMALL, SMALL),
                                         interpolation=cv2.INTER_NEAREST).astype(bool))
            gt_sets.append(gm)
        per_image.append((np.asarray(dscores), np.asarray(vs), pmasks, gt_sets))
        print(f"collected {name} nprop={len(dscores)}", flush=True)
    return per_image


def score(per_image, selector):
    iou_sum = tp = fp = fn = 0
    for dsc, vsc, pmasks, gt_sets in per_image:
        idx = selector(dsc, vsc)
        order = sorted(idx, key=lambda i: -dsc[i])
        owned = None
        kept = []
        for i in order:
            m = pmasks[i]
            if owned is None:
                owned = np.zeros_like(m)
            m = m & ~owned
            if int(m.sum()) < 1:
                continue
            owned |= m
            kept.append(m)
        kf = (np.stack([m.ravel() for m in kept]).astype(np.float32)
              if kept else np.zeros((0, 1), np.float32))
        # score the SAME kept set against EACH annotator GT (V17-identical summation)
        for gmasks in gt_sets:
            if not kept and not gmasks:
                continue
            if not kept:
                fn += len(gmasks)
                continue
            if not gmasks:
                fp += len(kept)
                continue
            gf = np.stack([g.ravel() for g in gmasks]).astype(np.float32)
            inter = kf @ gf.T
            iou = inter / (kf.sum(1)[:, None] + gf.sum(1)[None, :] - inter + 1e-9)
            hits = sorted(np.argwhere(iou > 0.5).tolist(), key=lambda pg: -iou[pg[0], pg[1]])
            up, ug = set(), set()
            for p, g in hits:
                if p not in up and g not in ug:
                    up.add(p)
                    ug.add(g)
                    iou_sum += float(iou[p, g])
            tp += len(up)
            fp += len(kept) - len(up)
            fn += len(gmasks) - len(ug)
    return {"pq": pq(iou_sum, tp, fp, fn), "tp": tp, "fp": fp, "fn": fn}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verifier", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    per_image = collect(args)

    results = {}
    best_base = (-1, None)
    for c in V17_C:
        def sel(d, v, c=c):
            return list(np.where(d >= c)[0])
        results[f"v17_c{c:.2f}"] = score(per_image, sel)
        if results[f"v17_c{c:.2f}"]["pq"] > best_base[0]:
            best_base = (results[f"v17_c{c:.2f}"]["pq"], f"v17_c{c:.2f}")
    best_rr = (-1, None)
    for a in ALPHAS:
        for cut in CUTOFFS:
            def sel(d, v, a=a, cut=cut):
                if len(v) == 0:
                    return []
                return list(np.where(logit(d) + a * logit(v) >= cut)[0])
            key = f"rr_a{a}_cut{cut}"
            results[key] = score(per_image, sel)
            if results[key]["pq"] > best_rr[0]:
                best_rr = (results[key]["pq"], key)
    delta = best_rr[0] - best_base[0]
    if delta <= 0.003:
        verdict = "oof_artifact"
    elif delta < 0.008:
        verdict = "polish"
    elif delta < 0.015:
        verdict = "real"
    elif delta < 0.025:
        verdict = "main_line"
    else:
        verdict = "ceiling_breaker"
    report = {"v17_baseline_best": {"policy": best_base[1], **results[best_base[1]]},
              "reranked_best": {"policy": best_rr[1], **results[best_rr[1]]},
              "delta": delta, "verdict": verdict, "all": results, "reads_test": False}
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("v17_baseline_best", "reranked_best",
                                             "delta", "verdict")}, indent=2))


if __name__ == "__main__":
    main()
