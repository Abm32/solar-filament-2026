"""V58 step 6: verifier-in-the-loop PQ evaluation (the decisive, metric-that-matters test).

Per user: a small +0.005 AUC can still matter at the operating threshold, and the
"8.2% score-unseparable" number bounds easy separation, NOT PQ-reorderable headroom. So
MEASURE PQ directly, don't infer.

Honest OOF setup: for each fold k, re-run fold k's detector on its held-out quarter (which
it never trained on), score every proposal with the SWEFil-pretrained verifier, then build
the disjoint prediction set under a POLICY and score project PQ against GT. Aggregate PQ
over all 4 folds = all 566 FIT files, out-of-fold.

Policies swept:
  * detector-only baseline: keep proposals with det_score >= c   (sweep c)  [the control]
  * combined: S = logit(det) + alpha*logit(ver); keep S >= cutoff (sweep alpha, cutoff)
  * tiered gate: det>=hi keep; lo<=det<hi keep iff ver>=vth; det<lo drop (sweep hi,lo,vth)

Verdict (user's bar): best-combined/tiered PQ minus best detector-only PQ:
  <+0.003 dead(archive) · 0.003-0.008 polish-only · >0.008 keep · >0.015 investigate.

Firewall: FIT-only OOF; verifier backbone pretrained on licensed SWEFil (recognition only);
no test, no submission.

  CUDA_VISIBLE_DEVICES=N python scripts/eval_v58_verifier_pq.py \
      --verifier outputs/v58_verifier_swefil/weights/verifier_best.pt \
      --annotations <ann> --images <train_images> --out outputs/v58_verifier_pq.json
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
import train_v52_cldice_semantic as v52  # noqa: E402
import train_v56_maskrcnn as v56  # noqa: E402
import train_v58_verifier as V  # noqa: E402

FULL = 2048
SMALL = 512                                   # sweep resolution (ranking-preserving, ~16x cheaper)
MIN_AREA_SMALL = max(1, int(v56.MIN_AREA * (SMALL / FULL) ** 2))
DET_CS = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80)
ALPHAS = (0.5, 1.0, 2.0)
CUTOFFS = (-1.0, -0.5, 0.0, 0.5, 1.0, 1.5)
HI, LO = 0.80, 0.40
VTHS = (0.3, 0.5, 0.7)


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def pq(iou_sum, tp, fp, fn):
    denom = tp + 0.5 * (fp + fn)
    return float(iou_sum / denom) if denom else 0.0


def collect(args):
    """Re-run each fold detector OOF on its held-out quarter; score with verifier;
    return per-image list of (det_scores, ver_scores, pred_rles, gt_rles)."""
    import cv2
    import torch
    from pycocotools import mask as mask_utils

    device = torch.device("cuda:0")
    document = json.loads(args.annotations.read_text())
    images = {str(r["id"]): r for r in document["images"]}
    annotations = v52.load_annotations(document)
    by_file = v52.canonical_split(document)[2]

    # verifier
    vck = torch.load(args.verifier, map_location=device, weights_only=False)
    vmodel = V.build_verifier(torch, len(vck["feature_keys"])).to(device)
    vmodel.load_state_dict(vck["model"])
    vmodel.eval()
    vmean, vstd = np.asarray(vck["mean"]), np.asarray(vck["std"])
    vkeys = vck["feature_keys"]

    per_image = []
    for k in range(4):
        wk = Path(f"outputs/v58_fold_{k}/weights/v56_mrcnn_best.pt")
        dck = torch.load(wk, map_location=device, weights_only=False)
        imgsz = dck.get("imgsz", 1536)
        det = v56.build_model(torch, imgsz).to(device)
        det.load_state_dict(dck["model"])
        det.eval()
        files = B.resolve_files(document, f"fold{k}")
        recs = v52.records_for(sorted(files), by_file, images, annotations)
        with torch.no_grad():
            for name, image_id in recs:
                info = images[image_id]
                gray = cv2.imread(str(args.images / name), cv2.IMREAD_GRAYSCALE)
                disk = B.disk_params(gray, np)
                x = (cv2.resize(gray, (imgsz, imgsz), interpolation=cv2.INTER_AREA)
                     if imgsz != FULL else gray)
                xt = torch.from_numpy(x.astype(np.float32) / 255.0)[None].repeat(3, 1, 1).to(device)
                with torch.cuda.amp.autocast():
                    out = det([xt])[0]
                keep = out["scores"].cpu().numpy() >= 0.10
                boxes = out["boxes"].cpu().numpy()[keep] * (FULL / imgsz)
                dsc = out["scores"].cpu().numpy()[keep]
                masks = (out["masks"][:, 0].cpu().numpy()[keep] >= 0.5).astype(np.uint8)
                pmasks, pareas, dscores, crops, feats = [], [], [], [], []
                for b, s, m in zip(boxes, dsc, masks, strict=True):
                    # downscaled bool mask for fast scoring
                    msmall = cv2.resize(m, (SMALL, SMALL),
                                        interpolation=cv2.INTER_NEAREST).astype(bool)
                    area_small = int(msmall.sum())
                    if area_small < MIN_AREA_SMALL:
                        continue
                    # full-res area only for the feature (approx from small * scale^2)
                    mfull_area = area_small * (FULL / SMALL) ** 2
                    pmasks.append(msmall)
                    pareas.append(area_small)
                    dscores.append(float(s))
                    ft = B.candidate_features(np, b, s,
                                              np.zeros((2, 2), np.uint8), gray, disk)
                    ft["mask_area"] = float(mfull_area)
                    ft["log_area"] = float(np.log1p(mfull_area))
                    feats.append([ft[kk] for kk in vkeys])
                    x0, y0, x1, y1 = b
                    cxb, cyb = (x0 + x1) / 2, (y0 + y1) / 2
                    half = max(x1 - x0, y1 - y0) * B.CONTEXT / 2
                    a0, b0 = int(max(0, cxb - half)), int(max(0, cyb - half))
                    c0, d0 = int(min(FULL, cxb + half)), int(min(FULL, cyb + half))
                    cr = gray[b0:d0, a0:c0]
                    cr = (cv2.resize(cr, (B.CROP, B.CROP), interpolation=cv2.INTER_AREA)
                          if cr.size else np.zeros((B.CROP, B.CROP), np.uint8))
                    crops.append(cr.astype(np.uint8))
                # verifier scores
                if crops:
                    ci = torch.tensor(np.stack(crops), dtype=torch.float32)[:, None]
                    ci = (ci / 255.0).repeat(1, 3, 1, 1).to(device)
                    fzn = (np.asarray(feats) - vmean) / vstd
                    fz = torch.tensor(fzn, dtype=torch.float32).to(device)
                    with torch.no_grad():
                        vs = torch.sigmoid(vmodel(ci, fz)).cpu().numpy()
                else:
                    vs = np.zeros(0)
                gt_rles = [v52.annotation_rle(mask_utils, info, e) for e in annotations[image_id]]
                gmasks = []
                for r in gt_rles:
                    if mask_utils.area(r) <= 0:
                        continue
                    gm = mask_utils.decode(r)
                    if gm.ndim == 3:
                        gm = gm[..., 0]
                    gmasks.append(cv2.resize(gm, (SMALL, SMALL),
                                             interpolation=cv2.INTER_NEAREST).astype(bool))
                per_image.append((np.asarray(dscores), np.asarray(vs), pmasks, pareas, gmasks))
        print(f"fold{k} collected", flush=True)
    return per_image


def score_policy(per_image, selector):
    """Fast PQ: operates on pre-decoded downscaled boolean masks (no pycocotools re-decode).
    per_image rows: (dsc, vsc, pred_masks[list of small bool arrays], pred_areas, gt_masks)."""
    iou_sum = tp = fp = fn = 0
    for dsc, vsc, pmasks, _pa, gmasks in per_image:
        kept_idx = selector(dsc, vsc)
        order = sorted(kept_idx, key=lambda i: -dsc[i])
        owned = None
        kept = []
        for i in order:
            m = pmasks[i]
            if owned is None:
                owned = np.zeros_like(m)
            m = m & ~owned
            if int(m.sum()) < MIN_AREA_SMALL:
                continue
            owned |= m
            kept.append(m)
        if not kept and not gmasks:
            continue
        if not kept:
            fn += len(gmasks)
            continue
        if not gmasks:
            fp += len(kept)
            continue
        # IoU matrix from small boolean arrays
        kflat = np.stack([m.ravel() for m in kept]).astype(np.float32)
        gflat = np.stack([g.ravel() for g in gmasks]).astype(np.float32)
        inter = kflat @ gflat.T
        ka = kflat.sum(1)[:, None]
        ga = gflat.sum(1)[None, :]
        iou = inter / (ka + ga - inter + 1e-9)
        hits = np.argwhere(iou > 0.5)
        ordered = sorted(hits.tolist(), key=lambda pg: (-iou[pg[0], pg[1]], pg))
        up, ug = set(), set()
        for p, g in ordered:
            if p not in up and g not in ug:
                up.add(p)
                ug.add(g)
                iou_sum += float(iou[p, g])
        tp += len(up)
        fp += len(kept) - len(up)
        fn += len(gmasks) - len(ug)
    return pq(iou_sum, tp, fp, fn), tp, fp, fn


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verifier", type=Path, required=True)
    ap.add_argument("--annotations", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if v52.sha256(args.annotations) != v56.ANNOTATION_SHA256:
        raise RuntimeError("annotation identity changed")
    per_image = collect(args)

    results = {}
    # 1) detector-only baseline
    best_base = (-1, None)
    for c in DET_CS:
        def det_sel(d, v, c=c):
            return list(np.where(d >= c)[0])
        p, tp, fp, fn = score_policy(per_image, det_sel)
        results[f"det_c{c:.2f}"] = {"pq": p, "tp": tp, "fp": fp, "fn": fn}
        if p > best_base[0]:
            best_base = (p, f"det_c{c:.2f}")
    # 2) combined logit score
    best_comb = (-1, None)
    for a in ALPHAS:
        for cut in CUTOFFS:
            def sel(d, v, a=a, cut=cut):
                S = logit(d) + a * logit(v) if len(v) else np.zeros(0)
                return list(np.where(S >= cut)[0])
            p, tp, fp, fn = score_policy(per_image, sel)
            results[f"comb_a{a}_cut{cut}"] = {"pq": p, "tp": tp, "fp": fp, "fn": fn}
            if p > best_comb[0]:
                best_comb = (p, f"comb_a{a}_cut{cut}")
    # 3) tiered gate
    best_tier = (-1, None)
    for vth in VTHS:
        def sel(d, v, vth=vth):
            if len(v) == 0:
                return []
            k = (d >= HI) | ((d >= LO) & (d < HI) & (v >= vth))
            return list(np.where(k)[0])
        p, tp, fp, fn = score_policy(per_image, sel)
        results[f"tier_vth{vth}"] = {"pq": p, "tp": tp, "fp": fp, "fn": fn}
        if p > best_tier[0]:
            best_tier = (p, f"tier_vth{vth}")

    best_overall = max(best_comb, best_tier)
    delta = best_overall[0] - best_base[0]
    verdict = ("dead_archive" if delta < 0.003 else "polish_only" if delta < 0.008
               else "keep" if delta < 0.015 else "investigate")
    report = {"best_detector_only": {"policy": best_base[1], "pq": best_base[0]},
              "best_combined": {"policy": best_comb[1], "pq": best_comb[0]},
              "best_tiered": {"policy": best_tier[1], "pq": best_tier[0]},
              "delta_vs_detector_only": delta, "verdict": verdict,
              "all": results, "reads_test": False}
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("best_detector_only", "best_combined",
                                             "best_tiered", "delta_vs_detector_only",
                                             "verdict")}, indent=2))


if __name__ == "__main__":
    main()
