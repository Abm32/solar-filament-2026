"""V59-0: the two decisive cheap diagnostics BEFORE any topology training.

Per the plan: don't declare the 440 V17 FN "irreducible label noise" by inference. Measure.

(1) PROPOSAL-ORACLE RECALL — ignore confidence entirely. For every GT filament, is ANY
    V17 proposal (full pool, pre-c-cutoff) a match at IoU>0.5 / 0.4 / 0.3? Broken down by
    thin / small / faint / normal. Tells us whether missed GT are already-in-the-pool
    (score/topology problem) or never-proposed (detector/representation problem).

(2) AGGRESSIVE-THRESHOLD STRUCTURAL CURVE — sweep V17 confidence DOWN (0.30..0.05).
    For each: recall, thin-recall, TP, FP, FN, fragmentation (one-to-many), merge
    (many-to-one), PQ. Tells us whether extra recall at low threshold comes as recoverable
    TP + topology errors (spine can help) or pure spurious FP (spine can't help).

Both from V17's exact proposal pool (signed caches -> V11 refiner at t0.45, scores kept),
94-file tune, per-annotator GT, 512px for speed. Firewall: tune labels for scoring only;
no test, no submission, no training.

  CUDA_VISIBLE_DEVICES=N python scripts/diagnose_v59_0_recall.py --out outputs/v59_0.json
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

FULL, SMALL = 2048, 512
T_REFINE = 0.45
V17_SEED, V17_CELL = "seed1", "mr1_iou030"
CS = (0.30, 0.25, 0.20, 0.15, 0.10, 0.07, 0.05)
SMALL_AREA_FULL = 1000               # <1000 px (full-res) = "small" (41% of GT per public analysis)
THIN_ELONG = 4.0
FAINT_PCTL = 25                      # faint = GT mean intensity in bottom quartile


def to_small(cv2, np_, m):
    if m.ndim == 3:
        m = m[..., 0]
    return cv2.resize(m, (SMALL, SMALL), interpolation=cv2.INTER_NEAREST).astype(bool)


def gt_class(np_, gm_small, gray_small):
    area_full = gm_small.sum() * (FULL / SMALL) ** 2
    ys, xs = np.nonzero(gm_small)
    if len(xs):
        w, h = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
        elong = max(w, h) / max(min(w, h), 1)
        meani = float(gray_small[gm_small].mean())
    else:
        elong, meani = 1.0, 255.0
    return area_full, elong, meani


def iou_matrix(np_, preds, gts):
    if not preds or not gts:
        return np.zeros((len(preds), len(gts)))
    pf = np.stack([m.ravel() for m in preds]).astype(np.float32)
    gf = np.stack([g.ravel() for g in gts]).astype(np.float32)
    inter = pf @ gf.T
    return inter / (pf.sum(1)[:, None] + gf.sum(1)[None, :] - inter + 1e-9)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    import cv2
    from pycocotools import mask as mask_utils

    rt = v50.runtime_state()
    core, tta, base = rt["core"], rt["tta"], rt["base"]
    _, tune, _, _, by_file = rt["split"]

    # oracle recall accumulators (by class, by iou thr)
    oracle = {thr: {"all": [0, 0], "thin": [0, 0], "small": [0, 0], "faint": [0, 0],
                    "normal": [0, 0]} for thr in (0.5, 0.4, 0.3)}
    # structural curve per confidence
    curve = {c: {"iou_sum": 0.0, "tp": 0, "fp": 0, "fn": 0, "gt": 0, "thin_gt": 0,
                 "thin_tp": 0, "frag": 0, "merge": 0} for c in CS}

    for name in sorted(tune):
        gray = cv2.imread(str(base.CFG.train_dir / name), cv2.IMREAD_GRAYSCALE)
        gray_s = cv2.resize(gray, (SMALL, SMALL), interpolation=cv2.INTER_AREA)
        faint_cut = np.percentile(gray_s[gray_s > 20], FAINT_PCTL) if (gray_s > 20).any() else 0
        views = v50.read_signed_views(name, "train", V17_CELL, v50.DETECTORS[V17_SEED])
        props, scores = tta.cluster_views(views, v50.POLICY, v50.MATCH_IOU)
        if props:
            masks = [mask_utils.decode(r) for r in props]
            bounds, _, probs = core.crop_probabilities(base, rt["model"], gray, masks, True)
            rles, sc = core.clean_refined_predictions(masks, scores, bounds, probs,
                                                      T_REFINE, v50.MODE)
            pool = [to_small(cv2, np, mask_utils.decode(r)) for r in rles]
            psc = np.asarray(sc, float)
        else:
            pool, psc = [], np.zeros(0)
        for image_id in sorted(by_file[name]):
            info = rt["images"][image_id]
            gts, cls = [], []
            for e in rt["annotations"].get(image_id, []):
                r = core.annotation_rle(info, e)
                if mask_utils.area(r) <= 0:
                    continue
                gm = to_small(cv2, np, mask_utils.decode(r))
                gts.append(gm)
                cls.append(gt_class(np, gm, gray_s))
            if not gts:
                continue
            iou_all = iou_matrix(np, pool, gts)   # [P, G] over FULL pool
            best_per_gt = iou_all.max(0) if len(pool) else np.zeros(len(gts))
            for thr in (0.5, 0.4, 0.3):
                for gi, (area, elong, meani) in enumerate(cls):
                    hit = int(best_per_gt[gi] >= thr) if len(pool) else 0
                    oracle[thr]["all"][0] += hit
                    oracle[thr]["all"][1] += 1
                    kind = "normal"
                    if elong >= THIN_ELONG:
                        kind = "thin"
                    elif area < SMALL_AREA_FULL:
                        kind = "small"
                    elif meani <= faint_cut:
                        kind = "faint"
                    oracle[thr][kind][0] += hit
                    oracle[thr][kind][1] += 1
            # structural curve: for each confidence keep psc>=c, PQ match 0.5
            for c in CS:
                keep = np.where(psc >= c)[0] if len(psc) else np.array([], int)
                kept = [pool[i] for i in keep]
                cu = curve[c]
                cu["gt"] += len(gts)
                cu["thin_gt"] += sum(1 for _, el, _ in cls if el >= THIN_ELONG)
                if not kept:
                    cu["fn"] += len(gts)
                    continue
                im = iou_matrix(np, kept, gts)
                ovl = im >= 0.1
                cu["frag"] += int((ovl.sum(0) >= 2).sum())
                cu["merge"] += int((ovl.sum(1) >= 2).sum())
                hits = sorted(np.argwhere(im > 0.5).tolist(), key=lambda pg: -im[pg[0], pg[1]])
                up, ug = set(), set()
                for p, g in hits:
                    if p not in up and g not in ug:
                        up.add(p)
                        ug.add(g)
                        cu["iou_sum"] += float(im[p, g])
                        if cls[g][1] >= THIN_ELONG:
                            cu["thin_tp"] += 1
                cu["tp"] += len(up)
                cu["fp"] += len(kept) - len(up)
                cu["fn"] += len(gts) - len(ug)
        print(f"v59-0 {name}", flush=True)

    def rec(pair):
        return round(pair[0] / pair[1], 4) if pair[1] else 0.0

    oracle_out = {f"iou{thr}": {k: {"recall": rec(v), "n": v[1]} for k, v in d.items()}
                  for thr, d in oracle.items()}
    curve_out = {}
    for c, cu in curve.items():
        d = cu["tp"] + 0.5 * (cu["fp"] + cu["fn"])
        curve_out[f"c{c:.2f}"] = {
            "pq": round(cu["iou_sum"] / d, 4) if d else 0.0,
            "recall": round(cu["tp"] / (cu["tp"] + cu["fn"]), 4) if cu["tp"] + cu["fn"] else 0,
            "thin_recall": round(cu["thin_tp"] / cu["thin_gt"], 4) if cu["thin_gt"] else 0,
            "tp": cu["tp"], "fp": cu["fp"], "fn": cu["fn"],
            "frag": cu["frag"], "merge": cu["merge"]}
    report = {"proposal_oracle_recall": oracle_out, "threshold_curve": curve_out,
              "reads_test": False}
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
