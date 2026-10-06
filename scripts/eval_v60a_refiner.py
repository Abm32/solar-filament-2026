"""V60A refiner EVAL (v2): does the correction-needed head give the refiner SELECTIVITY?

Reservoir (diagnose_v60_maskquality): V17 candidates matched to GT at IoU 0.3-0.5
(under-segmented, median pred/GT area 0.686). The refiner transforms an EXISTING V17 candidate
mask; it NEVER creates an instance, so it is FP-safe by construction.

v1 (under-seg-only training) finding: pushed 17/132 reservoir over 0.5 but DAMAGED 172/607
good masks; no blind area-growth rule helped. v2 adds a learned correction-needed head and a
mixed identity/degraded training distribution. This eval asks the causal question:

  (1) SEPARABILITY: does corr_prob separate good (IoU>=0.5) from reservoir (0.3-0.5) V17
      candidates?  -> AUC(corr; reservoir vs good) + mean corr per population.
  (2) DAMAGE CURVE: under the gate  corr >= t  AND  refined-confidence >= c  how many good
      masks are still damaged (IoU drops below 0.5), and how many reservoir are pushed?
  (3) Selective-replacement PQ (same 512px V17-identical per-annotator harness, bar 0.4487).

The replacement gate uses ONLY model outputs (corr prob, refined-mask confidence) - never GT and
never area growth. GT is used only to score (populations, IoU, PQ).

Decision gate (user): CONTINUE iff good-mask damage falls dramatically AND reservoir recovery
stays near/above the v1 17/132. KILL iff it still damages a large fraction of the good masks or
the correction head cannot distinguish the two populations.

Firewall: V17 from signed caches; refiner FIT-only (tune held out of all folds). Tune labels
only for scoring. No test, no submission.

  CUDA_VISIBLE_DEVICES=N python scripts/eval_v60a_refiner.py \
      --weights outputs/v60a_mix_f0/weights/v60a_refiner.pt \
      --annotations <ann> --out outputs/v60a_mix_eval.json
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
import train_v52_cldice_semantic as v52  # noqa: E402
import train_v56_maskrcnn as v56  # noqa: E402
import train_v60a_refiner as R  # noqa: E402

FULL = 2048
SMALL = 512
T_REFINE = 0.45
V17_SEED, V17_CELL = "seed1", "mr1_iou030"
CORR_THRS = (0.5, 0.7, 0.8, 0.9)        # gate: replace only if P(needs correction) >= t
CONF_MINS = (0.0, 0.7, 0.8)             # ... and refined-mask mean prob >= c (model's own conf)


def pq(iou_sum, tp, fp, fn):
    d = tp + 0.5 * (fp + fn)
    return float(iou_sum / d) if d else 0.0


def _iou(a, b):
    inter = float((a & b).sum())
    union = float((a | b).sum())
    return inter / union if union else 0.0


def auc(pos, neg):
    """Mann-Whitney AUC: P(score_pos > score_neg) with ties = 0.5. None if a class is empty."""
    if len(pos) == 0 or len(neg) == 0:
        return None
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def refine_mask(torch, cv2, model, device, gray, full_mask, crop, net):
    """Apply the refiner to one V17 candidate: crop around it, seed=the candidate, -> refined.

    Returns (refined_full_mask, P(seed needs correction), refined-mask mean probability).
    """
    ys, xs = np.nonzero(full_mask)
    if len(xs) == 0:
        return full_mask, 0.0, 0.0
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    half = crop / 2
    ox = int(np.clip(cx - half, 0, FULL - crop))
    oy = int(np.clip(cy - half, 0, FULL - crop))
    g = gray[oy:oy + crop, ox:ox + crop]
    seed = full_mask[oy:oy + crop, ox:ox + crop].astype(np.uint8)
    g = cv2.resize(g, (net, net), interpolation=cv2.INTER_AREA)
    seed = cv2.resize(seed, (net, net), interpolation=cv2.INTER_NEAREST)
    x = np.stack([g.astype(np.float32) / 255.0, seed.astype(np.float32)])
    xt = torch.from_numpy(x)[None].to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        ml, _sl, cl = model(xt)
    prob = torch.sigmoid(ml)[0, 0].float().cpu().numpy()
    pr = (prob >= 0.5).astype(np.uint8)
    conf = float(prob[pr > 0].mean()) if pr.any() else 0.0
    corr = float(torch.sigmoid(cl.float())[0])
    pr = cv2.resize(pr, (crop, crop), interpolation=cv2.INTER_NEAREST)
    out = np.zeros((FULL, FULL), np.uint8)
    out[oy:oy + crop, ox:ox + crop] = pr
    return out, corr, conf


def collect(args):
    """Per tune file: V17 masks (full-res) + refiner outputs (+corr/conf) + GT (full + small)."""
    import cv2
    import torch
    from pycocotools import mask as mask_utils

    rt = v50.runtime_state()
    core, tta, base = rt["core"], rt["tta"], rt["base"]
    _, tune, _, _, by_file = rt["split"]
    device = torch.device("cuda:0")

    ck = torch.load(args.weights, map_location=device, weights_only=False)
    model = R.build_unet(torch, corr_head=bool(ck.get("corr_head", False))).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    crop, net = ck.get("crop", 1024), ck.get("net", R.NET)

    per_image = []
    for name in sorted(tune):
        gray = cv2.imread(str(base.CFG.train_dir / name), cv2.IMREAD_GRAYSCALE)
        views = v50.read_signed_views(name, "train", V17_CELL, v50.DETECTORS[V17_SEED])
        proposals, scores = tta.cluster_views(views, v50.POLICY, v50.MATCH_IOU)
        v17_full, refined_full, corr_p, conf_p = [], [], [], []
        if proposals:
            masks = [mask_utils.decode(r) for r in proposals]
            bounds, _, probs = core.crop_probabilities(base, rt["model"], gray, masks, True)
            rles, _dsc = core.clean_refined_predictions(masks, scores, bounds, probs,
                                                        T_REFINE, v50.MODE)
            for r in rles:
                mm = mask_utils.decode(r)
                if mm.ndim == 3:
                    mm = mm[..., 0]
                if int(mm.sum()) < 2:
                    continue
                v17_full.append(mm.astype(np.uint8))
                rf, cp, cf = refine_mask(torch, cv2, model, device, gray,
                                         mm.astype(np.uint8), crop, net)
                refined_full.append(rf)
                corr_p.append(cp)
                conf_p.append(cf)
        gt_full, gt_small = [], []
        for image_id in sorted(by_file[name]):
            info = rt["images"][image_id]
            for e in rt["annotations"].get(image_id, []):
                rr = core.annotation_rle(info, e)
                if mask_utils.area(rr) > 0:
                    dd = mask_utils.decode(rr)
                    if dd.ndim == 3:
                        dd = dd[..., 0]
                    gt_full.append((image_id, dd.astype(np.uint8)))
                    gt_small.append((image_id, cv2.resize(dd, (SMALL, SMALL),
                                     interpolation=cv2.INTER_NEAREST).astype(bool)))
        gt_sets = {}
        for image_id in sorted(by_file[name]):
            gt_sets[image_id] = [m for (iid, m) in gt_small if iid == image_id]
        per_image.append({"v17": v17_full, "refined": refined_full,
                          "corr": corr_p, "conf": conf_p,
                          "gt_full": gt_full, "gt_sets": gt_sets})
        print(f"collected {name} v17={len(v17_full)}", flush=True)
    return per_image, cv2


def candidate_table(per_image):
    """One row per V17 candidate: IoU before/after vs its best GT (full-res), population, model
    outputs. Computed ONCE; every gate is then evaluated by cheap filtering of this table."""
    rows = []
    for di, d in enumerate(per_image):
        corr = d.get("corr") or [0.0] * len(d["v17"])
        conf = d.get("conf") or [0.0] * len(d["v17"])
        for ci, (v, rf) in enumerate(zip(d["v17"], d["refined"], strict=True)):
            best_iou_v, best_g = 0.0, None
            for (_iid, g) in d["gt_full"]:
                iv = _iou(v.astype(bool), g.astype(bool))
                if iv > best_iou_v:
                    best_iou_v, best_g = iv, g
            if best_g is None:
                continue
            iou_r = _iou(rf.astype(bool), best_g.astype(bool))
            pop = ("reservoir" if 0.3 <= best_iou_v < 0.5
                   else "good" if best_iou_v >= 0.5 else "low")
            rows.append({"di": di, "ci": ci, "iou_v": best_iou_v, "iou_r": iou_r, "pop": pop,
                         "corr": float(corr[ci]), "conf": float(conf[ci]),
                         "area_v": float(v.sum() / max(best_g.sum(), 1)),
                         "area_r": float(rf.sum() / max(best_g.sum(), 1))})
    return rows


def gate_metrics(rows, corr_t=None, conf_c=None):
    """Reservoir/good outcomes if we REPLACE only rows passing the gate (None = replace all)."""
    def passes(r):
        if corr_t is None:
            return True
        return r["corr"] >= corr_t and r["conf"] >= conf_c
    res = [r for r in rows if r["pop"] == "reservoir"]
    good = [r for r in rows if r["pop"] == "good"]
    pushed = sum(1 for r in res if passes(r) and r["iou_r"] >= 0.5)
    damaged = sum(1 for r in good if passes(r) and r["iou_r"] < 0.5)
    replaced_good = sum(1 for r in good if passes(r))
    replaced_res = sum(1 for r in res if passes(r))
    gains = [r["iou_r"] - r["iou_v"] for r in res if passes(r)]
    return {"reservoir_n": len(res), "reservoir_replaced": replaced_res,
            "pushed_to_0.5": pushed,
            "median_iou_gain_on_replaced": float(np.median(gains)) if gains else 0.0,
            "good_n": len(good), "good_replaced": replaced_good, "good_damaged": damaged}


def separability(rows):
    res = [r["corr"] for r in rows if r["pop"] == "reservoir"]
    good = [r["corr"] for r in rows if r["pop"] == "good"]
    return {"auc_reservoir_vs_good": auc(res, good),
            "mean_corr_reservoir": float(np.mean(res)) if res else None,
            "mean_corr_good": float(np.mean(good)) if good else None}


def score_selective(per_image, cv2, rule, baseline=False):
    """Selective-replacement PQ (512px, V17-identical per-annotator).

    rule(v, rf, corr, conf) -> bool decides per candidate (model outputs only, no GT).
    """
    iou_sum = tp = fp = fn = 0
    for d in per_image:
        corr = d.get("corr") or [0.0] * len(d["v17"])
        conf = d.get("conf") or [0.0] * len(d["v17"])
        kept_small = []
        for ci, (v, rf) in enumerate(zip(d["v17"], d["refined"], strict=True)):
            use = v
            if not baseline and rule is not None and rule(v, rf, corr[ci], conf[ci]):
                use = rf
            sm = cv2.resize(use, (SMALL, SMALL), interpolation=cv2.INTER_NEAREST).astype(bool)
            if int(sm.sum()) >= 1:
                kept_small.append(sm)
        owned = None
        kept2 = []
        for m in kept_small:
            if owned is None:
                owned = np.zeros_like(m)
            mm = m & ~owned
            if int(mm.sum()) < 1:
                continue
            owned |= mm
            kept2.append(mm)
        kf = (np.stack([m.ravel() for m in kept2]).astype(np.float32)
              if kept2 else np.zeros((0, 1), np.float32))
        for _image_id, gmasks in d["gt_sets"].items():
            if not kept2 and not gmasks:
                continue
            if not kept2:
                fn += len(gmasks)
                continue
            if not gmasks:
                fp += len(kept2)
                continue
            gf = np.stack([g.ravel() for g in gmasks]).astype(np.float32)
            iou = kf @ gf.T
            iou = iou / (kf.sum(1)[:, None] + gf.sum(1)[None, :] - iou + 1e-9)
            hits = sorted(np.argwhere(iou > 0.5).tolist(), key=lambda pg: -iou[pg[0], pg[1]])
            up, ug = set(), set()
            for p, g in hits:
                if p not in up and g not in ug:
                    up.add(p)
                    ug.add(g)
                    iou_sum += float(iou[p, g])
            tp += len(up)
            fp += len(kept2) - len(up)
            fn += len(gmasks) - len(ug)
    return {"pq": pq(iou_sum, tp, fp, fn), "tp": tp, "fp": fp, "fn": fn}


def verdict(sep, gated, v1_pushed=17, v1_damaged=172):
    """User gate. CONTINUE iff some gate has damage << v1 AND pushed near/above v1; KILL else."""
    best = None
    for tag, m in gated.items():
        if tag == "replace_all":
            continue
        ok_damage = m["good_damaged"] <= max(10, 0.1 * v1_damaged)      # dramatic fall
        ok_push = m["pushed_to_0.5"] >= 0.8 * v1_pushed                 # near/above v1
        if ok_damage and ok_push and (
                best is None or m["pushed_to_0.5"] > gated[best]["pushed_to_0.5"]):
            best = tag
    a = sep["auc_reservoir_vs_good"]
    if best is not None:
        return {"verdict": "continue", "gate": best}
    if a is not None and a < 0.6:
        return {"verdict": "kill_head_cannot_separate", "gate": None}
    return {"verdict": "kill_still_damages_good", "gate": None}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--annotations", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if v52.sha256(args.annotations) != v56.ANNOTATION_SHA256:
        raise RuntimeError("annotation identity changed")

    per_image, cv2 = collect(args)
    rows = candidate_table(per_image)
    sep = separability(rows)

    gated = {"replace_all": gate_metrics(rows)}
    pqs = {}
    baseline = score_selective(per_image, cv2, None, baseline=True)
    pqs["replace_all"] = score_selective(per_image, cv2, lambda v, rf, c, f: True)
    for t in CORR_THRS:
        for c in CONF_MINS:
            tag = f"corr{t}_conf{c}"
            gated[tag] = gate_metrics(rows, t, c)
            pqs[tag] = score_selective(
                per_image, cv2, lambda v, rf, cp, cf, t=t, c=c: cp >= t and cf >= c)
    for r in pqs.values():
        r["delta"] = round(r["pq"] - baseline["pq"], 4)
    best_tag = max(pqs, key=lambda k: pqs[k]["pq"])
    report = {"separability": sep, "baseline_v17_pq": baseline,
              "gated_outcomes": gated, "gated_pq": pqs,
              "best_pq_gate": best_tag, "best_pq": pqs[best_tag],
              "pq_vs_bar_0.4487": round(pqs[best_tag]["pq"] - 0.4487, 4),
              **verdict(sep, gated), "v1_reference": {"pushed": 17, "good_damaged": 172},
              "reads_test": False}
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("separability", "baseline_v17_pq", "best_pq_gate",
                                             "best_pq", "verdict", "gate")}, indent=2))
    print(json.dumps(gated, indent=2))


if __name__ == "__main__":
    main()
