"""V53 pre-flight: measure whether V52 (clDice semantic U-Net) errors are
DECORRELATED from the V17 ensemble — the only condition under which fusing V52
as an extra source can beat V17's 0.448703 (project ledger E028; V51 addendum).

Mechanism being tested (SESSION_RESUME_2026-10-01 sec.4): every YOLO-family source
shares correlated false positives with V17, so NMS/agreement fusion never gains.
V52 is a different architecture (semantic U-Net) AND objective (soft-clDice), so its
FPs *might* fall elsewhere. This script quantifies that BEFORE spending the full
fusion-grid GPU budget.

It reproduces V17's exact selected proposals per tune file (seed1/mr1_iou030 ->
V11 refiner -> t0.45_c0.30, i.e. the V50 pipeline), loads V52's dumped instance
proposals (/mnt/v52/cache/<set>/<file>.json.gz), classifies every proposal of each
source as TP/FP against GT (greedy IoU>0.5, one-to-one per source, project
semantics), and reports:

  * standalone TP/FP/FN for V17 and V52 on the shared 94-file fusion-tune split
  * FP correlation: fraction of V52 FPs that overlap (IoU>=0.5) some V17 FP
    (high => correlated => fusion dead; low => orthogonal => worth the grid)
  * complementary recall: GT instances V17 misses (FN) that V52 recovers (TP)
    = the realizable recall upside a union fusion could add

Firewall: uses tune GT only to classify TP/FP for the diagnostic (same as a tune
eval). No test inference, no CSV, no submission. Read-only w.r.t. all weights/caches.

  CUDA_VISIBLE_DEVICES=0 python scripts/diagnose_v52_v17_overlap.py \
      --v52-cache /mnt/v52/cache/w1_t060 --out outputs/v53_diagnostic/w1_t060.json
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import reproduce_v17_from_cache as v50

V17_SEED, V17_CELL = "seed1", "mr1_iou030"
V17_T, V17_C = 0.45, 0.30
MATCH_IOU = 0.5


def load_v52_dump(cache_dir: Path, name: str):
    """Return (rles, scores) for one tune file from a V52 proposal dump."""
    path = cache_dir / (name + ".json.gz")
    with gzip.open(path, "rt") as stream:
        raw = json.load(stream)
    if raw["file"] != name:
        raise RuntimeError(f"V52 dump file mismatch: {path} -> {raw['file']}")
    rles = [dict(size=list(r["size"]), counts=r["counts"].encode()) for r in raw["rles"]]
    return rles, list(raw["scores"]), float(raw["threshold"])


def classify(mask_utils, pred_rles, gt_rles):
    """Greedy one-to-one IoU>0.5 match. Return (tp_pred_idx set, fp_pred_idx list,
    matched_gt_idx set) — project pq_eval semantics."""
    if not pred_rles:
        return set(), [], set()
    if not gt_rles:
        return set(), list(range(len(pred_rles))), set()
    iou = np.asarray(mask_utils.iou(pred_rles, gt_rles, [0] * len(gt_rles)), dtype=float)
    hits = np.argwhere(iou > MATCH_IOU)
    ordered = sorted(hits.tolist(), key=lambda pg: (-iou[pg[0], pg[1]], pg))
    used_pred, used_gt = set(), set()
    for p, g in ordered:
        if p not in used_pred and g not in used_gt:
            used_pred.add(p)
            used_gt.add(g)
    fp = [p for p in range(len(pred_rles)) if p not in used_pred]
    return used_pred, fp, used_gt


def fp_overlap_fraction(mask_utils, v52_fp_rles, v17_fp_rles):
    """Fraction of V52 FP proposals that overlap (IoU>=0.5) any V17 FP proposal."""
    if not v52_fp_rles:
        return 0, 0
    if not v17_fp_rles:
        return 0, len(v52_fp_rles)
    iou = np.asarray(mask_utils.iou(v52_fp_rles, v17_fp_rles, [0] * len(v17_fp_rles)), dtype=float)
    correlated = int((iou >= MATCH_IOU).any(axis=1).sum())
    return correlated, len(v52_fp_rles)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v52-cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from pycocotools import mask as mask_utils

    rt = v50.runtime_state()
    core, tta, base = rt["core"], rt["tta"], rt["base"]
    _, tune, _, _, by_file = rt["split"]  # tune = 94-file fusion split (V50 semantics)
    names = sorted(tune)

    agg = dict(
        n_files=len(names),
        v17=dict(tp=0, fp=0, fn=0),
        v52=dict(tp=0, fp=0, fn=0),
        v52_fp_total=0,
        v52_fp_correlated_with_v17_fp=0,
        v17_fn_total=0,
        v17_fn_recovered_by_v52=0,
        v52_threshold=None,
    )

    import cv2

    for index, name in enumerate(names, 1):
        gray = cv2.imread(str(base.CFG.train_dir / name), cv2.IMREAD_GRAYSCALE)

        # --- V17 exact selected proposals (seed1/mr1_iou030 -> V11 refiner -> t/c) ---
        views = v50.read_signed_views(name, "train", V17_CELL, v50.DETECTORS[V17_SEED])
        proposals, scores = tta.cluster_views(views, v50.POLICY, v50.MATCH_IOU)
        if proposals:
            masks = [mask_utils.decode(r) for r in proposals]
            bounds, _, probs = core.crop_probabilities(base, rt["model"], gray, masks, True)
            refined_rles, refined_scores = core.clean_refined_predictions(
                masks, scores, bounds, probs, V17_T, v50.MODE)
            keep = np.asarray(refined_scores) >= V17_C if len(refined_scores) else np.zeros(0, bool)
            v17_rles = [r for r, f in zip(refined_rles, keep, strict=True) if f]
        else:
            v17_rles = []

        # --- V52 dumped proposals ---
        v52_rles, _v52_scores, thr = load_v52_dump(args.v52_cache, name)
        agg["v52_threshold"] = thr

        # --- GT: project convention = max-instance annotator record per file ---
        image_ids = sorted(by_file[name])
        best_id = max(image_ids,
                      key=lambda v: (len(rt["annotations"].get(v, [])),
                                     sum(e.get("area", 0) for e in rt["annotations"].get(v, [])),
                                     v))
        info = rt["images"][best_id]
        gt_rles = [core.annotation_rle(info, e) for e in rt["annotations"].get(best_id, [])]
        gt_rles = [r for r in gt_rles if mask_utils.area(r) > 0]

        v17_tp, v17_fp, v17_gt = classify(mask_utils, v17_rles, gt_rles)
        v52_tp, v52_fp, v52_gt = classify(mask_utils, v52_rles, gt_rles)

        agg["v17"]["tp"] += len(v17_tp)
        agg["v17"]["fp"] += len(v17_fp)
        agg["v17"]["fn"] += len(gt_rles) - len(v17_gt)
        agg["v52"]["tp"] += len(v52_tp)
        agg["v52"]["fp"] += len(v52_fp)
        agg["v52"]["fn"] += len(gt_rles) - len(v52_gt)

        # FP correlation
        corr, total = fp_overlap_fraction(mask_utils, [v52_rles[i] for i in v52_fp],
                                          [v17_rles[i] for i in v17_fp])
        agg["v52_fp_correlated_with_v17_fp"] += corr
        agg["v52_fp_total"] += total

        # complementary recall: GT missed by V17 but recovered by V52
        v17_missed_gt = set(range(len(gt_rles))) - v17_gt
        recovered = v17_missed_gt & v52_gt
        agg["v17_fn_total"] += len(v17_missed_gt)
        agg["v17_fn_recovered_by_v52"] += len(recovered)

        print(f"{index}/{len(names)} {name} "
              f"v17[tp{len(v17_tp)} fp{len(v17_fp)}] v52[tp{len(v52_tp)} fp{len(v52_fp)}]",
              flush=True)

    agg["v52_fp_correlation_rate"] = (
        agg["v52_fp_correlated_with_v17_fp"] / agg["v52_fp_total"]
        if agg["v52_fp_total"] else 0.0)
    agg["v17_fn_recovery_rate"] = (
        agg["v17_fn_recovered_by_v52"] / agg["v17_fn_total"]
        if agg["v17_fn_total"] else 0.0)
    agg["reads_tune"] = True
    agg["reads_test"] = False

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(agg, indent=2) + "\n")
    print(json.dumps({k: v for k, v in agg.items()
                      if k not in ("v17", "v52")}, indent=2))
    print("V17 counts:", agg["v17"])
    print("V52 counts:", agg["v52"])


if __name__ == "__main__":
    main()
