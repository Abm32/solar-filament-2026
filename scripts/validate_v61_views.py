"""V61 gate 0: does MY view generation + consensus reproduce V17's signed caches and PQ?

Loads the pinned V16 seed-1 detector (sha ae476e66...), regenerates the four flip views at
V17's exact cell (mr1_iou030: retina_masks, agnostic_nms, iou 0.3, imgsz 1024, conf floor 0.1)
for N tune files, and compares against the signed caches:
    * per view: proposal count, score vector, and mask IoU of each matched pair
    * after consensus: cluster count and consensus scores
    * direct (no-refiner) pixel-owned PQ on those files, from cache vs from regeneration

PASS = counts identical, scores within 1e-3, every regenerated mask matches its cached mask at
IoU >= 0.99, and the two PQ computations agree exactly on counts. Numerics may differ by GPU/
driver; any mismatch is reported, not hidden, and blocks the OOF program until explained.

  CUDA_VISIBLE_DEVICES=0 python scripts/validate_v61_views.py --n 6 --out outputs/v61_validate.json
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
import v61_toolkit as K  # noqa: E402

DETECTOR = Path("outputs/v16_arm_r1_seed1/detector.pt")
SEED, CELL = "seed1", "mr1_iou030"


def compare_views(cached, regen):
    from pycocotools import mask as mask_utils

    rep = []
    for vi, ((cr, cs), (rr, rs)) in enumerate(zip(cached, regen, strict=True)):
        row = {"view": vi, "n_cached": len(cr), "n_regen": len(rr)}
        if len(cr) == len(rr) and len(cr):
            order_c = np.argsort(-np.asarray(cs), kind="stable")
            order_r = np.argsort(-np.asarray(rs), kind="stable")
            cs_s = np.asarray(cs)[order_c]
            rs_s = np.asarray(rs)[order_r]
            row["max_score_diff"] = float(np.abs(cs_s - rs_s).max())
            ious = mask_utils.iou([rr[i] for i in order_r], [cr[i] for i in order_c],
                                  [0] * len(cr))
            row["min_matched_iou"] = float(np.diag(ious).min())
        rep.append(row)
    return rep


def main():
    import cv2
    from pycocotools import mask as mask_utils
    from ultralytics import YOLO

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if v50.sha256(DETECTOR) != v50.DETECTORS[SEED]:
        raise RuntimeError("detector identity changed")
    rt = v50.runtime_state()
    core, base = rt["core"], rt["base"]
    _, tune, _, _, by_file = rt["split"]
    names = sorted(tune)[::max(1, len(tune) // args.n)][:args.n]
    model = YOLO(str(DETECTOR))

    files, cached_tot, regen_tot = [], np.zeros((len(K.CONFIDENCES), 4)), np.zeros(
        (len(K.CONFIDENCES), 4))
    worst_iou, worst_score, count_mismatch = 1.0, 0.0, 0
    for name in names:
        path = base.CFG.train_dir / name
        cached = v50.read_signed_views(name, "train", CELL, v50.DETECTORS[SEED])
        regen = K.generate_views(model, path, iou=0.3)
        rep = compare_views(cached, regen)
        for r in rep:
            count_mismatch += int(r["n_cached"] != r["n_regen"])
            worst_iou = min(worst_iou, r.get("min_matched_iou", 1.0))
            worst_score = max(worst_score, r.get("max_score_diff", 0.0))
        gt_by_record = []
        for image_id in sorted(by_file[name]):
            info = rt["images"][image_id]
            gt = [core.annotation_rle(info, e) for e in rt["annotations"].get(image_id, [])]
            gt_by_record.append([g for g in gt if mask_utils.area(g) > 0])
        totals = {}
        for tag, views in (("cached", cached), ("regen", regen)):
            rles, scores = K.cluster_views(views)
            rles, scores = K.own_pixels(rles, scores)
            totals[tag] = K.score_file(rles, scores, gt_by_record)
        for ci, c in enumerate(K.CONFIDENCES):
            cached_tot[ci] += totals["cached"][c]
            regen_tot[ci] += totals["regen"][c]
        files.append({"name": name, "views": rep})
        print(f"{name} views={[(r['n_cached'], r['n_regen']) for r in rep]}", flush=True)
        _ = cv2

    pq_rows = []
    for ci, c in enumerate(K.CONFIDENCES):
        pq_rows.append({"conf": c, "cached_counts": cached_tot[ci].tolist(),
                        "regen_counts": regen_tot[ci].tolist(),
                        "cached_pq": K.pq_value(cached_tot[ci]),
                        "regen_pq": K.pq_value(regen_tot[ci]),
                        "counts_equal": bool(np.allclose(cached_tot[ci], regen_tot[ci]))})
    passed = (count_mismatch == 0 and worst_iou >= 0.99 and worst_score <= 1e-3
              and all(r["counts_equal"] for r in pq_rows))
    report = {"n_files": len(names), "count_mismatch_views": count_mismatch,
              "worst_matched_iou": worst_iou, "worst_score_diff": worst_score,
              "pq": pq_rows, "passed": passed, "files": files}
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
