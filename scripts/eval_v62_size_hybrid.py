"""V62 size-gated hybrid: does the V11 refiner erase p2first's small-object masks? (zero training)

Same 141 canonical held-out files, OOF proposals and V17 cell as `eval_v62_refiner.py`. Per recipe,
three arms through ONE cleaning path (`clean_hybrid`), differing only in which proposals keep their
unrefined flip-consensus mask instead of the V11-refined crop:

    refined    no proposal kept direct (must reproduce outputs/v62_refiner_test.json exactly)
    hybrid     proposals with area < SMALL_AREA kept direct
    unrefined  every proposal kept direct (refiner off, same consensus + overlap cleaning)

Pre-registered (fixed before any run; SMALL_AREA is the existing small-object definition and is
NOT swept):
  PRIMARY   p2first/hybrid - baseline/refined at t0.45/c0.30 (the V17-equivalent pipeline).
  PASS iff  delta >= +0.010 AND file-bootstrap P(delta > 0) >= 0.90.
  Secondary (never gates): p2first/hybrid - baseline/hybrid; SQ (mean matched IoU) and RQ per arm,
  which split the direct gain into mask quality vs detection.

  CUDA_VISIBLE_DEVICES=N python scripts/eval_v62_size_hybrid.py --out outputs/v62_size_hybrid.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS.parent / "kaggle_inference"))
import optimize_inference as core  # noqa: E402

SMALL_AREA = 1000
T, C = 0.45, 0.30
ARMS = ("refined", "hybrid", "unrefined")
RECIPES = ("baseline", "p2first")
PRIMARY = (("baseline", "refined"), ("p2first", "hybrid"))
BAR_DELTA, BAR_P = 0.010, 0.90


def clean_hybrid(masks, scores, bounds, probabilities, threshold, direct):
    """core.clean_refined_predictions(mode='refined_constrained'), except proposals with
    direct[i] keep their own mask instead of the refined crop."""
    if not masks:
        return [], np.asarray([])
    occupied = np.zeros_like(masks[0], dtype=bool)
    kernel = core.cv2.getStructuringElement(core.cv2.MORPH_ELLIPSE, (7, 7))
    rles, kept = [], []
    for i in sorted(range(len(scores)), key=lambda i: -scores[i]):
        x0, y0, x1, y1 = bounds[i]
        proposal = masks[i][y0:y1, x0:x1]
        if direct[i]:
            crop = proposal.astype(bool)
        else:
            crop = core.cv2.resize(probabilities[i], (x1 - x0, y1 - y0)) >= threshold
            crop &= core.cv2.dilate(proposal, kernel) > 0
            if not crop.any():
                crop = proposal.astype(bool)
        crop &= ~occupied[y0:y1, x0:x1]
        if crop.sum() < core.MIN_AREA:
            continue
        occupied[y0:y1, x0:x1] |= crop
        rles.append(core.encode_crop(crop, bounds[i], occupied.shape))
        kept.append(scores[i])
    return rles, np.asarray(kept)


def direct_flags(masks, arm):
    if arm == "refined":
        return [False] * len(masks)
    if arm == "unrefined":
        return [True] * len(masks)
    return [int(m.sum()) < SMALL_AREA for m in masks]


def file_counts(rt, recipe, fold, name, gray, gt_by_image):
    """-> {arm: np.array(4) [iou_sum, tp, fp, fn]} at the V17 cell."""
    import eval_v61_oof as O
    import reproduce_v17_from_cache as v50
    import v61_toolkit as K
    from pycocotools import mask as mask_utils

    views = K.read_cache_views(O.cache_path(recipe, fold, name))
    proposals, scores = rt["tta"].cluster_views(views, v50.POLICY, v50.MATCH_IOU)
    masks = [mask_utils.decode(r) for r in proposals]
    bounds, probs = [], []
    if masks:
        bounds, _, probs = core.crop_probabilities(rt["base"], rt["model"], gray, masks, True)
    out = {}
    for arm in ARMS:
        rles, final = clean_hybrid(masks, scores, bounds, probs, T, direct_flags(masks, arm))
        kept = [r for r, s in zip(rles, final, strict=True) if s >= C]
        total = np.zeros(4)
        for gt in gt_by_image.values():
            iou = (mask_utils.iou(kept, gt, [0] * len(gt)).T if kept and gt
                   else np.zeros((len(gt), len(kept))))
            total += core.pq_counts(iou, len(gt), len(kept))
        out[arm] = total
    return out


def summary(tot):
    iou, tp, fp, fn = tot
    return {"pq": core.pq_value(tot), "sq": float(iou / tp) if tp else 0.0,
            "rq": float(tp / (tp + 0.5 * (fp + fn))) if tp + fp + fn else 0.0,
            "tp": int(tp), "fp": int(fp), "fn": int(fn)}


def paired(a, b, seed=62, n=2000):
    """File bootstrap of PQ(b) - PQ(a); a, b are (files, 4) count arrays."""
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        idx = rng.integers(0, len(a), len(a))
        d.append(core.pq_value(b[idx].sum(0)) - core.pq_value(a[idx].sum(0)))
    return {"delta": core.pq_value(b.sum(0)) - core.pq_value(a.sum(0)),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_gt_0": float(np.mean(np.asarray(d) > 0))}


def main():
    import cv2
    import eval_v62_refiner as R
    import reproduce_v17_from_cache as v50
    from pycocotools import mask as mask_utils

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N files only")
    args = ap.parse_args()

    rt = v50.runtime_state()
    held, _, _, _, by_file = rt["split"]
    canon = sorted(held)
    if len(canon) != 141:
        raise RuntimeError("canonical held-out population changed")
    fmap = R.fold_of_file()
    if args.limit:
        canon = canon[: args.limit]
    per = {r: {a: [] for a in ARMS} for r in RECIPES}
    for i, name in enumerate(canon, 1):
        gray = cv2.imread(str(rt["base"].CFG.train_dir / name), cv2.IMREAD_GRAYSCALE)
        gt_by_image = {}
        for image_id in sorted(by_file[name]):
            info = rt["images"][image_id]
            gt = [core.annotation_rle(info, e) for e in rt["annotations"].get(image_id, [])]
            gt_by_image[image_id] = [r for r in gt if mask_utils.area(r) > 0]
        for r in RECIPES:
            for arm, tot in file_counts(rt, r, fmap[name], name, gray, gt_by_image).items():
                per[r][arm].append(tot)
        if i % 10 == 0:
            print(f"{i}/{len(canon)} {name}", flush=True)

    arr = {r: {a: np.asarray(per[r][a]) for a in ARMS} for r in RECIPES}
    (ra, aa), (rb, ab) = PRIMARY
    primary = paired(arr[ra][aa], arr[rb][ab])
    rep = {
        "files": len(canon), "cell": {"t": T, "c": C}, "small_area": SMALL_AREA,
        "reads_test": False, "arms": {r: {a: summary(arr[r][a].sum(0)) for a in ARMS}
                                      for r in RECIPES},
        "primary": {"contrast": f"{rb}/{ab} - {ra}/{aa}", **primary,
                    "bar": {"delta": BAR_DELTA, "p_gt_0": BAR_P},
                    "passed": primary["delta"] >= BAR_DELTA and primary["p_gt_0"] >= BAR_P},
        "secondary_hybrid_vs_hybrid": paired(arr["baseline"]["hybrid"], arr["p2first"]["hybrid"]),
    }
    ref = SCRIPTS.parent / "outputs/v62_refiner_test.json"
    if not args.limit and ref.exists():
        old = json.loads(ref.read_text())["recipes"]
        rep["refined_reproduces_eval_v62_refiner"] = {
            r: abs(rep["arms"][r]["refined"]["pq"] - old[r]["v17_cell"]["pq"]) < 1e-9
            for r in RECIPES}
    args.out.write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
