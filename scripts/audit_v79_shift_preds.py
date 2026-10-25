"""V79 audit: test-prediction stats (v65 r3all / v74 CSVs) vs OOF refined predictions
(outputs/v71/cache.pkl, same p2first+refiner pipeline, c0.30) and OOF GT. CPU only.
python scripts/audit_v79_shift_preds.py -> outputs/v79_audit/shift_preds.json"""
import csv, json, pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as mu

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/v79_audit"
TEST = sorted(p.stem for p in (ROOT / "data/raw/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test/test_images").glob("*"))


def rdist(m):
    ys, xs = np.nonzero(m)
    return float(np.hypot(ys.mean() - 1024, xs.mean() - 1024) / 935)  # ~median disk radius


def summ(per_img):  # name -> list of (area, rdist)
    n = np.array([len(v) for v in per_img.values()])
    a = np.array([x[0] for v in per_img.values() for x in v]); r = np.array([x[1] for v in per_img.values() for x in v])
    p = lambda v, q=(10, 25, 50, 75, 90): [round(float(np.percentile(v, x)), 3) for x in q]
    return dict(images=len(n), inst_total=int(n.sum()), inst_per_img_mean=round(float(n.mean()), 3),
                inst_per_img_pctl=p(n), empty_frac=round(float((n == 0).mean()), 3),
                area_pctl=p(a), frac_area_lt200=round(float((a < 200).mean()), 3),
                frac_area_gt5000=round(float((a > 5000).mean()), 3),
                rdist_pctl=p(r), frac_r_gt09=round(float((r > 0.9).mean()), 3))


def csv_stats(path):
    per = defaultdict(list)
    with open(path) as fh:
        for row in csv.DictReader(fh):
            img = row["filament_id"].rsplit("_", 1)[0]
            m = mu.decode({"size": [2048, 2048], "counts": row["segmentation_rle"].encode()})
            per[img].append((int(m.sum()), rdist(m)))
    return summ({n: per.get(n, []) for n in TEST})


def main():
    c = pickle.load(open(ROOT / "outputs/v71/cache.pkl", "rb"))
    oof_pred = {n: [(int(mu.area(p["rle"])), rdist(mu.decode(p["rle"]))) for p in d["preds"]] for n, d in c.items()}
    oof_gt = {n: [(int(mu.area(r)), rdist(mu.decode(r))) for r in d["gts"][0]] for n, d in c.items()}
    oof_scores = [p["score"] for d in c.values() for p in d["preds"]]
    res = dict(oof_pred_141=summ(oof_pred), oof_gt_141_first_record=summ(oof_gt),
               oof_score_pctl=[round(float(np.percentile(oof_scores, q)), 3) for q in (10, 50, 90)],
               test_v65=csv_stats(ROOT / "outputs/v65/export/submission_v65_p2first_r3all.csv"),
               test_v17=csv_stats(ROOT / "submission_ready/v17_2026-09-08/submission_v17.csv"),
               test_v74=csv_stats(ROOT / "outputs/v74/dryrun_fold0/submission_v74.csv"))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "shift_preds.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
