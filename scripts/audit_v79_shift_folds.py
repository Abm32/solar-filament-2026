"""V79 audit: does the 5-fold OOF mimic the train/test split? PQ by site/year/temporal gap
to the nearest training frame, from existing per-file OOF counts. CPU only, no models.
python scripts/audit_v79_shift_folds.py -> outputs/v79_audit/shift_folds.json"""
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data/raw/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"
OUT = ROOT / "outputs/v79_audit"
KEY = "t0.45_c0.30"


def ts(n):
    return datetime.strptime(n[:14], "%Y%m%d%H%M%S").timestamp() / 86400.0


def pq(c):
    iou, tp, fp, fn = np.asarray(c, float).sum(0)
    return float(iou / max(tp + 0.5 * fp + 0.5 * fn, 1e-9))


def boot_ci(rows, seed=79):
    rows = np.asarray(rows, float); rng = np.random.default_rng(seed)
    v = [pq(rows[rng.integers(0, len(rows), len(rows))]) for _ in range(1000)]
    return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]


def table(per, keyf):
    g = defaultdict(list)
    for n, c in per.items():
        g[keyf(n)].append(c)
    return {str(k): dict(n=len(v), pq=round(pq(v), 4), ci=boot_ci(v)) for k, v in sorted(g.items())}


def main():
    train = sorted(p.name for p in (D / "train/train_images").glob("*"))
    test = sorted(p.name for p in (D / "test/test_images").glob("*"))
    folds = json.loads((ROOT / "outputs/v61_folds.json").read_text())["folds"]
    fold_of = {n: int(f) for f, ns in folds.items() for n in ns}
    T = {n: ts(n) for n in train + test}
    tr_t = np.array([T[n] for n in train])

    def gap(n, pool):
        d = np.abs(np.array([T[m] for m in pool]) - T[n]); return float(d.min())

    # nearest labelled training frame available to the model at inference
    test_gap = [gap(n, train) for n in test]
    oof_gap = {n: gap(n, [m for m in train if fold_of[m] != fold_of[n]]) for n in train}
    pct = lambda v: [round(float(np.percentile(v, p)), 3) for p in (10, 25, 50, 75, 90)]
    res = dict(gap_days_pctl_10_25_50_75_90=dict(test_to_train=pct(test_gap), oof_to_otherfolds=pct(list(oof_gap.values()))),
               frac_gap_lt_1d=dict(test=float(np.mean(np.array(test_gap) < 1)),
                                   oof=float(np.mean(np.array(list(oof_gap.values())) < 1))),
               frac_gap_lt_3d=dict(test=float(np.mean(np.array(test_gap) < 3)),
                                   oof=float(np.mean(np.array(list(oof_gap.values())) < 3))))

    # full 707-file OOF (v73 base, 5 folds) and the refined 141-file pipeline (v63 r3)
    per707 = {}
    for f in range(5):
        pf = json.loads((ROOT / f"outputs/v73/base_fold{f}.json").read_text())["per_file"]
        per707.update({n: v["k7"][KEY] for n, v in pf.items()})
    per141 = {n: v[KEY] for n, v in json.loads((ROOT / "outputs/v63/r3.json").read_text())["per_file"].items()}
    bins = [0, 1, 3, 7, 30, 1e9]
    gbin = lambda n: next(f"{lo}-{hi}d" for lo, hi in zip(bins, bins[1:]) if oof_gap[n] < hi)
    for tag, per in (("v73_base_707", per707), ("v63_r3_141", per141)):
        res[tag] = dict(all=dict(n=len(per), pq=round(pq(list(per.values())), 4)),
                        by_site=table(per, lambda n: n[14]), by_year=table(per, lambda n: n[:4]),
                        by_oof_gap=table(per, gbin))
        # reweight the OOF by the test gap distribution (importance weights per gap bin)
        tb = np.histogram(test_gap, bins)[0] / len(test_gap)
        ob = np.histogram([oof_gap[n] for n in per], bins)[0] / len(per)
        w = {n: tb[np.digitize(oof_gap[n], bins) - 1] / max(ob[np.digitize(oof_gap[n], bins) - 1], 1e-9) for n in per}
        c = np.array([np.asarray(per[n], float) * w[n] for n in per])
        res[tag]["test_gap_reweighted_pq"] = round(pq(c), 4)
    res["note"] = "folds are seed-61 random per-file (group_days=0); test is a random file sample (site/year/image stats match)"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "shift_folds.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
