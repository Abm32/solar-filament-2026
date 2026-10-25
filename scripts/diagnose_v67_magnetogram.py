"""V67 diagnostic (docs/v67_preregistration_2026-10-07.md): positive control, D1, D2.

Does a co-temporal magnetogram carry information on detector errors beyond score+geometry?
Runs on the VM (CPU). Needs data/external/gong_bz/registered512.npz (built from TRAIN files).

  python scripts/diagnose_v67_magnetogram.py --recipe p2first --out outputs/v67_diag.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import eval_v61_compare as C  # noqa: E402
import eval_v61_oof as O  # noqa: E402
import v61_toolkit as K  # noqa: E402
import v67_features as F  # noqa: E402
from pycocotools import mask as mask_utils  # noqa: E402

SMALL = 512
SCORE_MIN = 0.20
DETECT_CONF = 0.35
N_TRANSLATES = 20
ROOT = SCRIPTS.parent
REG = ROOT / "data/external/gong_bz/registered512.npz"


def to512(rle) -> np.ndarray:
    m = mask_utils.decode(rle).astype(np.float32)
    return cv2.resize(m, (SMALL, SMALL), interpolation=cv2.INTER_AREA) > 0.5


def translated_dpil(mask, dpil, disk_r, rng):
    """Mean d_PIL of `mask` translated by random on-disk vectors (30..120 px)."""
    ys, xs = np.nonzero(mask)
    out = []
    for _ in range(200):
        ang, mag = rng.uniform(0, 2 * np.pi), rng.uniform(30, 120)
        yy, xx = ys + int(mag * np.sin(ang)), xs + int(mag * np.cos(ang))
        ok = ((xx - SMALL / 2) ** 2 + (yy - SMALL / 2) ** 2 <= (0.97 * disk_r) ** 2).all()
        if ok:
            out.append(float(dpil[yy, xx].mean()))
        if len(out) == N_TRANSLATES:
            break
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recipe", default="p2first")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    reg = np.load(REG)
    disk = reg["disk"]
    med_r = float(np.median(disk[:, 2]))
    good = ((np.abs(disk[:, 2] - med_r) <= 6) & (np.abs(disk[:, 0] - 1023.5) <= 6)
            & (np.abs(disk[:, 1] - 1024.3) <= 6))
    print(f"frames: {len(good)}  geometry-ok: {int(good.sum())}  excluded: {int((~good).sum())}",
          flush=True)
    index = {str(n): i for i, n in enumerate(reg["names"]) if good[i]}
    folds = O.load_folds()
    rng = np.random.default_rng(67)

    prop_rows, gt_rows, control = [], [], []
    cache = {}
    for fold in sorted(folds):
        gt = C.load_gt(fold)
        for name in sorted(gt):
            if name not in index:
                continue
            i = index[name]
            bz = F.smooth_bz(reg["bz"][i].astype(np.float32))
            dpil = F.pil_distance(bz)
            disk_r = float(reg["disk"][i][2]) / 4.0
            cache[name] = (bz, dpil)
            views = K.read_cache_views(O.cache_path(args.recipe, fold, name))
            rles, scores = K.own_pixels(*K.cluster_views(views))
            keep = [j for j, s in enumerate(scores) if s >= SCORE_MIN]
            rles = [rles[j] for j in keep]
            scores = np.asarray([scores[j] for j in keep])
            gts_all = [r for rec in gt[name][0] for r in rec]
            iou = (mask_utils.iou(rles, gts_all, [0] * len(gts_all)) if rles and gts_all
                   else np.zeros((len(rles), len(gts_all))))
            for j, rle in enumerate(rles):
                m = to512(rle)
                if m.sum() < 4:
                    continue
                la, el = F.geometry(m)
                tp = bool(iou.shape[1] and iou[j].max() > 0.5)
                prop_rows.append((fold, name, tp, scores[j], la, el, *F.mask_features(bz, dpil, m)))
            # D2: per GT instance per record
            off = 0
            for rec in gt[name][0]:
                for g in rec:
                    m = to512(g)
                    if m.sum() < 4:
                        off += 1
                        continue
                    hit = bool(rles and (scores >= DETECT_CONF).any() and
                               iou[(scores >= DETECT_CONF), off].max() > 0.5)
                    la, el = F.geometry(m)
                    gt_rows.append((fold, name, hit, la, el, *F.mask_features(bz, dpil, m)))
                    t = translated_dpil(m, dpil, disk_r, rng)
                    if len(t) >= 5:
                        real = float(dpil[m].mean())
                        control.append(float(np.mean([(x > real) + 0.5 * (x == real) for x in t])))
                    off += 1
    print(f"files with magnetogram: {len(cache)}  proposals {len(prop_rows)}  gt {len(gt_rows)}",
          flush=True)

    def table(rows):
        arr = np.array(rows, dtype=object)
        return (arr[:, 0].astype(int), arr[:, 1], arr[:, 2].astype(bool),
                arr[:, 3:].astype(np.float64))

    report = {"files": len(cache), "recipe": args.recipe,
              "positive_control_auc": float(np.mean(control)) if control else None,
              "n_control": len(control)}
    ok_control = bool(control) and np.mean(control) >= 0.60
    report["positive_control_pass"] = bool(ok_control)
    print(f"positive control AUC {report['positive_control_auc']:.3f} (n {len(control)}) "
          f"-> {'PASS' if ok_control else 'FAIL (diagnostic INVALID)'}", flush=True)

    results = {}
    # column layout after (fold, name, label): proposals = score, log area, elongation, features;
    # GT instances = log area, elongation, features. Baseline = the non-magnetogram columns.
    for tag, rows, base_cols in (("D1_tp_vs_fp", prop_rows, 3),
                                 ("D2_detected_vs_missed", gt_rows, 2)):
        fold, names, y, xa = table(rows)
        xb = xa[:, :base_cols]
        sb = F.crossfit_scores(xb, y, fold)
        sa = F.crossfit_scores(xa, y, fold)
        d, lo, hi, p = F.delta_auc_bootstrap(y, sb, sa, names)
        per_fold = {int(k): F.auc(y[fold == k], sa[fold == k]) - F.auc(y[fold == k], sb[fold == k])
                    for k in np.unique(fold)}
        single = {}
        for j, feat in enumerate(F.FEATURES):
            single[feat] = F.auc(y, xa[:, base_cols + j])
        band = (xa[:, 0] >= 0.20) & (xa[:, 0] < 0.45) if tag.startswith("D1") else None
        results[tag] = {
            "n": int(len(y)), "pos_rate": float(y.mean()), "auc_base": F.auc(y, sb),
            "auc_aug": F.auc(y, sa), "delta": d, "lo": lo, "hi": hi, "p_gt0": p,
            "per_fold_delta": per_fold, "folds_pos": int(sum(v > 0 for v in per_fold.values())),
            "single_feature_auc": single}
        if band is not None and band.sum() > 50:
            results[tag]["band_0.20_0.45"] = {
                "n": int(band.sum()), "auc_base": F.auc(y[band], sb[band]),
                "auc_aug": F.auc(y[band], sa[band])}
        print(f"{tag}: n {len(y)} pos {y.mean():.2f}  AUC base {F.auc(y, sb):.3f} aug "
              f"{F.auc(y, sa):.3f}  d {d:+.4f} CI [{lo:+.4f},{hi:+.4f}] P {p:.3f} "
              f"folds+ {results[tag]['folds_pos']}/5", flush=True)
    report["results"] = results
    report["verdict"] = F.verdict(results["D1_tp_vs_fp"], results["D2_detected_vs_missed"]) \
        if ok_control else "invalid_positive_control_failed"
    print("VERDICT (pre-registered, mechanical):", report["verdict"])
    args.out.write_text(json.dumps(report, indent=2, default=float) + "\n")


if __name__ == "__main__":
    main()
