"""V93: V91 spine-net instances as extra candidates through the V84 soft refiner + a LightGBM quality rule.
Pre-registration: docs/v93_v91_candidates_preregistration_2026-10-19.md
  CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/v93_v91_candidates.py extract [limit]   # CPU, all folds
  .venv/bin/python scripts/v93_v91_candidates.py score
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from multiprocessing import Pool, set_start_method
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import audit_v80_extract as E  # noqa: E402
import train_v73_refiner as V73  # noqa: E402
import v83_extract as X  # noqa: E402
import v83_rule as V  # noqa: E402
import v91_spine as S91  # noqa: E402

ROOT = SCRIPTS.parent
OUT = ROOT / "outputs/v93"
TF, TS, MINA, OVER, T = 0.4, 0.3, 200, 0.1, 0.50
F91 = ["p_in", "p_max", "p_seed", "margin", "fallback", "seed_area", "ratio_pre_seed", "retain", "iou_seed_post",
       "log_area", "elong", "width", "rdist", "contrast", "dark", "disk_mean", "disk_std", "lowfreq_cv", "area",
       "v91_mpf", "v91_msp", "v91_mxp", "v91_log_area", "v91_log_slen", "n_v91", "n_v84"]
GATE = dict(delta=0.005, p=0.95, folds=4)
PRED = {}  # fold -> memmap (lazy per process)


def pred_of(fold, k):
    if fold not in PRED:
        PRED[fold] = S91.mm(f"pred_f{fold}.u8", "r", len(S91.folds()[fold]), (2,))
    return PRED[fold][k]


def work(job):
    import cv2
    from pycocotools import mask as mu
    name, fold, k, v84 = job
    rt, core = E.RT, E.RT["core"]
    gray = cv2.imread(str(rt["base"].CFG.train_dir / name), cv2.IMREAD_GRAYSCALE)
    recs = []
    for image_id in sorted(rt["by_file"][name], key=str):
        info = rt["images"][image_id]
        rl = [core.annotation_rle(info, a) for a in rt["annotations"].get(image_id, [])]
        recs.append([r for r in rl if mu.area(r) > 0])
    U = np.any([mu.decode(r) for r in v84], 0).astype(bool) if v84 else np.zeros((S91.SIZE, S91.SIZE), bool)
    p = pred_of(fold, k)
    lab, f5 = S91.decode(p[0] / 255.0, p[1] / 255.0, TF, TS)
    n = len(f5)
    area = np.bincount(lab.ravel(), minlength=n + 1)[1:]
    inU = np.bincount(lab[U], minlength=n + 1)[1:]
    elig = [j for j in range(n) if area[j] >= MINA and inU[j] / max(area[j], 1) <= OVER]
    masks = [((lab == j + 1) & ~U).astype(np.uint8) for j in elig]
    rows, post = [], []
    if masks:
        disk = E.disk_fit(gray)
        scores = [float(f5[j, 0]) for j in elig]
        bounds, _, probs = core.crop_probabilities(rt["base"], E.MODELS[fold], gray, masks, True)
        X.T = T
        posts, feats = X.refine_feats(masks, scores, bounds, probs, gray, disk, occupied=U.copy())
        for i, j in enumerate(elig):
            if posts[i] is None or mu.area(posts[i]) < MINA:
                continue
            f = {key: v for key, v in feats[i].items() if key != "bbox"}
            f.update(area=float(mu.area(posts[i])), v91_mpf=float(f5[j, 0]), v91_msp=float(f5[j, 1]),
                     v91_mxp=float(f5[j, 2]), v91_log_area=float(f5[j, 3]), v91_log_slen=float(f5[j, 4]),
                     n_v91=float(len(elig)), n_v84=float(len(v84)))
            rows.append(f)
            post.append(posts[i])
    i84 = [E.iou_mat(g, v84) for g in recs]
    i91 = [E.iou_mat(g, post) for g in recs]
    return dict(name=name, fold=fold, n_gt=[len(g) for g in recs], i84=i84, i91=i91, rows=rows, post=post,
                n_dec=int(n), n_elig=len(elig))


def cmd_extract(lim):
    import torch
    set_start_method("fork")
    V73.DEV = "cpu"
    E.RT = V73.runtime()
    for f in range(5):
        m = V73.build_model(E.RT["base"], "imagenet", "resnet18")
        m.load_state_dict(torch.load(ROOT / f"outputs/v84/soft_fold{f}_refiner.pt", map_location="cpu",
                                     weights_only=True)["model"])
        E.MODELS[f] = m.eval()
    folds = S91.folds()
    jobs = []
    for f in range(5):
        v84 = S91.v84_rles(f)
        for k, name in enumerate(folds[f]):
            jobs.append((name, f, k, v84.get(Path(name).stem, [])))
    jobs.sort(key=lambda j: j[0])
    if lim:  # smoke: only folds whose V91 predictions exist yet
        jobs = [j for j in jobs if (S91.DATA / f"pred_f{j[1]}.u8").exists()]
        jobs = jobs[:: max(1, len(jobs) // lim)][:lim]
    with torch.no_grad(), Pool(22, initializer=E.init) as p:
        res = p.map(work, jobs, chunksize=1)
    OUT.mkdir(parents=True, exist_ok=True)
    pickle.dump(res, open(OUT / f"extract{'_smoke' if lim else ''}.pkl", "wb"))
    print("files", len(res), "decoded", sum(r["n_dec"] for r in res), "eligible", sum(r["n_elig"] for r in res),
          "refined", sum(len(r["rows"]) for r in res), flush=True)


def counts(res, keep91, fi):
    """Per-file host-exact (iou, tp, fp, fn): V84 kept columns + kept V91 columns."""
    out = np.zeros((len(res), 4))
    for f, r in enumerate(res):
        k = keep91[fi == f]
        for m84, m91, n in zip(r["i84"], r["i91"], r["n_gt"]):
            cols = [m for m in (m84, m91[:, k] if m91.size else np.zeros((n, int(k.sum())))) if m.size]
            sub = np.concatenate(cols, 1) if cols else np.zeros((n, 0))
            if not n or not sub.shape[1]:
                out[f] += (0.0, 0, sub.shape[1], n)
                continue
            h = sub > 0.5
            out[f] += (sub[h].sum(), h.sum(), (~h.any(0)).sum(), (~h.any(1)).sum())
    return out


def cmd_score():
    res = pickle.load(open(OUT / "extract.pkl", "rb"))
    assert len(res) == 707, len(res)
    Xs, y, w, fi, fold = [], [], [], [], []
    for f, r in enumerate(res):
        R = len(r["n_gt"])
        for c, row in enumerate(r["rows"]):
            Xs.append([row[k] for k in F91])
            y.append(min(1.0, sum(float(m[:, c][m[:, c] > 0.5].sum()) for m in r["i91"] if m.size) / R))
            w.append(R)
            fi.append(f)
            fold.append(r["fold"])
    Xs, y, w, fi, fold = (np.asarray(a) for a in (Xs, y, w, fi, fold))
    Xs, w = Xs.astype(float), w.astype(float)
    ffold = np.asarray([r["fold"] for r in res])
    none = np.zeros(len(y), bool)
    base = counts(res, none, fi)
    pq0 = V.pq(base.sum(0))
    assert abs(pq0 - 0.45898) < 5e-5, pq0
    yhat, inner = np.zeros(len(y)), {}
    for k in range(5):
        tr = fold != k
        yhat[fold == k] = V.fit(Xs[tr], y[tr], w[tr]).predict(Xs[fold == k])
        yin = np.zeros(len(y))
        for j in range(5):
            if j != k:
                t2 = tr & (fold != j)
                yin[fold == j] = V.fit(Xs[t2], y[t2], w[t2]).predict(Xs[fold == j])
        inner[k] = yin
    lam84 = [json.loads((ROOT / f"outputs/v84/export/parity/demote_fold{k}.json").read_text())["lambda"]
             for k in range(5)]
    out = {"n_files": len(res), "v84_pq": pq0, "n_v91_rows": int(len(y)), "lambda_v84": lam84, "arms": {}}
    for arm in ("A", "B"):
        keep, lams = none.copy(), []
        for k in range(5):
            if arm == "A":
                lam = lam84[k]
            else:  # Dinkelbach on folds != k with inner predictions, combined set
                files = ffold != k
                lam = V.pq(counts(res, none, fi)[files].sum(0))
                for _ in range(50):
                    new = V.pq(counts(res, (inner[k] > lam / 2) & (fold != k), fi)[files].sum(0))
                    if abs(new - lam) < 1e-6:
                        break
                    lam = new
            lams.append(lam)
            te = fold == k
            keep[te] = yhat[te] > lam / 2
        c = counts(res, keep, fi)
        a = V.summ(c)
        a.update(lambdas=lams, added=int(keep.sum()), boot=V.boot(base, c, seed=93),
                 fold_delta=[V.pq(c[ffold == k].sum(0)) - V.pq(base[ffold == k].sum(0)) for k in range(5)])
        # orthogonality of the added masks
        tp_add = fp_add = 0
        for f, r in enumerate(res):
            k = keep[fi == f]
            for m in r["i91"]:
                if m.size and k.any():
                    h = m[:, k] > 0.5
                    tp_add += int(h.sum())
                    fp_add += int((~h.any(0)).sum())
        a.update(tp_rows_added=tp_add, fp_rows_added=fp_add)
        out["arms"][arm] = a
        print(arm, round(a["pq"], 5), json.dumps(a["boot"]), np.round(a["fold_delta"], 4).tolist(),
              "added", a["added"], "tp/fp rows", tp_add, fp_add, flush=True)
    a = out["arms"]["A"]
    npos = sum(d > 0 for d in a["fold_delta"])
    out["verdict"] = ("CONTINUE" if a["boot"]["delta"] >= GATE["delta"] and a["boot"]["p_gt_0"] >= GATE["p"]
                      and npos >= GATE["folds"] else "KILL")
    (OUT / "result.json").write_text(json.dumps(out, indent=1))
    print("verdict", out["verdict"])


if __name__ == "__main__":
    if sys.argv[1] == "extract":
        cmd_extract(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    else:
        cmd_score()
