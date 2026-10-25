"""V86 stage 2 (CPU): pooled multi-source candidates -> cross-fitted LightGBM selector -> greedy non-overlapping
selection by y-hat (pixel ownership, MIN200, 50% retain) -> Dinkelbach keep (y-hat > lambda/2).
Pre-registration: docs/v86_pooled_selector_preregistration_2026-10-12.md
  python scripts/v86_select.py [extract.pkl]   (writes outputs/v86/result.json)
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from multiprocessing import Pool, set_start_method
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import v83_rule as V  # noqa: E402  (pq, fit, boot, FEATS; lightgbm from outputs/v70/pylib)
import v84_bundle as B  # noqa: E402  (crossfit)
SOURCES = ["p2first", "p2first_1280", "baseline", "p2first_cp", "baseline_seed2"]  # = v86_extract.SOURCES

ROOT = SCRIPTS.parent
OUT = ROOT / "outputs/v86"
MINA, RETAIN = 200, 0.5
ARMS = {f"P{i}": SOURCES[:i] for i in range(1, 6)}
PRIMARY = "P3"
GATE = dict(delta=0.010, p=0.90)
XFEATS = (V.FEATS + [f"src_{s}" for s in SOURCES] + [f"{k}_{s}" for s in SOURCES for k in ("iou", "msc", "mpin")]
          + ["n_agree", "agree_mean_score", "agree_max_pin", "n_ovl01", "n_pool"])
RES = None


def pool_file(r, srcs):
    """Pooled candidates of one file: (source, index) list, feature rows, target, pairwise IoU."""
    from pycocotools import mask as mu
    cands = [(s, c) for s in srcs for c in range(len(r["src"][s]["rows"]))]
    rles = [r["src"][s]["rles"][c] for s, c in cands]
    n = len(cands)
    I = np.asarray(mu.iou(rles, rles, [0] * n)).reshape(n, n) if n else np.zeros((0, 0))
    R = len(r["n_gt"])
    X, y = [], []
    for i, (s, c) in enumerate(cands):
        row = r["src"][s]["rows"][c]
        f = [row[k] for k in V.FEATS] + [float(s == t) for t in SOURCES]
        agree_sc, agree_pin, slot = [], [], []
        for t in SOURCES:
            if t not in srcs:
                slot.append((-1.0, -1.0, -1.0))
                continue
            if t == s:
                m, sc, pin = 1.0, r["src"][s]["scores"][c], row["p_in"]
            else:
                js = [j for j, (u, _) in enumerate(cands) if u == t]
                if js:
                    b = js[int(np.argmax(I[i, js]))]
                    m, sc, pin = float(I[i, b]), r["src"][t]["scores"][cands[b][1]], r["src"][t]["rows"][cands[b][1]]["p_in"]
                else:
                    m, sc, pin = 0.0, 0.0, 0.0
            slot.append((m, sc, pin))
            if m > 0.5:
                agree_sc.append(sc)
                agree_pin.append(pin)
        f += [v[0] for v in slot] + [v[1] for v in slot] + [v[2] for v in slot]
        f += [len(agree_sc), float(np.mean(agree_sc)), float(max(agree_pin)), float((I[i] > 0.1).sum() - 1), float(n)]
        X.append(f)
        y.append(min(1.0, sum(float(m[:, c][m[:, c] > 0.5].sum()) for m in r["src"][s]["iou"] if m.size) / R))
    return cands, X, y


def select_job(job):
    """Greedy ownership selection for several prediction vectors of one file.
    Returns {key: (accepted preds in acceptance order, [IoU(GT record, accepted trimmed masks)])}."""
    from pycocotools import mask as mu
    fidx, cands, preds = job
    r = RES[fidx]
    masks = [mu.decode(r["src"][s]["rles"][c]).astype(bool) for s, c in cands]
    out = {}
    for key, p in preds.items():
        occ = np.zeros(masks[0].shape, bool) if masks else None
        acc_p, acc_r = [], []
        for i in np.argsort(-np.asarray(p), kind="stable"):
            m = masks[i]
            t = m & ~occ
            a = int(t.sum())
            if a < MINA or a < RETAIN * m.sum():
                continue
            occ |= t
            acc_p.append(float(p[i]))
            acc_r.append(mu.encode(np.asfortranarray(t.astype(np.uint8))))
        ious = [np.asarray(mu.iou(g, acc_r, [0] * len(acc_r))).reshape(len(g), len(acc_r)) if g and acc_r
                else np.zeros((len(g), len(acc_r))) for g in r["gts"]]
        out[key] = (np.asarray(acc_p), ious)
    return fidx, out


def counts(sel, thr):
    """Per-file (iou, tp, fp, fn) keeping the accepted prefix with pred > thr."""
    acc_p, ious = sel
    k = acc_p > thr
    c = np.zeros(4)
    for m in ious:
        sub = m[:, k]
        n, nk = m.shape[0], int(k.sum())
        if not n or not nk:
            c += (0.0, 0, nk, n)
            continue
        h = sub > 0.5
        c += (sub[h].sum(), h.sum(), (~h.any(0)).sum(), (~h.any(1)).sum())
    return c


def dinkelbach(sels, files):
    lam = V.pq(np.sum([counts(sels[f], -np.inf) for f in files], 0))
    for _ in range(50):
        new = V.pq(np.sum([counts(sels[f], lam / 2) for f in files], 0))
        if abs(new - lam) < 1e-6:
            break
        lam = new
    return lam


def main():
    global RES
    set_start_method("fork")
    RES = pickle.load(open(OUT / (sys.argv[1] if len(sys.argv) > 1 else "extract.pkl"), "rb"))
    full = len(RES) == 707
    ffold = np.asarray([r["fold"] for r in RES])
    ref = pickle.load(open(ROOT / "outputs/v84/export/parity/bundle_ref.pkl", "rb"))
    base = np.asarray([ref[r["name"]]["counts"] for r in RES])
    assert not full or abs(V.pq(base.sum(0)) - 0.45898) < 5e-5, V.pq(base.sum(0))
    out = dict(n_files=len(RES), v84=dict(pq=V.pq(base.sum(0))), arms={})
    file_c = {}
    with Pool(int(os.environ.get("V86_PROCS", "26"))) as pool:
        for arm, srcs in ARMS.items():
            pf = [pool_file(r, srcs) for r in RES]
            fi = np.concatenate([[f] * len(p[0]) for f, p in enumerate(pf)]).astype(int)
            A = dict(X=np.asarray([x for p in pf for x in p[1]], float), y=np.asarray([v for p in pf for v in p[2]]),
                     w=np.asarray([len(RES[f]["n_gt"]) for f in fi], float), fold=ffold[fi])
            yhat, inner = B.crossfit(A)
            jobs = []
            for f in range(len(RES)):
                sl = fi == f
                pr = {"outer": yhat[sl], "oracle": A["y"][sl]}
                pr.update({f"in{k}": inner[k][sl] for k in range(5) if ffold[f] != k})
                if arm == "P1":  # parity: detector-score order + cut 0.30 must equal V84 loo_demote
                    pr["score"] = np.asarray([RES[f]["src"][s]["scores"][c] for s, c in pf[f][0]])
                jobs.append((f, pf[f][0], pr))
            sels = dict(pool.map(select_job, jobs, chunksize=1))
            lams = [dinkelbach({f: sels[f][f"in{k}"] for f in range(len(RES)) if ffold[f] != k},
                               [f for f in range(len(RES)) if ffold[f] != k]) for k in range(5)]
            c = np.asarray([counts(sels[f]["outer"], lams[ffold[f]] / 2) for f in range(len(RES))])
            olam = dinkelbach({f: sels[f]["oracle"] for f in range(len(RES))}, list(range(len(RES))))
            oc = np.asarray([counts(sels[f]["oracle"], olam / 2) for f in range(len(RES))])
            file_c[arm] = c
            if arm == "P1":
                par = V.pq(np.sum([counts(sels[f]["score"], 0.30 - 1e-9) for f in range(len(RES))], 0))
                want = json.loads((ROOT / "outputs/v84/result.json").read_text())["arms"]["loo_demote"]["pq"]
                out["parity_P1_score_cut030"] = dict(pq=par, v84_loo_demote=want)
                print("parity", par, want, flush=True)
                assert not full or abs(par - want) < 1e-9, (par, want)
            t = c.sum(0)
            a = dict(sources=srcs, n_cand=int(len(A["y"])),
                     n_cand_by_source={s: int(sum(len(r["src"][s]["rows"]) for r in RES)) for s in srcs},
                     pq=V.pq(t), sq=float(t[0] / max(t[1], 1)), tp=int(t[1]), fp=int(t[2]), fn=int(t[3]),
                     lambdas=lams, boot=V.boot(base, c, seed=86),
                     fold_delta=[V.pq(c[ffold == k].sum(0)) - V.pq(base[ffold == k].sum(0)) for k in range(5)],
                     oracle=dict(pq=V.pq(oc.sum(0)), lam=olam, delta_vs_v84=V.pq(oc.sum(0)) - V.pq(base.sum(0))))
            if arm == PRIMARY:
                m = V.fit(A["X"], A["y"], A["w"]).feature_importance("gain")
                a["importance_top15"] = dict(sorted(zip(XFEATS, (m / m.sum()).round(4).tolist()),
                                                    key=lambda kv: -kv[1])[:15])
            out["arms"][arm] = a
            print(arm, round(a["pq"], 5), json.dumps(a["boot"]), np.round(a["fold_delta"], 4).tolist(),
                  "oracle", round(a["oracle"]["pq"], 5), "cands", a["n_cand"], flush=True)
    out["P3_vs_P1"] = V.boot(file_c["P1"], file_c["P3"], seed=86)
    p = out["arms"][PRIMARY]["boot"]
    out["verdict"] = "CONTINUE" if p["delta"] >= GATE["delta"] and p["p_gt_0"] >= GATE["p"] else "KILL"
    (OUT / ("result.json" if full else "result_smoke.json")).write_text(json.dumps(out, indent=1))
    print("verdict", out["verdict"], "P3_vs_P1", json.dumps(out["P3_vs_P1"]))


if __name__ == "__main__":
    main()
