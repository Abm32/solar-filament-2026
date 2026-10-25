"""Analyse v69_metric_audit per-file output. Usage: python v69_audit_analyze.py <jsonl> <out.json>"""
from __future__ import annotations

import json
import sys

import numpy as np

rows = [json.loads(line) for line in open(sys.argv[1])]
out = {}


def pq(t):
    d = t[1] + 0.5 * (t[2] + t[3])
    return t[0] / d if d else 0.0


tk = np.sum([r["toolkit_035"] for r in rows], 0)
ho = np.sum([r["host_035"] for r in rows], 0)
dif = np.abs(np.array([r["toolkit_035"] for r in rows]) - np.array([r["host_035"] for r in rows]))
out["equivalence_035"] = {"toolkit": tk.tolist(), "host": ho.tolist(), "pq_toolkit": pq(tk),
                          "pq_host": pq(ho), "max_abs_iou_sum_diff_per_file": float(dif[:, 0].max()),
                          "max_count_diff_per_file": float(dif[:, 1:].max()),
                          "rle_roundtrip_all_ok": all(r["rle_roundtrip_ok"] for r in rows),
                          "R_eq_R_host": all(r["R"] == r["R_host"] for r in rows)}
N_GT = sum(r["n_gt"] for r in rows)
fold = np.array([r["fold"] for r in rows])


def table(variant):
    """Flat arrays per prediction: file idx, conf, area, R, s, nhit, radius."""
    f, c, a, R, s, h, rad = [], [], [], [], [], [], []
    for i, r in enumerate(rows):
        for j, p in enumerate(r["preds"][variant]):
            f.append(i); c.append(p[0]); a.append(p[1]); R.append(r["R"]); s.append(p[2])
            h.append(p[3]); rad.append(r["radius"][j])
    return {k: np.asarray(v, float) for k, v in
            dict(f=f, c=c, a=a, R=R, s=s, h=h, rad=rad).items()}


def pq_sel(t, keep, files=None):
    """Exact pooled PQ for a kept subset (1:1 matching; overlapping GT pairs are 5/8199)."""
    if files is None:
        ngt = N_GT
    else:
        ngt = sum(rows[i]["n_gt"] for i in files)
        keep = keep & np.isin(t["f"], list(files))
    tp = t["h"][keep].sum()
    fp = (t["R"][keep]).sum() - tp
    fn = ngt - tp
    return float(t["s"][keep].sum() / (tp + 0.5 * fp + 0.5 * fn))


CUTS = np.round(np.arange(0.20, 0.5501, 0.025), 3)
base = table("base")
out["base_check_035"] = pq_sel(base, base["c"] >= 0.35)
out["variants"] = {}
for v in rows[0]["preds"]:
    t = base if v == "base" else table(v)
    out["variants"][v] = {f"{c:.3f}": pq_sel(t, (t["c"] >= c) & (t["a"] > 0)) for c in CUTS}
out["min_area"] = {str(A): {f"{c:.3f}": pq_sel(base, (base["c"] >= c) & (base["a"] >= A))
                            for c in (0.30, 0.35, 0.40)} for A in (20, 50, 100, 150, 200, 300, 500)}
# Oracle decision layer (uses labels: upper bound only)
u = base["s"] / base["R"]
out["oracle_keep_u_gt_tau"] = {f"{tau:.2f}": pq_sel(base, u > tau) for tau in (0.15, 0.2, 0.25)}
# per-prediction u vs conf calibration
bins = np.arange(0.1, 1.01, 0.05)
cal = []
for lo, hi in zip(bins[:-1], bins[1:]):
    m = (base["c"] >= lo) & (base["c"] < hi)
    if m.sum():
        cal.append([round(lo, 2), int(m.sum()), float(u[m].mean()),
                    float((base["h"][m] / base["R"][m]).mean())])
out["calibration_conf_bin_n_meanu_matchrate"] = cal
# area bins at conf>=0.35
ab = []
for lo, hi in ((20, 100), (100, 200), (200, 400), (400, 800), (800, 1600), (1600, 1e9)):
    m = (base["c"] >= 0.35) & (base["a"] >= lo) & (base["a"] < hi)
    ab.append([lo, int(m.sum()), float(u[m].mean()) if m.sum() else None])
out["area_bins_conf035_n_meanu"] = ab

# Cross-fitted learned decision layer: u_hat(conf, log area, radius) -> keep u_hat > tau


class BinModel:
    """R-weighted mean u per (conf 0.05 bin x log-area quartile x limb) cell, backoff to conf bin."""

    def fit(self, X, y, sample_weight):
        w = sample_weight
        self.q = np.quantile(X[:, 1], [0.25, 0.5, 0.75])
        k1, k3 = self._keys(X)
        self.m1, self.m3 = {}, {}
        for keys, store, mn in ((k1, self.m1, 1), (k3, self.m3, 40)):
            for key in set(keys):
                sel = np.array([kk == key for kk in keys])
                if sel.sum() >= mn:
                    store[key] = float(np.average(y[sel], weights=w[sel]))
        return self

    def _keys(self, X):
        cb = np.floor(X[:, 0] / 0.05).astype(int)
        ab = np.searchsorted(self.q, X[:, 1])
        lb = (X[:, 2] > 0.85).astype(int)
        return list(cb), list(zip(cb, ab, lb))

    def predict(self, X):
        k1, k3 = self._keys(X)
        return np.array([self.m3.get(b, self.m1.get(a, 0.0)) for a, b in zip(k1, k3)])



X = np.c_[base["c"], np.log1p(base["a"]), base["rad"]]
pf = fold[base["f"].astype(int)]
uhat = np.zeros_like(u)
taus = np.round(np.arange(0.12, 0.30, 0.01), 2)
chosen_cut, chosen_tau = {}, {}
keep_fixed = np.zeros(len(u), bool)
keep_learn = np.zeros(len(u), bool)
for k in sorted(set(fold.tolist())):
    tr, te = pf != k, pf == k
    tr_files = [i for i in range(len(rows)) if fold[i] != k]
    # fixed conf cut chosen on training folds
    best_c = max(CUTS, key=lambda c: pq_sel(base, (base["c"] >= c) & tr, tr_files))
    keep_fixed |= te & (base["c"] >= best_c)
    chosen_cut[k] = float(best_c)
    # inner cross-fit for tau selection on train folds
    inner = np.zeros(len(u))
    for k2 in sorted(set(fold.tolist()) - {k}):
        a_, b_ = tr & (pf != k2), pf == k2
        m = BinModel()
        m.fit(X[a_], u[a_], sample_weight=base["R"][a_])
        inner[b_] = m.predict(X[b_])
    best_t = max(taus, key=lambda t: pq_sel(base, (inner > t) & tr, tr_files))
    m = BinModel()
    m.fit(X[tr], u[tr], sample_weight=base["R"][tr])
    uhat[te] = m.predict(X[te])
    keep_learn |= te & (uhat > best_t)
    chosen_tau[k] = float(best_t)
fixed_pq, learn_pq = pq_sel(base, keep_fixed), pq_sel(base, keep_learn)
# paired bootstrap over files
rng = np.random.default_rng(69)
per_file = []
for keep in (keep_fixed, keep_learn):
    arr = np.zeros((len(rows), 4))
    for i in range(len(rows)):
        m = keep & (base["f"] == i)
        tp = base["h"][m].sum()
        arr[i] = (base["s"][m].sum(), tp, base["R"][m].sum() - tp, rows[i]["n_gt"] - tp)
    per_file.append(arr)
d = []
for _ in range(2000):
    idx = rng.integers(0, len(rows), len(rows))
    d.append(pq(per_file[1][idx].sum(0)) - pq(per_file[0][idx].sum(0)))
out["decision_layer_crossfit"] = {"fixed_cut_pq": fixed_pq, "learned_pq": learn_pq,
                                  "delta": learn_pq - fixed_pq,
                                  "delta_ci95": [float(np.percentile(d, 2.5)),
                                                 float(np.percentile(d, 97.5))],
                                  "p_gt0": float(np.mean(np.array(d) > 0)),
                                  "cuts": chosen_cut, "taus": chosen_tau,
                                  "n_kept_fixed": int(keep_fixed.sum()),
                                  "n_kept_learned": int(keep_learn.sum())}
json.dump(out, open(sys.argv[2], "w"), indent=1)
print(json.dumps(out, indent=1))
