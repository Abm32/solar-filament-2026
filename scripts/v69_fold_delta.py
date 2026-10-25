import json, sys
import numpy as np
rows = [json.loads(l) for l in open(sys.argv[1])]
def counts(r, v, c, amin=0):
    s = tp = nk = 0
    for p in r["preds"][v]:
        if p[0] >= c and p[1] > 0 and p[1] >= amin:
            s += p[2]; tp += p[3]; nk += 1
    return np.array([s, tp, r["R"] * nk - tp, r["n_gt"] - tp])
pq = lambda t: t[0] / (t[1] + 0.5 * (t[2] + t[3]))
fold = np.array([r["fold"] for r in rows])
for v, amin in (("erode1", 0), ("base", 200), ("erode1", 200), ("open_cc", 0)):
    A = np.array([counts(r, "base", 0.35) for r in rows]); B = np.array([counts(r, v, 0.35, amin) for r in rows])
    rng = np.random.default_rng(0); d = []
    for _ in range(2000):
        ix = rng.integers(0, len(rows), len(rows)); d.append(pq(B[ix].sum(0)) - pq(A[ix].sum(0)))
    print(v, amin, "pooled", round(pq(B.sum(0)) - pq(A.sum(0)), 4), "CI", np.round(np.percentile(d, [2.5, 97.5]), 4),
          "folds", [round(pq(B[fold == k].sum(0)) - pq(A[fold == k].sum(0)), 4) for k in range(5)],
          "TP/FP/FN", B.sum(0)[1:].astype(int), "meanIoU", round(B.sum(0)[0] / B.sum(0)[1], 4))
A = np.array([counts(r, "base", 0.35) for r in rows]); print("base TP/FP/FN", A.sum(0)[1:].astype(int), "meanIoU", round(A.sum(0)[0] / A.sum(0)[1], 4))
