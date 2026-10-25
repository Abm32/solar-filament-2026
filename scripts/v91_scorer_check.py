"""V91: verify per-file host-exact PQ bookkeeping reproduces host V84 fold 0 (0.4589877735617702)."""
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from pycocotools import mask as mu

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v91_spine as V  # noqa: E402

V84 = V.v84_rles(0)


def one(f):
    G = pickle.load(open(V.DATA / "gt.pkl", "rb"))["gt"][V.NAMES.index(f)]
    p = V84.get(Path(f).stem, [])
    recs = []
    for r in G:
        M = mu.iou(r, p, [0] * len(p)) if len(r) and len(p) else np.zeros((0, 0))
        recs.append((len(r), [[(g, j, float(M[g, j])) for g, j in zip(*np.where(M > 0.5))]]))
    return V.stats(recs, [np.ones(len(p), bool)])


if __name__ == "__main__":
    with Pool(32) as pool:
        S = pool.map(one, V.folds()[0])
    print("v84 fold0", V.pq(S), "host 0.4589877735617702", np.sum(S, 0)[1:])
