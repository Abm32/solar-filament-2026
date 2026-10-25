# V94 record-only: direct (unrefined) fold-0 PQ, p2first vs p2first_limb, conf 0.30, paired bootstrap.
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "scripts")
import eval_v61_oof as E
import v61_toolkit as K

doc = json.loads(E.ANNOTATIONS.read_text()); gts = E.gt_by_file(doc)
names = sorted(E.load_folds()[0])
def rows(recipe, conf=0.30):
    return np.asarray([E.file_counts(recipe, 0, n, gts[n])[conf] for n in names])
a, b = rows("p2first"), rows("p2first_limb")
rng = np.random.default_rng(94)
d = [K.pq_value(b[i].sum(0)) - K.pq_value(a[i].sum(0))
     for i in (rng.integers(0, len(a), len(a)) for _ in range(2000))]
rep = {"files": len(names), "conf": 0.30,
       "pq_p2first": K.pq_value(a.sum(0)), "pq_p2first_limb": K.pq_value(b.sum(0)),
       "delta": K.pq_value(b.sum(0)) - K.pq_value(a.sum(0)),
       "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
       "p_gt_0": float(np.mean(np.asarray(d) > 0)),
       "counts_p2first": a.sum(0).tolist(), "counts_limb": b.sum(0).tolist()}
Path("outputs/v94/direct_fold0.json").write_text(json.dumps(rep, indent=2) + "\n")
print(json.dumps(rep, indent=2))
