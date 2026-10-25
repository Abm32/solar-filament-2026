"""V82: agreement stats of the fold-0 stage-B cohort (fit files only). CPU."""
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.argv = [sys.argv[0]]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import train_v82_refiner_targets as T  # noqa: E402

V73 = T.V73
V73.DEV = "cpu"
rt = V73.runtime()
fmap = V73.R.fold_of_file()
fit = sorted(n for n, k in fmap.items() if k != 0)
recs = T.fit_records(rt, V73.RECIPE, fit, fmap)
q = np.asarray([r["q"] for r in recs])
nrec = np.asarray([round(len(r["agree"]) / r["q"]) for r in recs])
out = dict(samples=len(recs), q_mean=float(q.mean()), frac_q_lt_1=float((q < 1).mean()),
           frac_single_record_image=float((nrec == 1).mean()),
           q_hist={str(v): int(c) for v, c in zip(*np.unique(np.round(q, 3), return_counts=True))})
print(json.dumps(out, indent=1))
(V73.ROOT / "outputs/v82/cohort_stats.json").write_text(json.dumps(out, indent=1) + "\n")
