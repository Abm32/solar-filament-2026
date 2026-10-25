"""V69 misc: FN category x annotator agreement; confidence-grid marginals."""
import json
import pickle
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "outputs/v69_analysis/p2first"
gt = pickle.load(open(D / "gt.pkl", "rb"))
fn = gt[~gt.matched].copy()
cats = pd.read_csv(D / "fn_rows.csv")
fn["cat_err"] = cats.cat_err.values
m = fn[fn.n_rec >= 2]
print(pd.crosstab(m.cat_err, m.n_other_agree.clip(upper=1), margins=True))
print(pd.crosstab(fn.cat_err, fn.n_rec, margins=True))
for p in sorted(ROOT.glob("outputs/v61_oof_p2first*.json")):
    g = json.loads(p.read_text()).get("grid", {})
    print(p.name)
    for c, v in g.items():
        print(c, round(v["pq"], 4), v["tp"], v["fp"], v["fn"])
