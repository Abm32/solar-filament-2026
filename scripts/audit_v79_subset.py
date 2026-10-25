"""V79: is the 141-file canonical set easier than the 707? Direct p2first PQ on both."""
import json, pickle
from pathlib import Path
import numpy as np
R = Path(__file__).resolve().parents[1]
c141 = set(pickle.load(open(R / "outputs/v71/cache.pkl", "rb")))
L = [json.loads(x) for x in (Path.home() / "v69_kernels/audit/p2first_perfile.jsonl").read_text().splitlines()]
pq = lambda c: c[0] / (c[1] + .5 * c[2] + .5 * c[3])
a = np.array([x["host_035"] for x in L]); m = np.array([x["file"] in c141 for x in L])
r = dict(n141=int(m.sum()), direct_all707=pq(a.sum(0)), direct_141=pq(a[m].sum(0)), direct_566_rest=pq(a[~m].sum(0)))
print(r); (R / "outputs/v79_audit/subset.json").write_text(json.dumps(r))
