"""Print the V62 refiner-test JSON (baseline+V11 refiner vs p2first+V11 refiner), plain tables."""
import json
import sys
from pathlib import Path

d = json.loads(Path(sys.argv[1]).read_text())
print(f"files {d['files']}  ({d['note']})")
hdr = f"{'recipe':10s} {'cell':14s} {'PQ':>7s} {'TP':>5s} {'FP':>5s} {'FN':>5s}"
print(hdr)
for r, v in d["recipes"].items():
    for tag, cell in (("V17 cell", "v17_cell"), ("own best", "best")):
        x = v[cell]
        name = f"{tag} ({v['best_cell']})" if cell == "best" else "t0.45_c0.30"
        print(f"{r:10s} {name:14s} {x['pq']:7.4f} {x['tp']:5d} {x['fp']:5d} {x['fn']:5d}")
p = d.get("paired")
if p:
    for k, v in p.items():
        print(f"B - A {k:20s} {v['delta']:+.4f}  95% CI [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]"
              f"  P(>0) {v['p_gt_0']:.3f}")
