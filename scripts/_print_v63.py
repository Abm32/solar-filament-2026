"""Print the V63 compare.json (refiner re-fit arms) as a plain table."""
import json
import sys
from pathlib import Path

d = json.loads(Path(sys.argv[1]).read_text())
print("top-level keys:", list(d.keys()))
for k, v in d.items():
    if isinstance(v, dict) and "pq" in v:
        print(f"  {k:14s} PQ {v['pq']:.4f}  " + "  ".join(
            f"{a} {v[a]}" for a in ("tp", "fp", "fn") if a in v))
arms = d.get("arms") or d.get("recipes") or {}
for k, v in arms.items():
    if isinstance(v, dict):
        pq = v.get("pq", v.get("v17_cell", {}).get("pq"))
        extra = {a: v.get(a, v.get("v17_cell", {}).get(a)) for a in ("tp", "fp", "fn")}
        print(f"  arm {k:14s} PQ {pq if pq is None else round(pq, 4)}  {extra}")
for key in ("primary", "primary_result"):
    if key in d:
        print("PRIMARY:", json.dumps(d[key], indent=1)[:600])
for group in ("secondary", "comparisons", "paired"):
    if group in d and isinstance(d[group], dict):
        for name, c in d[group].items():
            if isinstance(c, dict) and "delta" in c:
                print(f"  {name:24s} {c.get('a', '')} -> {c.get('b', '')}  delta {c['delta']:+.4f}"
                      f"  CI [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]  P(>0) {c['p_gt_0']:.3f}")
print("primary_passed:", d.get("primary_passed"))
