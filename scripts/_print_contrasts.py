"""Print the `contrasts` block of a compare_v63.py JSON."""
import json
import sys
from pathlib import Path

d = json.loads(Path(sys.argv[1]).read_text())
for name, v in d["contrasts"].items():
    lo, hi = v["ci95"]
    print(f"  {name:26s} {v['a']:13s} -> {v['b']:13s} delta {v['delta']:+.4f}  "
          f"CI [{lo:+.4f}, {hi:+.4f}]  P(>0) {v['p_gt_0']:.3f}")
print("primary_passed:", d["primary_passed"], "| cell", d["cell"], "| files", d["files"])
