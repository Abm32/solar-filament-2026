"""Print a DenseCRF eval JSON as a table (V61)."""
import json
import sys
from pathlib import Path

d = json.loads(Path(sys.argv[1]).read_text())
b = d["baseline_v17_pq"]
print("baseline PQ %.4f tp%d fp%d fn%d" % (b["pq"], b["tp"], b["fp"], b["fn"]))
print("%-10s %7s %7s %5s %5s %5s | res_push/n  dIoU | good_dmg/n  dIoU | area b->a" %
      ("variant", "PQ", "delta", "tp", "fp", "fn"))
for k, r in d["variants"].items():
    print("%-10s %.4f %+.4f %5d %5d %5d | %3d/%d %+.3f | %3d/%d %+.3f | %.2f->%.2f" % (
        k, r["pq"], r["delta"], r["tp"], r["fp"], r["fn"], r["pushed_to_0.5"],
        r["reservoir_n"], r["median_iou_delta_reservoir"], r["good_damaged"], r["good_n"],
        r["median_iou_delta_good"], r["median_area_ratio_before"],
        r["median_area_ratio_after"]))
