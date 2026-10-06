"""Print the V62 5-fold pooled report as plain tables (no interpretation)."""
import json
import sys
from pathlib import Path

d = json.loads(Path(sys.argv[1]).read_text())
print(f"files {d['files']}  folds {d['folds']}  primary conf {d['primary_conf']}")
tp_a, tp_b = d["tp_fp_fn_a"], d["tp_fp_fn_b"]
print(f"A baseline : PQ {d['pq_a']:.4f}  TP/FP/FN {tp_a}  mean matched IoU {d['mean_iou_a']:.4f}")
print(f"B p2first  : PQ {d['pq_b']:.4f}  TP/FP/FN {tp_b}  mean matched IoU {d['mean_iou_b']:.4f}")
p, o = d["paired_primary"], d["paired_own_best"]
print(f"\nB - A at primary cut : {p['delta']:+.4f}  95% CI [{p['ci95'][0]:+.4f}, "
      f"{p['ci95'][1]:+.4f}]  P(>0) {p['p_gt_0']:.3f}")
print(f"B - A at B own best ({d['b_own_best_conf']}): {o['delta']:+.4f}  95% CI "
      f"[{o['ci95'][0]:+.4f}, {o['ci95'][1]:+.4f}]  P(>0) {o['p_gt_0']:.3f}")
nz = d["noise_aprime_minus_a"]
if nz:
    print(f"A' - A (seed noise)  : {nz['delta']:+.4f}  95% CI [{nz['ci95'][0]:+.4f}, "
          f"{nz['ci95'][1]:+.4f}]  P(>0) {nz['p_gt_0']:.3f}   -> n = {d['noise_scale_n']:.4f}, "
          f"2n = {2 * d['noise_scale_n']:.4f}")
print("\nper-fold B - A at primary cut:")
for k, v in d["per_fold_delta"].items():
    print(f"   fold {k}: {v:+.4f}")
print("\nrecall by GT category at primary cut (A -> B):")
for c in ("thin", "small", "faint", "normal"):
    a, b = d["recall_a"][c], d["recall_b"][c]
    print(f"   {c:7s} n={d['n_gt'][c]:5d}  {a:.3f} -> {b:.3f}  ({b - a:+.3f})")
sa, sb = d["structural_a"], d["structural_b"]
print(f"\nproposals/img @cut: A {sa['proposals_per_image']:.2f}  B {sb['proposals_per_image']:.2f}"
      f"  ratio {sb['proposals_per_image'] / sa['proposals_per_image']:.3f}")
print(f"FP/img @cut       : A {sa['fp_per_image']:.2f}  B {sb['fp_per_image']:.2f}"
      f"  ratio {sb['fp_per_image'] / sa['fp_per_image']:.3f}")
print("\nVERDICT (pre-registered rule, mechanical):", d["verdict"])
