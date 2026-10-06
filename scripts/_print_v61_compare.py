"""Print the V61 fold comparison JSON as plain tables (no interpretation)."""
import json
import sys
from pathlib import Path

d = json.loads(Path(sys.argv[1]).read_text())
cats = ("thin", "small", "faint", "normal")
print(f"fold {d['fold']}  files {d['files']}  baseline best conf {d['baseline_best_conf']}")
print()
head = "recipe           cut               conf      PQ    TP    FP    FN    mIoU | recall: "
print(head + "  ".join(f"{c:>7s}" for c in cats))
for r, v in d["recipes"].items():
    for cut in ("own_best", "at_baseline_conf"):
        s = v[cut]
        rec = "  ".join(f"{s['recall'][c]:7.3f}" if s["recall"][c] is not None else "    n/a"
                        for c in cats)
        print(f"{r:16s} {cut:16s} {s['conf']:5.2f} {s['pq']:7.4f} {s['tp']:5d} {s['fp']:5d} "
              f"{s['fn']:5d} {s['mean_matched_iou']:7.4f} | {rec}")
print("GT per category:", d["recipes"]["baseline"]["own_best"]["n_gt"])
print()
rows = (("B - A  (own best cut)", d["paired_B_vs_A"]["own_best"]),
        ("B - A  (baseline cut)", d["paired_B_vs_A"]["at_baseline_conf"]),
        ("A' - A (seed noise)", d.get("paired_Aprime_vs_A")))
for k, v in rows:
    if v:
        print(f"{k:24s} delta {v['delta']:+0.4f}  95% CI [{v['ci95'][0]:+0.4f}, "
              f"{v['ci95'][1]:+0.4f}]  P(delta>0) {v['p_gt_0']:.3f}")
print()
print("noise scale n = |A'-A| =", round(d["noise_scale_n"], 4))
print("recall gain, small+thin+faint pooled (B-A):", round(d["recall_gain_fine_pooled"], 4))
print("recall gain, normal (B-A):", round(d["recall_gain_normal"], 4))
print("VERDICT (pre-registered rule, mechanical):", d["verdict"])
