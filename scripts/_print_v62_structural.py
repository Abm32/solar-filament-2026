"""Print the V62 structural analysis JSON as plain tables (no interpretation)."""
import json
import sys
from pathlib import Path

d = json.loads(Path(sys.argv[1]).read_text())
cats = ("thin", "small", "faint", "normal")
print(f"fold {d['fold']}  files {d['files']}")
head = "recipe           stage     props/im  fp/im  recall_all  small+thin | "
print(head + "  ".join(f"{c:>7s}" for c in cats))
for r, stages in d["recipes"].items():
    for st, s in stages.items():
        rec = "  ".join(f"{s['recall'][c]:7.3f}" if s["recall"][c] is not None else "    n/a"
                        for c in cats)
        print(f"{r:16s} {st:9s} {s['proposals_per_image']:8.1f} {s['fp_per_image']:6.1f} "
              f"{s['recall_all']:10.3f} {s['recall_small_thin']:10.3f} | {rec}")
print("GT per category:", d["recipes"]["baseline"]["pool"]["n_gt"])
print()
print("seed-noise gain on pooled small+thin oracle recall (A' - A):",
      round(d["seed_noise_small_thin_pool_gain"], 4))
for r, g in d["gate"].items():
    print(f"\n{r}:")
    for k, v in g.items():
        print(f"   {k:28s} {round(v, 4) if isinstance(v, float) else v}")
