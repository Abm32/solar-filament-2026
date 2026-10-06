"""Check: for every recipe, which training arguments differ from the baseline recipe?"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_v61_fold as T  # noqa: E402

ref = {st: T.stage_args("baseline", st, False) for st in ("base", "mask_ratio1")}
for r in T.RECIPES:
    for st in ("base", "mask_ratio1"):
        a = T.stage_args(r, st, False)
        diff = {k: (ref[st][k], a[k]) for k in a if a[k] != ref[st][k]}
        print(f"{r:15s} {st:11s} differs from baseline only in: {diff or 'nothing'}")
