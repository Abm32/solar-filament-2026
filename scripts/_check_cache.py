"""Verify each recipe's fold cache covers exactly the fold's held-out files (no missing/extra/tmp)."""
import json
import os
import sys

FOLD = sys.argv[1] if len(sys.argv) > 1 else "0"
RECIPES = sys.argv[2:] or ["baseline", "baseline_seed2"]
folds = json.load(open("outputs/v61_folds.json"))["folds"]
held = {os.path.splitext(n)[0] for n in folds[FOLD]}
for r in RECIPES:
    d = f"outputs/v61_oof_cache/{r}/fold{FOLD}"
    files = os.listdir(d)
    have = {f.replace(".json.gz", "") for f in files if f.endswith(".json.gz")}
    other = [f for f in files if not f.endswith(".json.gz")]
    print(f"{r:15s} held {len(held)} cached {len(have)} missing {len(held - have)} "
          f"extra {len(have - held)} non-gz {len(other)}")
