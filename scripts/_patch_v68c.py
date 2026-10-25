from pathlib import Path
p = Path("scripts/train_v61_fold.py"); t = p.read_text()
old = '    if args.recipe.endswith("_cp"):\n        # V65 resume'
new = '    if args.recipe.endswith("_cp") or args.recipe == "p2first_1280":\n        # V68b: 1280 fold 2 stage 2 OOMed at batch 2 (Ultralytics fell back to batch 1).\n        # V65 resume'
assert t.count(old) == 1, t.count(old)
p.write_text(t.replace(old, new)); print("ok")
