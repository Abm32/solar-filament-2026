"""V81 patch: p2first_thin (labels shrunk 1 px) and p2first_union (one image per file, union labels).

Idempotent. Labels are built by scripts/v81_labels.py into <work>/yolo_fold<K>_<kind>, so the baseline
yolo_fold<K> dataset (shared by other recipes) is never touched.
  python scripts/patch_v81.py [target=scripts/train_v61_fold.py]
"""
import sys
from pathlib import Path

T = Path(sys.argv[1] if len(sys.argv) > 1 else "scripts/train_v61_fold.py")
assert (Path(__file__).resolve().parent / "v81_labels.py").exists(), "v81_labels.py missing"


def sub(t, old, new):
    assert t.count(old) == 1, (T, old[:60], t.count(old))
    return t.replace(old, new)


t = T.read_text()
if "p2first_thin" in t:
    print("already patched"); sys.exit(0)
t = sub(t, '''           "p2first_ns", "p2first_lt1", "p2first_lt2", "p2first_lt5")
V68_ARMS = ("p2first_m", "p2first_1280", "p2first_long", "p2first_flip", "p2first_rot",
            "p2first_ns", "p2first_lt1", "p2first_lt2", "p2first_lt5")  # all p2first graphs''',
'''           "p2first_ns", "p2first_lt1", "p2first_lt2", "p2first_lt5",
           "p2first_thin", "p2first_union")
V68_ARMS = ("p2first_m", "p2first_1280", "p2first_long", "p2first_flip", "p2first_rot",
            "p2first_ns", "p2first_lt1", "p2first_lt2", "p2first_lt5",
            "p2first_thin", "p2first_union")  # all p2first graphs''')
t = sub(t, '''LT_BUCKETS = {"p2first_lt1": ("m1", "p1"), "p2first_lt2": ("m2", "p2"), "p2first_lt5": ("m5", "p5")}
''', '''LT_BUCKETS = {"p2first_lt1": ("m1", "p1"), "p2first_lt2": ("m2", "p2"), "p2first_lt5": ("m5", "p5")}
V81_LABELS = {"p2first_thin": "thin", "p2first_union": "union"}  # V81: own dataset dir per arm
''')
t = sub(t, '''    ds = prepare_fold_dataset(args.fold, train_files, args.work)
''', '''    data_dir = args.work / f"yolo_fold{args.fold}"
    if args.recipe in V81_LABELS:
        import v81_labels
        data_dir = data_dir.with_name(f"{data_dir.name}_{V81_LABELS[args.recipe]}")
        ds = v81_labels.build(V81_LABELS[args.recipe], train_files, data_dir, ANNOTATIONS,
                              IMAGE_DIR, v21.polygon_line)
    else:
        ds = prepare_fold_dataset(args.fold, train_files, args.work)
''')
t = sub(t, '''    data_yaml = args.work / f"yolo_fold{args.fold}" / "data.yaml"''',
        '''    data_yaml = data_dir / "data.yaml"''')
T.write_text(t)
print("patched", T)
