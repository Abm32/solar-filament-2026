from pathlib import Path
def sub(path, old, new):
    p = Path(path); t = p.read_text()
    assert t.count(old) == 1, (path, old[:60], t.count(old))
    p.write_text(t.replace(old, new))
E = "scripts/eval_v61_oof.py"
sub(E, '''        if p.exists():
            continue
        write_views(''', '''        if p.exists():
            continue
        if args.image_dir and not (args.image_dir / name).exists():
            continue  # V69: no neighbour frame for this file in this bucket
        write_views(''')
C = "scripts/eval_v61_compare.py"
sub(C, '''    args = ap.parse_args()
    gt = load_gt(args.fold)
    names = sorted(gt)''', '''    ap.add_argument("--only", type=Path, default=None,
                    help="V69: restrict to the file names listed in this file")
    args = ap.parse_args()
    gt = load_gt(args.fold)
    names = sorted(gt)
    if args.only:
        keep = set(args.only.read_text().split())
        names = [n for n in names if n in keep]''')
print("ok")
