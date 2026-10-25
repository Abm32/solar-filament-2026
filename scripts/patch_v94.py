"""V94: register the p2first_limb recipe (limb-divided detector input) in train_v61_fold.py and let
eval_v61_oof.py infer find <stem>.png under --image-dir. Idempotent; run on the VM copy.
  python scripts/patch_v94.py
"""
from pathlib import Path

S = Path(__file__).resolve().parent
EDITS = {
    S / "train_v61_fold.py": [
        ('"p2first_thin", "p2first_union")\nV68_ARMS',
         '"p2first_thin", "p2first_union", "p2first_limb")\nV68_ARMS'),
        ('"p2first_thin", "p2first_union")  # all p2first graphs',
         '"p2first_thin", "p2first_union", "p2first_limb")  # all p2first graphs\n'
         'LIMB_IMAGES = ROOT / "outputs/v94/images"  # V94: limb-divided train images, <stem>.png'),
        ("def prepare_fold_dataset(fold: int, train_files, work: Path) -> dict:\n"
         '    """YOLO dataset containing ONLY the training files\' records (symlinked images)."""\n'
         '    data = work / f"yolo_fold{fold}"\n',
         "def prepare_fold_dataset(fold: int, train_files, work: Path, image_dir: Path = IMAGE_DIR,\n"
         "                         suffix: str | None = None, data_dir: Path | None = None) -> dict:\n"
         '    """YOLO dataset containing ONLY the training files\' records (symlinked images)."""\n'
         '    data = data_dir or work / f"yolo_fold{fold}"\n'),
        ('        src = IMAGE_DIR / info["file_name"]\n'
         '        p = Path(info["file_name"])\n'
         '        target = f"{p.stem}_ann{image_id}{p.suffix}"\n',
         '        p = Path(info["file_name"])\n'
         '        src = image_dir / (p.stem + suffix if suffix else p.name)\n'
         '        if not src.exists():\n'
         '            raise FileNotFoundError(src)\n'
         '        target = f"{p.stem}_ann{image_id}{suffix or p.suffix}"\n'),
        ("    else:\n        ds = prepare_fold_dataset(args.fold, train_files, args.work)\n",
         '    elif args.recipe == "p2first_limb":\n'
         '        data_dir = data_dir.with_name(f"{data_dir.name}_limb")\n'
         '        ds = prepare_fold_dataset(args.fold, train_files, args.work, image_dir=LIMB_IMAGES,\n'
         '                                  suffix=".png", data_dir=data_dir)\n'
         "    else:\n        ds = prepare_fold_dataset(args.fold, train_files, args.work)\n"),
        ('    if args.recipe.endswith("_cp") or args.recipe == "p2first_1280":\n',
         '    if args.recipe.endswith("_cp") or args.recipe in ("p2first_1280", "p2first_limb"):\n'
         '        # V94: shares GPU 2 with another job (12.5 GB free < 13.9 GB p2first peak)\n'),
    ],
    S / "eval_v61_oof.py": [
        ("        if args.image_dir and not (args.image_dir / name).exists():\n"
         "            continue  # V69: no neighbour frame for this file in this bucket\n"
         "        write_views(p, K.generate_views(model, (args.image_dir or IMAGE_DIR) / name, iou=0.3, imgsz=args.imgsz))\n",
         "        src = (args.image_dir or IMAGE_DIR) / name\n"
         "        if args.image_dir and not src.exists():\n"
         "            src = src.with_suffix(\".png\")  # V94: transformed images are png\n"
         "            if not src.exists():\n"
         "                continue  # V69: no neighbour frame for this file in this bucket\n"
         "        write_views(p, K.generate_views(model, src, iou=0.3, imgsz=args.imgsz))\n"),
    ],
}

for path, edits in EDITS.items():
    text = path.read_text()
    for old, new in edits:
        if new in text:
            continue
        if text.count(old) != 1:
            raise SystemExit(f"{path.name}: expected exactly one match for {old[:60]!r}")
        text = text.replace(old, new)
    path.write_text(text)
    print("patched", path.name)
