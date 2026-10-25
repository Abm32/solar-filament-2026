from pathlib import Path
def sub(path, old, new):
    p = Path(path); t = p.read_text()
    assert t.count(old) == 1, (path, old[:60], t.count(old))
    p.write_text(t.replace(old, new))
E = "scripts/eval_v61_oof.py"
sub(E, "IMAGE_DIR / name, iou=0.3, imgsz=args.imgsz)", "(args.image_dir or IMAGE_DIR) / name, iou=0.3, imgsz=args.imgsz)")
sub(E, '''    i.add_argument("--imgsz", type=int, default=1024)''', '''    i.add_argument("--imgsz", type=int, default=1024)
    i.add_argument("--image-dir", type=Path, default=None, dest="image_dir",
                   help="V69: read same-named images from here (neighbour frames)")''')
print("ok")
