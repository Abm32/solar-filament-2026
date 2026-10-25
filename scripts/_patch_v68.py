"""V68 patch: four p2first scaling arms (m backbone, 1280 input, 2x schedule, flip aug)."""
from pathlib import Path

def sub(path, old, new):
    p = Path(path); t = p.read_text()
    assert t.count(old) == 1, (path, old[:60], t.count(old))
    p.write_text(t.replace(old, new))

T = "scripts/train_v61_fold.py"
sub(T, '''           "p2last_seed2", "p2first_seed2", "p2first_cp")''',
'''           "p2last_seed2", "p2first_seed2", "p2first_cp",
           "p2first_m", "p2first_1280", "p2first_long", "p2first_flip")
V68_ARMS = ("p2first_m", "p2first_1280", "p2first_long", "p2first_flip")  # all p2first graphs''')
sub(T, '''        args = dict(args, copy_paste=COPY_PASTE)
    return args''',
'''        args = dict(args, copy_paste=COPY_PASTE)
    if recipe == "p2first_m":       # V68: yolo11m scale; batch 2 (T4 memory), nbs keeps eff. batch
        args = dict(args, batch=2)
    if recipe == "p2first_1280":    # V68: 1280 input -> 640x640 stride-2 prototypes
        args = dict(args, imgsz=1280, batch=2)
    if recipe == "p2first_long" and not smoke:  # V68: 2x schedule (160 + 40)
        args = dict(args, epochs=args["epochs"] * 2)
    if recipe == "p2first_flip":    # V68: flips at train time (inference already uses 4 flips)
        args = dict(args, fliplr=0.5, flipud=0.5)
    return args''')
sub(T, '''    arch = args.recipe.removesuffix("_seed2").removesuffix("_cp")
''', '''    arch = args.recipe.removesuffix("_seed2").removesuffix("_cp")
    init_name = arch
    if arch in V68_ARMS:
        arch, init_name = "p2first", ("p2first_m" if arch == "p2first_m" else "p2first")
''')
sub(T, '''        init_source = ROOT / "outputs/v62_init" / f"{arch}_init.pt"
        if not init_source.exists():
            v62_init.build(arch, init_source)''',
'''        init_source = ROOT / "outputs/v62_init" / f"{init_name}_init.pt"
        if not init_source.exists():
            if init_name == "p2first_m":
                v62_init.build(arch, init_source, coco="yolo11m-seg.pt",
                               cfg_name="yolo11m-p2first-seg.yaml")
            else:
                v62_init.build(arch, init_source)''')

I = "scripts/v62_init.py"
sub(I, '''def build(variant: str, out: Path, coco: str = "yolo11s-seg.pt") -> dict:''',
'''def build(variant: str, out: Path, coco: str = "yolo11s-seg.pt", cfg_name: str | None = None) -> dict:''')
sub(I, '''    cfg = v62_arch.CONFIGS / CFG[variant]''', '''    cfg = v62_arch.CONFIGS / (cfg_name or CFG[variant])''')

E = "scripts/eval_v61_oof.py"
sub(E, '''        write_views(p, K.generate_views(model, IMAGE_DIR / name, iou=0.3))''',
'''        write_views(p, K.generate_views(model, IMAGE_DIR / name, iou=0.3, imgsz=args.imgsz))''')
sub(E, '''    i.add_argument("--weights", type=Path, required=True)''',
'''    i.add_argument("--weights", type=Path, required=True)
    i.add_argument("--imgsz", type=int, default=1024)''')
print("patched")
