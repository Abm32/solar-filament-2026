"""V99 arm `d4`: 8-view detector TTA (the dihedral group: 4 rotations x {identity, left-right flip}) for the
existing p2first fold-0 detector, written as a V61 view cache under outputs/v61_oof_cache/p2first_d4/fold0.

Pre-registration: docs/v99_arms_preregistration_2026-10-25.md. Train images only (fold-0 held-out files).
Views 0..7 = (rot k, flip) for k in 0..3, flip in (no, yes); (0,no) (0,yes) (2,no) (2,yes) are exactly the four
V61 flip views (identity, LR, UD+LR, UD). cluster_views normalises by len(views), so consensus scores are on a
/8 scale; the gate sweeps the cut.
  CUDA_VISIBLE_DEVICES=3 python scripts/v99_d4_views.py [--weights outputs/v61_persistent/p2first_fold0/final.pt]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_v61_oof as O  # noqa: E402
import v61_toolkit as K  # noqa: E402

RECIPE = "p2first_d4"
D4 = [(k, flip) for k in range(4) for flip in (False, True)]


def fwd(a, k, flip):
    x = np.rot90(a, k)
    return np.ascontiguousarray(np.fliplr(x) if flip else x)


def inv(a, k, flip):
    x = np.fliplr(a) if flip else a
    return np.ascontiguousarray(np.rot90(x, -k))


def d4_views(model, image_path, iou=0.3, imgsz=1024):
    import cv2
    from pycocotools import mask as mu

    image = cv2.imread(str(image_path))
    if image is None or image.shape[:2] != (K.FULL, K.FULL):
        raise ValueError(f"bad image {image_path}")
    views = []
    for k, flip in D4:
        result = model.predict(fwd(image, k, flip), imgsz=imgsz, conf=K.CONF_FLOOR, agnostic_nms=True,
                               retina_masks=True, iou=iou, verbose=False)[0]
        rles, scores = [], []
        if result.masks is not None:
            scores = result.boxes.conf.detach().cpu().numpy().tolist()
            for m in result.masks.data:
                restored = inv(m.cpu().numpy().astype(np.uint8), k, flip)
                if restored.shape != image.shape[:2]:
                    raise ValueError("masks must be in original image coordinates")
                rles.append(K.encode(restored))
        keep = [i for i, r in enumerate(rles) if mu.area(r) > 0]
        views.append(([rles[i] for i in keep], [scores[i] for i in keep]))
    return views


def demo():
    a = np.arange(16, dtype=np.uint8).reshape(4, 4)
    for k, flip in D4:
        assert (inv(fwd(a, k, flip), k, flip) == a).all()
    print("demo ok: 8 transforms invert")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=O.ROOT / "outputs/v61_persistent/p2first_fold0/final.pt")
    ap.add_argument("--fold", type=int, default=0)
    args = ap.parse_args()
    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    held = O.load_folds()[args.fold]
    for i, name in enumerate(held, 1):
        p = O.cache_path(RECIPE, args.fold, name)
        if p.exists():
            continue
        O.write_views(p, d4_views(model, O.IMAGE_DIR / name))
        if i % 20 == 0:
            print(f"fold{args.fold} {i}/{len(held)}", flush=True)
    print("done", len(held))


if __name__ == "__main__":
    if sys.argv[1:] == ["demo"]:
        demo()
    else:
        main()
