"""Smoke: build remapped V62 init checkpoints and verify they LOAD, TRAIN one tiny epoch (with the
validator patch), and PREDICT with masks in original-image coordinates.

  CUDA_VISIBLE_DEVICES=0 python scripts/_smoke_v62.py --variant p2first
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=("p2first", "p2last"), required=True)
    ap.add_argument("--data", default="/mnt/v61/yolo_fold0/data.yaml")
    args = ap.parse_args()

    import torch
    import v62_init
    import v62_patch
    from ultralytics import YOLO

    out = Path(f"outputs/v62_init/{args.variant}_init.pt")
    rep = v62_init.build(args.variant, out)
    print("BUILD copied", rep["copied"], "/", rep["total_new_tensors"],
          "| shape_mismatch", len(rep["shape_mismatch"]), "| coco_unmapped",
          len(rep["coco_unmapped"]))
    v62_patch.install(2 if args.variant == "p2first" else 4)

    ref = YOLO("yolo11s-seg.pt").model.state_dict()
    src = "model.23.cv2.1.0.conv.weight" if args.variant == "p2first" \
        else "model.23.cv2.2.0.conv.weight"
    model = YOLO(str(out))
    equal = torch.equal(model.model.state_dict()["model.26.cv2.2.0.conv.weight"], ref[src])
    print("reload: shared-scale head tensor equals COCO:", bool(equal))

    model.train(data=args.data, epochs=1, imgsz=1024, batch=2, device=0, workers=2,
                fraction=0.01, val=True, plots=False, amp=True, single_cls=True, mask_ratio=2,
                overlap_mask=False, optimizer="AdamW", lr0=1e-4, project="/tmp/v62_smoke",
                name=args.variant, exist_ok=True, verbose=False, seed=1, deterministic=False)
    print("TRAINED one tiny epoch; peak mem GiB",
          round(torch.cuda.max_memory_allocated() / 2**30, 2))
    last = Path(model.trainer.last)
    img_dir = Path("/mnt/v61/yolo_fold0/images/train")
    first = sorted(p.name for p in img_dir.iterdir())[0]
    res = YOLO(str(last)).predict(str(img_dir / first), imgsz=1024, conf=0.01, iou=0.3,
                                  retina_masks=True, verbose=False)[0]
    shape = None if res.masks is None else tuple(res.masks.data.shape)
    print("PREDICT ok; retina masks shape:", shape, "| orig image:", res.orig_shape)


if __name__ == "__main__":
    main()
