"""Probe: under the real training config (AMP), what dtype reaches single_mask_loss, and does the
Focal+Dice loss ever go non-finite? Wraps the installed loss, trains a tiny slice, prints stats.

  CUDA_VISIBLE_DEVICES=3 python scripts/_probe_v61_amp.py --mask-ratio 2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask-ratio", type=int, default=2)
    ap.add_argument("--fraction", type=float, default=0.03)
    ap.add_argument("--data", default="/mnt/v61/yolo_fold0/data.yaml")
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO

    import v61_seg_loss as S

    stats = {"calls": 0, "pred_dtypes": {}, "proto_dtypes": {}, "nonfinite": 0,
             "max_n": 0, "max_box_px": 0.0, "max_sum_den": 0.0}
    orig_make = S.make_single_mask_loss

    def make(*a, **k):
        fn = orig_make(*a, **k)

        def wrapped(gt_mask, pred, proto, xyxy, area):
            stats["calls"] += 1
            stats["pred_dtypes"][str(pred.dtype)] = stats["pred_dtypes"].get(str(pred.dtype), 0) + 1
            stats["proto_dtypes"][str(proto.dtype)] = stats["proto_dtypes"].get(
                str(proto.dtype), 0) + 1
            out = fn(gt_mask, pred, proto, xyxy, area)
            if not torch.isfinite(out):
                stats["nonfinite"] += 1
            stats["max_n"] = max(stats["max_n"], int(pred.shape[0]))
            box_px = float(((xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])).max())
            stats["max_box_px"] = max(stats["max_box_px"], box_px)
            return out
        return wrapped

    S.make_single_mask_loss = make
    S.install(1.0, 1.0)
    model = YOLO("yolo11s-seg.pt")
    model.train(data=args.data, epochs=1, imgsz=1024, batch=4, device=0, workers=2,
                fraction=args.fraction, val=False, plots=False, amp=True, single_cls=True,
                mask_ratio=args.mask_ratio, overlap_mask=False, optimizer="AdamW", lr0=1e-4,
                project="/tmp/v61_amp_probe", name=f"mr{args.mask_ratio}", exist_ok=True,
                verbose=False, seed=1, deterministic=False)
    print("PROBE_STATS", stats)
    print("max_mem_gib", torch.cuda.max_memory_allocated() / 2**30)


if __name__ == "__main__":
    main()
