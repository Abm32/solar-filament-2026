"""Memory probe for the chunked-upstream loss in REAL stage-2 training (p2first + copy-paste).

Runs a short stage-2-style training on fold 1's training data from its finished base checkpoint
with the same arguments the resumed run uses (mask_ratio=1, copy_paste=0.3, batch 4), tracks the
peak CUDA memory, and reports it. Nothing is kept: output goes to /tmp.

  CUDA_VISIBLE_DEVICES=3 python scripts/_probe_v65_mem.py --fraction 0.15
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fraction", type=float, default=0.15)
    ap.add_argument("--fold", type=int, default=1)
    args = ap.parse_args()

    import torch
    import train_v61_fold as T
    import v61_seg_loss
    import v62_patch
    from ultralytics import YOLO

    v62_patch.install(2)
    print("loss patch:", v61_seg_loss.install_chunked_upstream())
    a = T.stage_args("p2first_cp", "mask_ratio1", False)
    a.update(epochs=1, fraction=args.fraction, close_mosaic=0, save_period=-1, val=False,
             plots=False)
    model = YOLO(f"/mnt/v61/p2first_cp_fold{args.fold}/base/final.pt")
    torch.cuda.reset_peak_memory_stats()
    model.train(data=f"/mnt/v61/yolo_fold{args.fold}/data.yaml", device=0,
                project="/tmp/v65_probe", name=f"f{args.fold}", exist_ok=True, verbose=False, **a)
    print("PEAK_GIB", round(torch.cuda.max_memory_allocated() / 2**30, 2))


if __name__ == "__main__":
    main()
