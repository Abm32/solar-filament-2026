"""Measure peak CUDA memory of single_mask_loss (upstream vs Focal+Dice) at stage-2 geometry.

Stage 2 (mask_ratio=1) keeps masks at full proto resolution. Ultralytics 8.4.138 at imgsz 1024 +
mask_ratio 1 gives proto/gt masks of 256x256 -> n_instances x 256 x 256 floats per temporary.
We time/measure the loss + backward for n in {50, 100, 200, 300} instances per image.

  CUDA_VISIBLE_DEVICES=3 python scripts/_probe_v61_loss_mem.py
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))


def peak_gib(fn):
    import torch

    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2**30


def main():
    import torch
    import torch.nn.functional as functional
    from ultralytics.utils import loss as U
    from ultralytics.utils.ops import crop_mask

    import v61_seg_loss as S

    upstream = U.v8SegmentationLoss.__dict__["single_mask_loss"].__func__
    dev = torch.device("cuda:0")
    for hw in (1024,):
        print(f"--- proto {hw}x{hw} ---")
        for n in (100, 200):
            g = torch.Generator(device=dev).manual_seed(0)
            proto = torch.randn(32, hw, hw, device=dev, generator=g)
            gt = (torch.rand(n, hw, hw, device=dev, generator=g) > 0.9).float()
            xyxy = torch.tensor([[hw * 0.1, hw * 0.1, hw * 0.9, hw * 0.9]], device=dev).repeat(n, 1)
            area = torch.full((n,), 0.64, device=dev)

            def run(fn):
                pred = torch.randn(n, 32, device=dev, generator=g).half().requires_grad_(True)
                with torch.autocast("cuda", dtype=torch.float16):
                    out = fn(gt.clone(), pred, proto, xyxy.clone(), area)
                out.backward()

            row = {}
            for tag, fn in (("upstream", upstream),
                            ("focal+dice", S.make_single_mask_loss(
                                torch, functional, crop_mask, upstream, 1.0, 1.0)),
                            ("fd chunk16", S.make_single_mask_loss(
                                torch, functional, crop_mask, upstream, 1.0, 1.0, chunk=16))):
                try:
                    row[tag] = peak_gib(lambda f=fn: run(f))
                except torch.OutOfMemoryError:
                    row[tag] = float("nan")
                    torch.cuda.empty_cache()
            print(f"n={n:3d}  upstream {row['upstream']:.2f} GiB   focal+dice {row['focal+dice']:.2f}"
                  f" GiB   chunk16 {row['fd chunk16']:.2f} GiB"
                  f"   ratio(chunk16/upstream) {row['fd chunk16'] / row['upstream']:.2f}x")


if __name__ == "__main__":
    main()
