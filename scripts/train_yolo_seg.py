from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/yolo_seg.yaml"))
    args = parser.parse_args()

    config = load_config(args.config)
    gpu_count = torch.cuda.device_count()
    if gpu_count > 1:
        device: int | str = ",".join(str(index) for index in range(gpu_count))
    elif gpu_count == 1:
        device = 0
    else:
        device = "cpu"

    model = YOLO(config["model"])
    model.train(
        data=str(Path(config["yolo_dir"]) / "data.yaml"),
        epochs=config["epochs"],
        imgsz=config["imgsz"],
        batch=config["batch_size"],
        device=device,
        patience=config["patience"],
        seed=config["seed"],
        single_cls=not config["use_categories"],
        mask_ratio=config["mask_ratio"],
        overlap_mask=False,
        hsv_h=0.0,
        hsv_s=0.0,
        fliplr=0.0,
        flipud=0.0,
        mosaic=0.3,
        degrees=10,
        scale=0.3,
        hsv_v=0.3,
        close_mosaic=5,
        project="outputs/runs_filament",
        name="yolo_seg",
    )

    best = Path(model.trainer.best)
    target = Path("models/yolo_best.pt")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(best.read_bytes())
    print(f"saved {target}")


if __name__ == "__main__":
    main()
