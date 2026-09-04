from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import BinaryMaskDataset
from src.metrics import dice_score
from src.model import BCEDiceLoss, build_model


def run_epoch(model, loader, loss_fn, optimizer, device: torch.device, train: bool) -> tuple[float, float]:
    model.train(train)
    total_loss = 0.0
    total_dice = 0.0

    for images, masks in tqdm(loader, leave=False):
        images = images.to(device)
        masks = masks.to(device)

        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = loss_fn(logits, masks)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        probs = torch.sigmoid(logits).detach().cpu().numpy() > 0.5
        targets = masks.detach().cpu().numpy() > 0.5
        batch_dice = sum(dice_score(p[0], t[0]) for p, t in zip(probs, targets, strict=False))
        total_loss += float(loss.item()) * images.size(0)
        total_dice += batch_dice

    n = len(loader.dataset)
    return total_loss / n, total_dice / n


def train(config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text())
    torch.manual_seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config["architecture"], config["encoder"]).to(device)
    loss_fn = BCEDiceLoss(**config["loss"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    train_ds = BinaryMaskDataset(Path(config["data"]["train_images"]), Path(config["data"]["train_masks"]))
    valid_ds = BinaryMaskDataset(Path(config["data"]["valid_images"]), Path(config["data"]["valid_masks"]))
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_ds, batch_size=config["batch_size"], shuffle=False, num_workers=4)

    checkpoint_dir = Path(config["outputs"]["checkpoints"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_dice = -1.0

    for epoch in range(1, config["epochs"] + 1):
        train_loss, train_dice = run_epoch(model, train_loader, loss_fn, optimizer, device, train=True)
        valid_loss, valid_dice = run_epoch(model, valid_loader, loss_fn, optimizer, device, train=False)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} train_dice={train_dice:.4f} "
            f"valid_loss={valid_loss:.4f} valid_dice={valid_dice:.4f}"
        )

        if valid_dice > best_dice:
            best_dice = valid_dice
            torch.save(
                {"model": model.state_dict(), "config": config, "valid_dice": valid_dice},
                checkpoint_dir / "best.pt",
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/baseline_unetplusplus.yaml"))
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args().config)

