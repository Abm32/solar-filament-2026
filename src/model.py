from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
from torch import nn


def build_model(architecture: str = "unetplusplus", encoder: str = "efficientnet-b3") -> nn.Module:
    architecture = architecture.lower()
    common = {
        "encoder_name": encoder,
        "encoder_weights": "imagenet",
        "in_channels": 1,
        "classes": 1,
        "activation": None,
    }

    if architecture == "unetplusplus":
        return smp.UnetPlusPlus(**common)
    if architecture == "deeplabv3plus":
        return smp.DeepLabV3Plus(**common)
    if architecture == "unet":
        return smp.Unet(**common)
    raise ValueError(f"Unknown architecture: {architecture}")


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = smp.losses.DiceLoss(mode="binary", from_logits=True)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.bce_weight * self.bce(logits, target) + self.dice_weight * self.dice(
            logits, target
        )

