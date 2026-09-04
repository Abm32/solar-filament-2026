from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class BinaryMaskDataset(Dataset):
    def __init__(self, image_dir: Path, mask_dir: Path, transform=None) -> None:
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.image_paths = sorted(self.image_dir.glob("*"))
        if not self.image_paths:
            raise ValueError(f"No images found in {self.image_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[index]
        mask_path = self.mask_dir / f"{image_path.stem}.png"

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        if mask is None:
            raise ValueError(f"Could not read mask: {mask_path}")

        image = image.astype(np.float32) / 255.0
        mask = (mask > 0).astype(np.float32)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        image_tensor = torch.from_numpy(image[None, ...]).float()
        mask_tensor = torch.from_numpy(mask[None, ...]).float()
        return image_tensor, mask_tensor

