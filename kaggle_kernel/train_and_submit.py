from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"], check=True)

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pycocotools import mask as mask_utils
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18
from tqdm.auto import tqdm
from ultralytics import YOLO


class CFG:
    root_candidates = [
        Path("/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"),
        Path("/kaggle/input/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"),
    ]
    root = next((candidate for candidate in root_candidates if candidate.exists()), root_candidates[0])
    train_dir = root / "train" / "train_images"
    test_dir = root / "test" / "test_images"
    train_json = root / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    yolo_dir = Path("/kaggle/working/yolo_dataset")
    seed = 42
    imgsz = 1024
    batch = 4
    epochs = 30
    patience = 5
    model = "yolo11s-seg.pt"
    mask_ratio = 2
    crop_size = 256
    crop_context = 1.8
    min_crop = 96
    refiner_epochs = 8
    refiner_threshold = 0.65
    min_component_size = 20


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_coco():
    with CFG.train_json.open() as file:
        coco = json.load(file)
    images = {image["id"]: image for image in coco["images"]}
    annotations = {}
    for annotation in coco["annotations"]:
        annotations.setdefault(annotation["image_id"], []).append(annotation)
    return coco, images, annotations


def best_annotator_ids(images, annotations):
    file_to_ids = {}
    for image_id, info in images.items():
        file_to_ids.setdefault(info["file_name"], []).append(image_id)

    def completeness(image_id):
        anns = annotations.get(image_id, [])
        return len(anns), sum(ann.get("area", 0) for ann in anns)

    return {file_name: max(ids, key=completeness) for file_name, ids in file_to_ids.items()}


def polygon_to_yolo_line(polygon, width, height, class_id):
    points = np.array(polygon, dtype=np.float32).reshape(-1, 2)
    points[:, 0] = np.clip(points[:, 0] / width, 0, 1)
    points[:, 1] = np.clip(points[:, 1] / height, 0, 1)
    return f"{class_id} " + " ".join(f"{value:.6f}" for value in points.flatten())


def write_yolo_split(split_name, image_ids, images, annotations, cat_id_to_yolo):
    image_output = CFG.yolo_dir / "images" / split_name
    label_output = CFG.yolo_dir / "labels" / split_name
    image_output.mkdir(parents=True, exist_ok=True)
    label_output.mkdir(parents=True, exist_ok=True)

    for image_id in image_ids:
        info = images[image_id]
        source = CFG.train_dir / info["file_name"]
        target = image_output / info["file_name"]
        if not target.exists():
            os.symlink(source, target)

        lines = []
        for annotation in annotations.get(image_id, []):
            class_id = cat_id_to_yolo[annotation["category_id"]]
            for polygon in annotation["segmentation"]:
                if len(polygon) >= 6:
                    lines.append(
                        polygon_to_yolo_line(polygon, info["width"], info["height"], class_id)
                    )
        (label_output / f"{Path(info['file_name']).stem}.txt").write_text("\n".join(lines))


def prepare_yolo_data(coco, images, annotations):
    file_to_best_id = best_annotator_ids(images, annotations)
    files = sorted(file_to_best_id)
    random.Random(CFG.seed).shuffle(files)
    n_valid = round(len(files) * 0.2)
    valid_files = files[:n_valid]
    train_files = files[n_valid:]
    train_ids = [file_to_best_id[file_name] for file_name in train_files]
    valid_ids = [file_to_best_id[file_name] for file_name in valid_files]

    categories = sorted(coco["categories"], key=lambda category: category["id"])
    cat_id_to_yolo = {category["id"]: index for index, category in enumerate(categories)}

    write_yolo_split("train", train_ids, images, annotations, cat_id_to_yolo)
    write_yolo_split("val", valid_ids, images, annotations, cat_id_to_yolo)

    names = "\n".join(f"  {index}: {category['name']}" for index, category in enumerate(categories))
    (CFG.yolo_dir / "data.yaml").write_text(
        f"path: {CFG.yolo_dir}\ntrain: images/train\nval: images/val\nnames:\n{names}\n"
    )
    return train_ids, valid_ids


def square_bounds(mask, context=CFG.crop_context, minimum=CFG.min_crop):
    height, width = mask.shape
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0, 0, width, height
    cx = (xs.min() + xs.max() + 1) / 2
    cy = (ys.min() + ys.max() + 1) / 2
    side = int(np.ceil(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) * context))
    side = min(max(side, minimum), min(height, width))
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x1, y1 = x0 + side, y0 + side
    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x1 > width:
        x0 -= x1 - width
        x1 = width
    if y1 > height:
        y0 -= y1 - height
        y1 = height
    return int(x0), int(y0), int(x1), int(y1)


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class CropUNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        encoder = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.pool = encoder.maxpool
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4
        self.up4 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec4 = ConvBlock(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec3 = ConvBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec2 = ConvBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.dec1 = ConvBlock(96, 32)
        self.up0 = nn.ConvTranspose2d(32, 16, 2, 2)
        self.head = nn.Conv2d(16, 1, 1)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, image):
        image = (image - self.mean) / self.std
        x0 = self.stem(image)
        x1 = self.layer1(self.pool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x = self.dec4(torch.cat([self.up4(x4), x3], 1))
        x = self.dec3(torch.cat([self.up3(x), x2], 1))
        x = self.dec2(torch.cat([self.up2(x), x1], 1))
        x = self.dec1(torch.cat([self.up1(x), x0], 1))
        return self.head(self.up0(x))


def annotation_mask(image_shape, annotation):
    mask = np.zeros(image_shape, dtype=np.uint8)
    for polygon in annotation["segmentation"]:
        points = np.round(np.array(polygon, dtype=np.float32).reshape(-1, 2)).astype(np.int32)
        cv2.fillPoly(mask, [points], 1)
    return mask


def build_crop_arrays(image_ids, images, annotations):
    crop_images, crop_masks = [], []
    for image_id in tqdm(image_ids, desc="building refiner crops"):
        info = images[image_id]
        image = cv2.imread(str(CFG.train_dir / info["file_name"]), cv2.IMREAD_GRAYSCALE)
        for annotation in annotations.get(image_id, []):
            mask = annotation_mask(image.shape, annotation)
            if mask.sum() == 0:
                continue
            x0, y0, x1, y1 = square_bounds(mask)
            crop_images.append(
                cv2.resize(
                    image[y0:y1, x0:x1],
                    (CFG.crop_size, CFG.crop_size),
                    interpolation=cv2.INTER_LINEAR,
                )
            )
            crop_masks.append(
                cv2.resize(
                    mask[y0:y1, x0:x1],
                    (CFG.crop_size, CFG.crop_size),
                    interpolation=cv2.INTER_NEAREST,
                )
            )
    return np.stack(crop_images), np.stack(crop_masks)


class CropDataset(Dataset):
    def __init__(self, images, masks, augment=False):
        self.images = images
        self.masks = masks
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = torch.from_numpy(self.images[index].copy()).float() / 255
        mask = torch.from_numpy(self.masks[index].copy()).float().unsqueeze(0)
        if self.augment:
            rotations = random.randrange(4)
            image = torch.rot90(image, rotations, (0, 1))
            mask = torch.rot90(mask, rotations, (1, 2))
            if random.random() < 0.5:
                image = torch.flip(image, (1,))
                mask = torch.flip(mask, (2,))
            if random.random() < 0.5:
                image = image ** random.uniform(0.85, 1.2)
            if random.random() < 0.5:
                image = torch.clamp(image * random.uniform(0.9, 1.1), 0, 1)
        return image.unsqueeze(0).repeat(3, 1, 1), mask


def soft_dice_loss(logits, targets):
    probs = logits.sigmoid()
    intersection = (probs * targets).sum((1, 2, 3))
    denominator = probs.sum((1, 2, 3)) + targets.sum((1, 2, 3))
    return (1 - (2 * intersection + 1) / (denominator + 1)).mean()


def train_refiner(train_ids, valid_ids, images, annotations):
    train_images, train_masks = build_crop_arrays(train_ids, images, annotations)
    valid_images, valid_masks = build_crop_arrays(valid_ids, images, annotations)
    train_loader = DataLoader(
        CropDataset(train_images, train_masks, augment=True),
        batch_size=16,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        CropDataset(valid_images, valid_masks),
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CropUNet(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.refiner_epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    best_dice = -1.0

    for epoch in range(1, CFG.refiner_epochs + 1):
        model.train()
        for imgs, masks in tqdm(train_loader, desc=f"refiner {epoch}/{CFG.refiner_epochs}", leave=False):
            imgs = imgs.to(device)
            masks = masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                logits = model(imgs)
                loss = 0.5 * nn.functional.binary_cross_entropy_with_logits(
                    logits, masks
                ) + 0.5 * soft_dice_loss(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        model.eval()
        dice_total = 0.0
        n = 0
        with torch.no_grad():
            for imgs, masks in valid_loader:
                imgs = imgs.to(device)
                masks = masks.to(device).bool()
                pred = model(imgs).sigmoid() >= CFG.refiner_threshold
                inter = (pred & masks).sum((1, 2, 3)).float()
                denom = pred.sum((1, 2, 3)) + masks.sum((1, 2, 3))
                dice_total += ((2 * inter + 1e-7) / (denom + 1e-7)).sum().item()
                n += len(imgs)
        val_dice = dice_total / n
        print(f"refiner epoch={epoch} val_dice={val_dice:.4f}")
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), "/kaggle/working/best_refiner.pt")
        scheduler.step()

    model.load_state_dict(torch.load("/kaggle/working/best_refiner.pt", map_location=device))
    model.eval()
    return model


@torch.no_grad()
def refine_masks(image_gray, masks, refiner, threshold=CFG.refiner_threshold):
    device = next(refiner.parameters()).device
    refined = []
    for mask in masks:
        if mask.sum() == 0:
            continue
        x0, y0, x1, y1 = square_bounds(mask)
        crop = cv2.resize(
            image_gray[y0:y1, x0:x1],
            (CFG.crop_size, CFG.crop_size),
            interpolation=cv2.INTER_LINEAR,
        )
        tensor = (
            torch.from_numpy(crop).float().unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(device)
            / 255
        )
        prob = refiner(tensor).sigmoid()[0, 0].cpu().numpy()
        crop_mask = cv2.resize(prob, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR) >= threshold
        if crop_mask.any():
            full = np.zeros_like(mask, dtype=np.uint8)
            full[y0:y1, x0:x1] = crop_mask
            refined.append(full)
    return refined


def predict_instances(model, image_path, conf, refiner):
    image_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    result = model.predict(
        str(image_path),
        imgsz=CFG.imgsz,
        conf=conf,
        agnostic_nms=True,
        retina_masks=True,
        verbose=False,
    )[0]
    if result.masks is None:
        return []
    masks = [mask.cpu().numpy().astype(np.uint8) for mask in result.masks.data]
    return refine_masks(image_gray, masks, refiner)


def pairwise_iou(gt_masks, pred_masks):
    if not gt_masks or not pred_masks:
        return np.zeros((len(gt_masks), len(pred_masks)), dtype=np.float32)
    gt = np.stack(gt_masks).reshape(len(gt_masks), -1).astype(bool)
    pred = np.stack(pred_masks).reshape(len(pred_masks), -1).astype(bool)
    inter = gt.astype(np.uint8) @ pred.astype(np.uint8).T
    gt_area = gt.sum(axis=1, keepdims=True)
    pred_area = pred.sum(axis=1, keepdims=True).T
    union = gt_area + pred_area - inter
    return np.where(union == 0, 0, inter / union)


def pq_for_image(gt_masks, pred_masks, threshold=0.5):
    iou = pairwise_iou(gt_masks, pred_masks)
    if iou.size == 0:
        tp = 0
        fp = len(pred_masks)
        fn = len(gt_masks)
        return 0.0, tp, fp, fn
    hits = iou > threshold
    tp_scores = iou[hits]
    fp = int((hits.sum(axis=0) == 0).sum())
    fn = int((hits.sum(axis=1) == 0).sum())
    denom = len(tp_scores) + 0.5 * fp + 0.5 * fn
    return (float(tp_scores.sum() / denom) if denom else 0.0), len(tp_scores), fp, fn


def validation_sweep(yolo_model, refiner, valid_ids, images, annotations):
    confs = [0.15, 0.2, 0.25, 0.3, 0.35]
    best = (0.0, 0.25)
    for conf in confs:
        iou_sum = 0.0
        tp_total = fp_total = fn_total = 0
        for image_id in tqdm(valid_ids, desc=f"val conf {conf}"):
            info = images[image_id]
            image_path = CFG.train_dir / info["file_name"]
            gt_masks = []
            for ann in annotations.get(image_id, []):
                mask = annotation_mask((info["height"], info["width"]), ann)
                if mask.sum() >= CFG.min_component_size:
                    gt_masks.append(mask)
            pred_masks = [
                mask
                for mask in predict_instances(yolo_model, image_path, conf, refiner)
                if mask.sum() >= CFG.min_component_size
            ]
            pq, tp, fp, fn = pq_for_image(gt_masks, pred_masks)
            denom = tp + 0.5 * fp + 0.5 * fn
            iou_sum += pq * denom
            tp_total += tp
            fp_total += fp
            fn_total += fn
        final_pq = iou_sum / (tp_total + 0.5 * fp_total + 0.5 * fn_total)
        print(f"conf={conf:.2f} val_pq={final_pq:.4f} tp={tp_total} fp={fp_total} fn={fn_total}")
        if final_pq > best[0]:
            best = (final_pq, conf)
    return best


def write_submission(yolo_model, refiner, conf):
    rows = []
    test_files = sorted(CFG.test_dir.glob("*.jpeg"))
    for image_path in tqdm(test_files, desc="test inference"):
        counter = 1
        for mask in predict_instances(yolo_model, image_path, conf, refiner):
            if mask.sum() < CFG.min_component_size:
                continue
            rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
            rows.append(
                {
                    "filament_id": f"{image_path.stem}_{counter}",
                    "segmentation_rle": rle["counts"].decode("utf-8"),
                }
            )
            counter += 1
    submission = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    submission.to_csv("/kaggle/working/submission.csv", index=False)
    print(submission.head())
    print(f"submission rows={len(submission)} conf={conf}")


def main():
    seed_everything(CFG.seed)
    coco, images, annotations = load_coco()
    train_ids, valid_ids = prepare_yolo_data(coco, images, annotations)
    device = "0,1" if torch.cuda.device_count() > 1 else (0 if torch.cuda.is_available() else "cpu")
    print(f"torch_cuda={torch.cuda.is_available()} gpu_count={torch.cuda.device_count()} device={device}")

    yolo_model = YOLO(CFG.model)
    yolo_model.train(
        data=str(CFG.yolo_dir / "data.yaml"),
        epochs=CFG.epochs,
        imgsz=CFG.imgsz,
        batch=CFG.batch,
        device=device,
        patience=CFG.patience,
        seed=CFG.seed,
        single_cls=False,
        mask_ratio=CFG.mask_ratio,
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
        project="/kaggle/working/runs_filament",
        name="yolo_seg_optimized",
    )
    best_weights = Path(yolo_model.trainer.best)
    shutil.copy(best_weights, "/kaggle/working/yolo_best.pt")
    yolo_model = YOLO("/kaggle/working/yolo_best.pt")

    refiner = train_refiner(train_ids, valid_ids, images, annotations)
    best_pq, best_conf = validation_sweep(yolo_model, refiner, valid_ids, images, annotations)
    Path("/kaggle/working/validation_summary.txt").write_text(
        f"best_conf={best_conf}\nbest_pq={best_pq}\n"
    )
    write_submission(yolo_model, refiner, best_conf)


if __name__ == "__main__":
    main()
