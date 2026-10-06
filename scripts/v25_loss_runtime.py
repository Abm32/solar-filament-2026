"""Pinned Ultralytics 8.4.138 runtime for V25 uncertain-background weighting.

This module is GPU-runtime-only. The local test environment intentionally lacks
Torch and Ultralytics; import it only from V25 train modes on the Azure runtime.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.segment.train import SegmentationTrainer
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils.loss import v8SegmentationLoss
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.tal import make_anchors
from ultralytics.utils.torch_utils import unwrap_model

UNCERTAINTY_JSON: Path | None = None
UNCERTAINTY_WEIGHT = 1.0
SENTINEL_CLASS = -1.0


class UncertaintyDataset(YOLODataset):
    """Append omitted-rater regions after normal class filtering.

    Sentinel instances participate in every geometric transform exactly like
    normal instances, but the custom loss removes them before assignment.
    Rectangle segments exist only to keep segmentation transforms aligned; they
    are never used as mask targets.
    """

    def __init__(self, *args, uncertainty_json: Path, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.augment:
            return
        mapping = json.loads(Path(uncertainty_json).read_text())
        appended = 0
        for label in self.labels:
            stem = Path(label["im_file"]).stem
            entries = mapping.get(stem, [])
            if not entries:
                continue
            boxes = np.asarray([entry["xywhn"] for entry in entries], dtype=np.float32).reshape(-1, 4)
            classes = np.full((len(entries), 1), SENTINEL_CLASS, dtype=np.float32)
            segments = []
            for x, y, w, h in boxes:
                left, right = x - w / 2, x + w / 2
                top, bottom = y - h / 2, y + h / 2
                segments.append(np.asarray(
                    [[left, top], [right, top], [right, bottom], [left, bottom]],
                    dtype=np.float32,
                ))
            label["bboxes"] = np.concatenate((label["bboxes"], boxes), axis=0)
            label["cls"] = np.concatenate((label["cls"], classes), axis=0)
            label["segments"].extend(segments)
            appended += len(entries)
        expected = sum(len(entries) for entries in mapping.values())
        if appended != expected:
            raise RuntimeError(f"uncertainty transport mismatch: appended {appended}, expected {expected}")
        self.uncertainty_instances = appended


class MissingLabelSegmentationModel(SegmentationModel):
    """Segmentation model whose criterion downweights known uncertain negatives."""

    def init_criterion(self):
        return MissingLabelSegmentationLoss(self, uncertainty_weight=UNCERTAINTY_WEIGHT)


class MissingLabelSegmentationLoss(v8SegmentationLoss):
    """Keep original positives; downweight only negative anchors in uncertainty boxes."""

    def __init__(self, model, uncertainty_weight: float):
        super().__init__(model)
        if not 0 < uncertainty_weight <= 1:
            raise ValueError("uncertainty_weight must be in (0, 1]")
        self.uncertainty_weight = float(uncertainty_weight)
        self.last_uncertain_negative_anchors = 0
        self.last_total_negative_anchors = 0
        self.total_uncertain_negative_anchors = 0
        self.total_negative_anchors = 0
        self.loss_batches = 0

    @staticmethod
    def split_uncertainty(batch: dict) -> dict:
        sentinel = batch["cls"].view(-1) < 0
        clean = dict(batch)
        clean["uncertainty_bboxes"] = batch["bboxes"][sentinel]
        clean["uncertainty_batch_idx"] = batch["batch_idx"][sentinel]
        keep = ~sentinel
        for key in ("cls", "bboxes", "batch_idx", "masks", "keypoints", "segments"):
            value = batch.get(key)
            if value is not None and hasattr(value, "shape") and value.shape[0] == sentinel.shape[0]:
                clean[key] = value[keep]
        if (clean["cls"] < 0).any():
            raise RuntimeError("sentinel leaked into positive supervision")
        return clean

    @staticmethod
    def uncertainty_weights(
        anchor_points_px: torch.Tensor,
        uncertainty_xywhn: torch.Tensor,
        uncertainty_batch_idx: torch.Tensor,
        batch_size: int,
        image_size_hw: torch.Tensor,
        fg_mask: torch.Tensor,
        uncertain_weight: float,
    ) -> torch.Tensor:
        weights = torch.ones(
            (batch_size, anchor_points_px.shape[0], 1),
            dtype=anchor_points_px.dtype,
            device=anchor_points_px.device,
        )
        if uncertainty_xywhn.numel() == 0 or uncertain_weight == 1.0:
            return weights
        scale = image_size_hw[[1, 0, 1, 0]].to(uncertainty_xywhn)
        boxes = xywh2xyxy(uncertainty_xywhn * scale)
        for batch_index in range(batch_size):
            selected = boxes[uncertainty_batch_idx.long() == batch_index]
            if not len(selected):
                continue
            x, y = anchor_points_px[:, 0], anchor_points_px[:, 1]
            inside = (
                (x[:, None] >= selected[None, :, 0])
                & (x[:, None] <= selected[None, :, 2])
                & (y[:, None] >= selected[None, :, 1])
                & (y[:, None] <= selected[None, :, 3])
            ).any(dim=1)
            uncertain_negative = inside & ~fg_mask[batch_index].bool()
            weights[batch_index, uncertain_negative, 0] = uncertain_weight
        return weights

    def get_assigned_targets_and_loss(self, preds: dict, batch: dict) -> tuple:
        loss = torch.zeros(3, device=self.device)
        pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
        pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(
            preds["feats"][0].shape[2:], device=self.device, dtype=dtype
        ) * self.stride[0]

        targets = torch.cat(
            (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1
        )
        targets = self.preprocess(
            targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]]
        )
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        target_scores_sum = max(target_scores.sum(), 1)
        bce_loss = self.bce(pred_scores, target_scores.to(dtype))
        if self.class_weights is not None:
            bce_loss *= self.class_weights
        cls_weights = self.uncertainty_weights(
            anchor_points * stride_tensor,
            batch["uncertainty_bboxes"].to(self.device),
            batch["uncertainty_batch_idx"].to(self.device),
            batch_size,
            imgsz,
            fg_mask,
            self.uncertainty_weight,
        )
        bce_loss *= cls_weights
        self.last_uncertain_negative_anchors = int((cls_weights < 1).sum().item())
        self.last_total_negative_anchors = int((~fg_mask.bool()).sum().item())
        self.total_uncertain_negative_anchors += self.last_uncertain_negative_anchors
        self.total_negative_anchors += self.last_total_negative_anchors
        self.loss_batches += 1
        loss[1] = bce_loss.sum() / target_scores_sum

        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )
        else:
            loss[0] += pred_distri[..., :0].sum()
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            dict(zip(self.loss_names, loss.detach())),
        )

    def loss(self, preds: dict, batch: dict):
        return super().loss(preds, self.split_uncertainty(batch))


class MissingLabelTrainer(SegmentationTrainer):
    """Trainer binding uncertainty transport and custom criterion to stock YOLO."""

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = self.set_model_names_for_load(
            MissingLabelSegmentationModel(
                cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose
            )
        )
        if weights:
            model.load(weights)
        return model

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        if UNCERTAINTY_JSON is None:
            raise RuntimeError("UNCERTAINTY_JSON is not configured")
        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        pad = 0.0 if mode == "train" else 0.5
        return UncertaintyDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=mode == "val",
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=gs,
            pad=pad,
            prefix=f"{mode}: ",
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
            fraction=1.0,
            uncertainty_json=UNCERTAINTY_JSON,
        )


def configure(uncertainty_json: Path, uncertainty_weight: float) -> type[MissingLabelTrainer]:
    global UNCERTAINTY_JSON, UNCERTAINTY_WEIGHT
    UNCERTAINTY_JSON = Path(uncertainty_json)
    UNCERTAINTY_WEIGHT = float(uncertainty_weight)
    return MissingLabelTrainer
