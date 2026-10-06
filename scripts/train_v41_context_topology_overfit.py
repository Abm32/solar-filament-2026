"""V41 G2: bounded learned contextual texture/topology overfit gate.

Remote GPU runner. Reads exactly 16 frozen distinct fit images/records and runs
at most 2,000 steps. No tune/audit/test, CSV, export, or submission path exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import threading
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ANNOTATION_SHA256 = "5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1"
SELECTION_ID = "741d8124fd834c465743cb4ddbee265298354b4e59b5ab9650379d162673ca2c"
TILE_SIZE = 1024
TILE_STARTS = (0, 512, 1024)
SAMPLES = 144
MAX_STEPS = 2000
EVAL_STEPS = (0, 250, 500, 1000, 1500, 2000)
THRESHOLDS = (0.35, 0.50, 0.65)
MEMORY_GATE_MIB = 14336
RUNTIME_GATE_SECONDS = 7200
TARGET_PQ = 0.85
TARGET_RECALL = 0.90
TARGET_MEDIAN_IOU = 0.80
MAX_FP = 40
MIN_AREA = 20
MIN_SEED_AREA = 5
EXPECTED_RECORDS = 16
EXPECTED_IMAGES = 16
EXPECTED_ANNOTATIONS = 308
EXPECTED_SMALL_THIN = 244


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def annotation_rle(mask_utils, info: dict, annotation: dict) -> dict:
    return mask_utils.merge(
        mask_utils.frPyObjects(
            annotation["segmentation"],
            int(info["height"]),
            int(info["width"]),
        )
    )


def densify_polyline(np, spine: list[float]):
    points = np.asarray(spine, dtype=np.float64).reshape(-1, 2)
    if len(points) < 2 or not np.isfinite(points).all():
        raise ValueError("spine must contain two finite points")
    dense = []
    for first, second in zip(points[:-1], points[1:], strict=True):
        steps = max(1, int(np.ceil(np.max(np.abs(second - first)))))
        dense.extend(first + (second - first) * (index / steps) for index in range(steps))
    dense.append(points[-1])
    return np.rint(np.asarray(dense)).astype(np.int32)


def project_spine(np, cv2, mask, spine, radius=3):
    dense = densify_polyline(np, spine)
    projected = []
    height, width = mask.shape
    for x, y in dense:
        best = None
        for dy in range(-radius, radius + 1):
            yy = int(y + dy)
            if not 0 <= yy < height:
                continue
            for dx in range(-radius, radius + 1):
                xx = int(x + dx)
                if 0 <= xx < width and mask[yy, xx]:
                    candidate = (dx * dx + dy * dy, yy, xx)
                    if best is None or candidate < best:
                        best = candidate
        if best is not None:
            point = (best[1], best[2])
            if not projected or projected[-1] != point:
                projected.append(point)
    output = np.zeros(mask.shape, dtype=np.uint8)
    for yy, xx in projected:
        output[yy, xx] = 1
    for first, second in zip(projected[:-1], projected[1:], strict=True):
        cv2.line(output, (first[1], first[0]), (second[1], second[0]), 1, 1, cv2.LINE_8)
    output = output.astype(bool) & mask
    if len(projected) < 2 or int(output.sum()) < 2:
        raise RuntimeError("invalid projected V41 expert spine")
    return output


def selection_records(np, cv2, mask_utils, document: dict) -> list[dict]:
    images = {str(row["id"]): row for row in document["images"]}
    annotations = defaultdict(list)
    by_file = defaultdict(list)
    for row in document["annotations"]:
        annotations[str(row["image_id"])].append(row)
    for image_id, info in images.items():
        by_file[info["file_name"]].append(image_id)
    files = sorted(by_file)
    import random

    random.Random(42).shuffle(files)
    fit = sorted(files[141:])
    candidates = []
    for name in fit:
        image_id = max(
            by_file[name],
            key=lambda value: (
                len(annotations[value]),
                sum(row.get("area", 0) for row in annotations[value]),
                value,
            ),
        )
        info, entries = images[image_id], annotations[image_id]
        small = thin = small_thin = 0
        for entry in entries:
            mask = mask_utils.decode(annotation_rle(mask_utils, info, entry)).astype(bool)
            spine = project_spine(np, cv2, mask, entry["spine"])
            area = int(mask.sum())
            width = area / max(int(spine.sum()), 1)
            small += int(area <= 2000)
            thin += int(width <= 16)
            small_thin += int(area <= 2000 and width <= 16)
        candidates.append(
            {
                "file_name": name,
                "image_id": image_id,
                "annotations": len(entries),
                "small": small,
                "thin": thin,
                "small_thin": small_thin,
            }
        )
    candidates.sort(
        key=lambda row: (
            -row["small_thin"],
            -row["small"],
            -row["thin"],
            -row["annotations"],
            row["file_name"],
            row["image_id"],
        )
    )
    records = candidates[:EXPECTED_RECORDS]
    identity = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if identity != SELECTION_ID:
        raise RuntimeError(f"V41 selection changed: {identity}")
    totals = {
        "annotations": sum(row["annotations"] for row in records),
        "small_thin": sum(row["small_thin"] for row in records),
    }
    if totals != {"annotations": EXPECTED_ANNOTATIONS, "small_thin": EXPECTED_SMALL_THIN}:
        raise RuntimeError(f"V41 cohort totals changed: {totals}")
    return records


def image_channels(np, cv2, gray):
    value = gray.astype(np.float32)
    low, high = np.percentile(value, (1, 99))
    raw = np.clip((value - low) / max(float(high - low), 1.0), 0, 1).astype(np.float32)
    raw8 = np.rint(raw * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(raw8).astype(np.float32) / 255
    blur = cv2.GaussianBlur(raw, (15, 15), 0)
    dark = np.maximum(blur - raw, 0)
    dark /= max(float(dark.max()), 1e-6)
    gx = cv2.Sobel(raw, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(raw, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.hypot(gx, gy)
    edge /= max(float(edge.max()), 1e-6)
    return np.stack([raw, clahe, dark, edge]).astype(np.float32)


def prepare_records(np, cv2, ndi, mask_utils, document, records, image_dir: Path):
    images = {str(row["id"]): row for row in document["images"]}
    annotations = defaultdict(list)
    for row in document["annotations"]:
        annotations[str(row["image_id"])].append(row)
    if len({row["file_name"] for row in records}) != EXPECTED_IMAGES:
        raise RuntimeError("V41 cohort must use 16 distinct images")
    prepared = []
    for record in records:
        path = image_dir / record["file_name"]
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None or gray.shape != (2048, 2048):
            raise RuntimeError(f"cannot read frozen V41 fit image: {path}")
        info = images[record["image_id"]]
        entries = annotations[record["image_id"]]
        masks = [
            mask_utils.decode(annotation_rle(mask_utils, info, row)).astype(bool) for row in entries
        ]
        occupancy = sum(
            (mask.astype(np.uint8) for mask in masks), np.zeros((2048, 2048), dtype=np.uint8)
        )
        foreground = occupancy > 0
        spine = np.zeros_like(foreground)
        boundary = np.zeros_like(foreground)
        structure = ndi.generate_binary_structure(2, 1)
        for mask, entry in zip(masks, entries, strict=True):
            center = project_spine(np, cv2, mask, entry["spine"])
            yy, xx = np.ogrid[-2:3, -2:3]
            radius_two = (xx * xx + yy * yy) <= 4
            spine |= ndi.binary_dilation(center, structure=radius_two) & mask
            boundary |= ndi.binary_dilation(mask, structure=structure) ^ ndi.binary_erosion(
                mask, structure=structure, border_value=0
            )
        prepared.append(
            {
                "record": record,
                "channels": image_channels(np, cv2, gray),
                "foreground": foreground.astype(np.float32),
                "spine": spine.astype(np.float32),
                "boundary": boundary.astype(np.float32),
                "valid": (occupancy <= 1),
                "targets": [annotation_rle(mask_utils, info, row) for row in entries],
            }
        )
    return prepared


def sample_tile(torch, prepared, sample_index, device):
    record_index, tile_index = divmod(sample_index, 9)
    y0, x0 = TILE_STARTS[tile_index // 3], TILE_STARTS[tile_index % 3]
    item = prepared[record_index]
    sl = np_slice = (slice(y0, y0 + TILE_SIZE), slice(x0, x0 + TILE_SIZE))
    image = torch.from_numpy(item["channels"][:, sl[0], sl[1]]).to(device)
    targets = [
        torch.from_numpy(item[key][np_slice]).to(device)
        for key in ("foreground", "spine", "boundary", "valid")
    ]
    return image.unsqueeze(0), *(value.unsqueeze(0) for value in targets)


def build_model(torch, device):
    functional = torch.nn.functional

    class SE(torch.nn.Module):
        def __init__(self, channels):
            super().__init__()
            hidden = max(8, channels // 4)
            self.layers = torch.nn.Sequential(
                torch.nn.AdaptiveAvgPool2d(1),
                torch.nn.Conv2d(channels, hidden, 1),
                torch.nn.SiLU(),
                torch.nn.Conv2d(hidden, channels, 1),
                torch.nn.Sigmoid(),
            )

        def forward(self, value):
            return value * self.layers(value)

    class ResidualSE(torch.nn.Module):
        def __init__(self, incoming, outgoing, stride=1):
            super().__init__()
            self.body = torch.nn.Sequential(
                torch.nn.Conv2d(incoming, outgoing, 3, stride=stride, padding=1, bias=False),
                torch.nn.GroupNorm(max(1, outgoing // 8), outgoing),
                torch.nn.SiLU(),
                torch.nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False),
                torch.nn.GroupNorm(max(1, outgoing // 8), outgoing),
                SE(outgoing),
            )
            self.skip = (
                torch.nn.Identity()
                if incoming == outgoing and stride == 1
                else torch.nn.Conv2d(incoming, outgoing, 1, stride=stride, bias=False)
            )
            self.activation = torch.nn.SiLU()

        def forward(self, value):
            return self.activation(self.body(value) + self.skip(value))

    class Up(torch.nn.Module):
        def __init__(self, incoming, skip, outgoing):
            super().__init__()
            self.block = ResidualSE(incoming + skip, outgoing)

        def forward(self, value, skip):
            value = functional.interpolate(
                value, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
            return self.block(torch.cat([value, skip], dim=1))

    class ContextTopologyUNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = torch.nn.Sequential(
                torch.nn.Conv2d(4, 24, 3, padding=1, bias=False),
                torch.nn.GroupNorm(3, 24),
                torch.nn.SiLU(),
            )
            self.e1 = ResidualSE(24, 24)
            self.e2 = ResidualSE(24, 48, 2)
            self.e3 = ResidualSE(48, 96, 2)
            self.e4 = ResidualSE(96, 144, 2)
            self.context = torch.nn.ModuleList(
                [
                    torch.nn.Conv2d(144, 36, 3, padding=dilation, dilation=dilation)
                    for dilation in (1, 2, 4, 8)
                ]
            )
            self.context_fuse = ResidualSE(144, 144)
            self.u3, self.u2, self.u1 = Up(144, 96, 96), Up(96, 48, 48), Up(48, 24, 24)
            self.texture = torch.nn.Conv2d(24, 2, 1)
            self.boundary = torch.nn.Conv2d(24, 1, 1)
            self.topology = torch.nn.Sequential(
                ResidualSE(29, 32),
                ResidualSE(32, 32),
                ResidualSE(32, 32),
                torch.nn.Conv2d(32, 1, 1),
            )

        def forward(self, value):
            x1 = self.e1(self.stem(value))
            x2 = self.e2(x1)
            x3 = self.e3(x2)
            x4 = self.e4(x3)
            context = torch.cat([functional.silu(branch(x4)) for branch in self.context], dim=1)
            decoded = self.u1(self.u2(self.u3(self.context_fuse(context), x3), x2), x1)
            texture = self.texture(decoded)
            topology_input = torch.cat(
                [
                    decoded,
                    texture[:, :1].sigmoid(),
                    texture[:, 1:2].sigmoid(),
                    value[:, 0:1],
                    value[:, 2:3],
                    value[:, 3:4],
                ],
                dim=1,
            )
            residual = self.topology(topology_input)
            return {
                "final": texture[:, :1] + residual,
                "coarse": texture[:, :1],
                "spine": texture[:, 1:2],
                "boundary": self.boundary(decoded),
            }

    return ContextTopologyUNet().to(device)


def balanced_bce(torch, logits, target, valid):
    values = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    positive, negative = valid & (target >= 0.5), valid & (target < 0.5)
    terms = []
    if positive.any():
        terms.append(values[positive].mean())
    if negative.any():
        terms.append(values[negative].mean())
    if not terms:
        return logits.sum() * 0.0
    return sum(terms) / len(terms)


def soft_dice(torch, logits, target, valid):
    probability = logits.sigmoid()
    intersection = (probability * target * valid).sum()
    denominator = (probability * valid).sum() + (target * valid).sum()
    return 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)


def loss_terms(torch, output, foreground, spine, boundary, valid):
    valid = valid.bool().unsqueeze(1)
    foreground, spine, boundary = (value.unsqueeze(1) for value in (foreground, spine, boundary))
    terms = {
        "final_bce": balanced_bce(torch, output["final"], foreground, valid),
        "final_dice": soft_dice(torch, output["final"], foreground, valid),
        "coarse_bce": balanced_bce(torch, output["coarse"], foreground, valid),
        "spine_bce": balanced_bce(torch, output["spine"], spine, valid),
        "spine_dice": soft_dice(torch, output["spine"], spine, valid),
        "boundary_bce": balanced_bce(torch, output["boundary"], boundary, valid),
    }
    total = (
        2 * terms["final_bce"]
        + terms["final_dice"]
        + 0.5 * terms["coarse_bce"]
        + terms["spine_bce"]
        + terms["spine_dice"]
        + 0.5 * terms["boundary_bce"]
    )
    return total, terms


def hann_window(torch, device):
    one = 0.1 + 0.9 * torch.hann_window(TILE_SIZE, periodic=False, device=device)
    return one[:, None] * one[None, :]


def clean_components(np, ndi, value, minimum):
    labels, count = ndi.label(value)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    return value & (sizes[labels] >= minimum)


def decode_instances(
    np, ndi, watershed, mask_utils, foreground_probability, spine_probability, threshold
):
    foreground = clean_components(np, ndi, foreground_probability >= threshold, MIN_AREA)
    if not foreground.any():
        return []
    seeds = clean_components(np, ndi, (spine_probability >= threshold) & foreground, MIN_SEED_AREA)
    markers, count = ndi.label(seeds)
    components, component_count = ndi.label(foreground)
    for component in range(1, component_count + 1):
        region = components == component
        if not np.any(markers[region]):
            index = int(np.argmax(np.where(region, spine_probability, -np.inf)))
            y, x = np.unravel_index(index, region.shape)
            count += 1
            markers[y, x] = count
    labels = watershed(-foreground_probability, markers=markers, mask=foreground)
    predictions = []
    for label in range(1, int(labels.max()) + 1):
        value = labels == label
        if int(value.sum()) >= MIN_AREA:
            predictions.append(mask_utils.encode(np.asfortranarray(value.astype(np.uint8))))
    return predictions


def pq_metric(np, mask_utils, predictions, targets):
    if not predictions or not targets:
        counts = [0.0, 0, len(predictions), len(targets)]
        matched = []
    else:
        iou = np.asarray(mask_utils.iou(predictions, targets, [0] * len(targets)), dtype=float)
        hits = np.argwhere(iou > 0.5)
        order = sorted(hits.tolist(), key=lambda pair: (-iou[pair[0], pair[1]], pair))
        used_p, used_t, matched = set(), set(), []
        for p, t in order:
            if p not in used_p and t not in used_t:
                used_p.add(p)
                used_t.add(t)
                matched.append(float(iou[p, t]))
        counts = [
            sum(matched),
            len(matched),
            len(predictions) - len(matched),
            len(targets) - len(matched),
        ]
    iou_sum, tp, fp, fn = counts
    denominator = tp + 0.5 * (fp + fn)
    return {
        "pq": iou_sum / denominator if denominator else 0.0,
        "sq": iou_sum / tp if tp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "iou_sum": float(iou_sum),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }, matched


def evaluate(np, torch, ndi, watershed, mask_utils, model, prepared, device):
    model.eval()
    blend = hann_window(torch, device)
    probabilities = {}
    with torch.inference_mode():
        for item in prepared:
            accumulator = torch.zeros((2, 2048, 2048), device=device)
            weights = torch.zeros((2048, 2048), device=device)
            for y0 in TILE_STARTS:
                for x0 in TILE_STARTS:
                    image = torch.from_numpy(
                        item["channels"][:, y0 : y0 + TILE_SIZE, x0 : x0 + TILE_SIZE]
                    ).to(device)
                    output = model(image.unsqueeze(0))
                    pair = torch.cat([output["final"], output["spine"]], dim=1)[0].sigmoid()
                    accumulator[:, y0 : y0 + TILE_SIZE, x0 : x0 + TILE_SIZE] += pair * blend
                    weights[y0 : y0 + TILE_SIZE, x0 : x0 + TILE_SIZE] += blend
            probabilities[item["record"]["image_id"]] = (
                (accumulator / weights.clamp_min(1e-6)).cpu().numpy()
            )
    cells = []
    for threshold in THRESHOLDS:
        totals = np.zeros(4, dtype=float)
        rows = []
        all_matched = []
        predicted_pixels = 0
        for item in prepared:
            probability = probabilities[item["record"]["image_id"]]
            predictions = decode_instances(
                np, ndi, watershed, mask_utils, probability[0], probability[1], threshold
            )
            metric, matched = pq_metric(np, mask_utils, predictions, item["targets"])
            totals += [metric["iou_sum"], metric["tp"], metric["fp"], metric["fn"]]
            all_matched.extend(matched)
            predicted_pixels += int((probability[0] >= threshold).sum())
            rows.append({**item["record"], "predictions": len(predictions), **metric})
        aggregate, _ = pq_metric(np, mask_utils, [], [])
        iou_sum, tp, fp, fn = totals
        denominator = tp + 0.5 * (fp + fn)
        aggregate = {
            "pq": float(iou_sum / denominator if denominator else 0),
            "sq": float(iou_sum / tp if tp else 0),
            "recall": float(tp / (tp + fn) if tp + fn else 0),
            "iou_sum": float(iou_sum),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "median_matched_iou": float(np.median(all_matched)) if all_matched else 0.0,
            "predicted_foreground_fraction": predicted_pixels / (len(prepared) * 2048 * 2048),
        }
        cells.append({"threshold": threshold, "metrics": aggregate, "rows": rows})
    return max(
        cells,
        key=lambda cell: (
            cell["metrics"]["pq"],
            cell["threshold"],
            -sum(row["predictions"] for row in cell["rows"]),
        ),
    ), cells


class MemoryMonitor:
    def __init__(self):
        self.peak_mib = 0.0
        self.samples = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop.is_set():
            try:
                text = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    text=True,
                    timeout=5,
                )
                values = [float(line.strip()) for line in text.splitlines() if line.strip()]
                value = max(values, default=0.0)
                self.peak_mib = max(self.peak_mib, value)
                self.samples.append(value)
            except Exception:
                pass
            self.stop.wait(1)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=5)


def gpu_preflight(torch):
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"V41 requires exactly one visible CUDA GPU, found {torch.cuda.device_count()}"
        )
    properties = torch.cuda.get_device_properties(0)
    name = properties.name
    capability = (properties.major, properties.minor)
    total_mib = properties.total_memory / 2**20
    if "T4" not in name.upper() or capability < (7, 5) or total_mib < 14000:
        raise RuntimeError(f"V41 unsupported GPU provenance: {name}, {capability}, {total_mib}")
    return {
        "name": name,
        "capability": list(capability),
        "total_mib": total_mib,
        "visible_devices": torch.cuda.device_count(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=MAX_STEPS)
    args = parser.parse_args()
    if not 0 <= args.steps <= MAX_STEPS:
        parser.error(f"--steps must be in [0,{MAX_STEPS}]")
    if sha256(args.annotations) != ANNOTATION_SHA256:
        raise RuntimeError("V41 annotation identity changed")

    import cv2
    import numpy as np
    import torch
    from pycocotools import mask as mask_utils
    from scipy import ndimage as ndi
    from skimage.segmentation import watershed

    provenance = gpu_preflight(torch)  # Must precede all image/model work.
    torch.manual_seed(20260914)
    np.random.seed(20260914)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda:0")
    document = json.loads(args.annotations.read_text())
    records = selection_records(np, cv2, mask_utils, document)
    prepared = prepare_records(np, cv2, ndi, mask_utils, document, records, args.images)
    model = build_model(torch, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    history = []
    loss_history = []
    started = time.monotonic()
    finite_losses = True
    torch.cuda.reset_peak_memory_stats()

    def checkpoint(step):
        selected, cells = evaluate(np, torch, ndi, watershed, mask_utils, model, prepared, device)
        row = {
            "step": step,
            "selected": selected,
            "cells": cells,
            "elapsed_seconds": time.monotonic() - started,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        }
        history.append(row)
        print(
            json.dumps(
                {
                    "step": step,
                    **selected["metrics"],
                    "threshold": selected["threshold"],
                    "peak_allocated_mib": row["peak_allocated_mib"],
                }
            ),
            flush=True,
        )
        metric = selected["metrics"]
        return (
            metric["pq"] >= TARGET_PQ
            and metric["recall"] >= TARGET_RECALL
            and metric["median_matched_iou"] >= TARGET_MEDIAN_IOU
            and metric["fp"] <= MAX_FP
            and 0 < metric["predicted_foreground_fraction"] < 0.02
        )

    with MemoryMonitor() as monitor:
        success = checkpoint(0)
        for step in range(1, args.steps + 1):
            model.train()
            batch = sample_tile(torch, prepared, (step - 1) % SAMPLES, device)
            image, foreground, spine, boundary, valid = batch
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=True):
                output = model(image)
                loss, terms = loss_terms(torch, output, foreground, spine, boundary, valid)
            if not torch.isfinite(loss):
                finite_losses = False
                raise RuntimeError(f"nonfinite V41 loss at step {step}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if step % 25 == 0:
                record = {
                    "train_step": step,
                    "loss": float(loss),
                    **{key: float(value) for key, value in terms.items()},
                }
                loss_history.append(record)
                print(json.dumps(record), flush=True)
            if step in EVAL_STEPS:
                success = checkpoint(step)
                if success:
                    break
        elapsed = time.monotonic() - started

    checkpoint_path = args.output.with_suffix(".pt")
    torch.save(model.state_dict(), checkpoint_path)
    reloaded = build_model(torch, device)
    reloaded.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True), strict=True
    )
    reloaded.eval()
    model.eval()
    probe = sample_tile(torch, prepared, 0, device)[0]
    with torch.inference_mode():
        reload_delta = max(
            float((model(probe)[key] - reloaded(probe)[key]).abs().max())
            for key in ("final", "coarse", "spine", "boundary")
        )
    selected = history[-1]["selected"]
    metric = selected["metrics"]
    gates = {
        "pq": metric["pq"] >= TARGET_PQ,
        "recall": metric["recall"] >= TARGET_RECALL,
        "median_iou": metric["median_matched_iou"] >= TARGET_MEDIAN_IOU,
        "fp": metric["fp"] <= MAX_FP,
        "foreground_fraction": 0 < metric["predicted_foreground_fraction"] < 0.02,
        "steps": history[-1]["step"] == MAX_STEPS or success,
        "finite": finite_losses,
        "reload": reload_delta == 0.0,
        "allocated_memory": torch.cuda.max_memory_allocated() / 2**20 < MEMORY_GATE_MIB,
        "external_memory": 0 < monitor.peak_mib < MEMORY_GATE_MIB,
        "runtime": elapsed <= RUNTIME_GATE_SECONDS,
        "firewall": True,
    }
    passed = all(gates.values())
    result = {
        "status": "complete",
        "experiment": "v41_context_topology_g2",
        "selection_id": SELECTION_ID,
        "records": records,
        "samples": SAMPLES,
        "requested_steps": args.steps,
        "completed_steps": history[-1]["step"],
        "history": history,
        "loss_history": loss_history,
        "selected": selected,
        "gpu": provenance,
        "elapsed_seconds": elapsed,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_external_mib": monitor.peak_mib,
        "memory_samples": monitor.samples,
        "reload_max_abs_delta": reload_delta,
        "checkpoint_sha256": sha256(checkpoint_path),
        "gates": gates,
        "g2_passed": passed,
        "full_training_design_authorized": passed,
        "reads_tune": False,
        "reads_audit": False,
        "reads_test": False,
        "exports": False,
        "submits": False,
        "action": "freeze separate V41 full-population design"
        if passed
        else "close V41 before full training",
    }
    write_json(args.output, result)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "completed_steps",
                    "selected",
                    "elapsed_seconds",
                    "peak_allocated_mib",
                    "peak_external_mib",
                    "gates",
                    "g2_passed",
                    "action",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
