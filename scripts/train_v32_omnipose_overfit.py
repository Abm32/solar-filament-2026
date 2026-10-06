"""V32 G2: bounded 16-record Omnipose image-model overfit gate.

Remote-only runner. It reads exactly the frozen first 16 fit annotator records
and nine official fit images. It cannot read tune/audit/test, export CSV, or
submit. Full training is forbidden here; maximum steps is 2,000.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os

# Must be set before any function-local Torch import is executed.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import random
import sys
import time
import types
import warnings
from collections import defaultdict
from pathlib import Path

ANNOTATION_SHA256 = "5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1"
BACKBONE_SHA256 = "0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a"
SELECTION_ID = "e561c523b60993e93f82ce03116737ce512f54f4103b4557c29e409c88c92653"
UPSTREAM_COMMIT = "21e4d0199cc03662669c150f6151e3365dc201fd"
UPSTREAM_CORE_SHA256 = "300f4cf24ad5256c35726442e024efa9e98cb27b200a3f36a6da01e3839ab332"
TILE_SIZE = 1024
TILE_STARTS = (0, 512, 1024)
SAMPLES = 16 * 9
MAX_STEPS = 2000
EVAL_STEPS = (0, 250, 500, 1000, 1500, 2000)
TARGET_PQ = 0.85
MEMORY_GATE_MIB = 14336
MIN_AREA = 20


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def selection_records(document: dict) -> list[dict]:
    images = {str(row["id"]): row for row in document["images"]}
    annotations: dict[str, list[dict]] = defaultdict(list)
    by_file: dict[str, list[str]] = defaultdict(list)
    for row in document["annotations"]:
        annotations[str(row["image_id"])].append(row)
    for image_id, info in images.items():
        by_file[info["file_name"]].append(image_id)
    files = sorted(by_file); random.Random(42).shuffle(files)
    fit = sorted(files[141:])
    pairs = [(name, image_id) for name in fit for image_id in sorted(by_file[name])][:16]
    records = [{"file_name": name, "image_id": image_id,
                "annotations": len(annotations[image_id])} for name, image_id in pairs]
    identity = hashlib.sha256(json.dumps(
        records, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    if identity != SELECTION_ID:
        raise RuntimeError(f"V32 G2 selection changed: {identity}")
    return records


def load_upstream(upstream: Path):
    source = upstream.resolve() / "omnipose"
    package = types.ModuleType("omnipose")
    package.__path__ = [str(source)]
    package.__package__ = "omnipose"
    sys.modules["omnipose"] = package
    return importlib.import_module("omnipose.core")


def build_model(backbone_path: Path, device):
    import torch
    from torchvision.models.segmentation import deeplabv3_resnet50
    model = deeplabv3_resnet50(weights=None, weights_backbone=None, aux_loss=False)
    state = torch.load(backbone_path, map_location="cpu", weights_only=True)
    result = model.backbone.load_state_dict(state, strict=False)
    expected_unexpected = {"fc.weight", "fc.bias"}
    if set(result.unexpected_keys) != expected_unexpected or result.missing_keys:
        raise RuntimeError(
            f"unexpected backbone load result: missing={result.missing_keys} "
            f"unexpected={result.unexpected_keys}"
        )
    head = torch.nn.Conv2d(256, 4, kernel_size=1)
    torch.nn.init.zeros_(head.weight)
    torch.nn.init.zeros_(head.bias)
    with torch.no_grad():
        head.bias[2] = -5.0
        head.bias[3] = -5.0
    model.classifier[-1] = head
    return model.to(device)


def set_batchnorm_eval(torch, model) -> None:
    """Use frozen running statistics for the G2 batch-one contract."""
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


def hann_window(torch, device):
    one = torch.hann_window(TILE_SIZE, periodic=False, device=device)
    one = 0.1 + 0.9 * one
    return one[:, None] * one[None, :]


def annotation_rle(mask_utils, info: dict, annotation: dict) -> dict:
    return mask_utils.merge(mask_utils.frPyObjects(
        annotation["segmentation"], int(info["height"]), int(info["width"]),
    ))


def pq_counts(np, mask_utils, predictions: list[dict], targets: list[dict]):
    if not predictions or not targets:
        return np.asarray([0.0, 0, len(predictions), len(targets)], dtype=np.float64)
    iou = np.asarray(mask_utils.iou(predictions, targets, [0] * len(targets)), dtype=np.float64)
    hits = iou > 0.5
    return np.asarray([float(iou[hits].sum()), int(hits.sum()),
                       int((~hits.any(axis=1)).sum()), int((~hits.any(axis=0)).sum())],
                      dtype=np.float64)


def metric(counts) -> dict:
    iou_sum, tp, fp, fn = map(float, counts)
    denominator = tp + 0.5 * (fp + fn)
    return {"pq": iou_sum / denominator if denominator else 0.0,
            "sq": iou_sum / tp if tp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "iou_sum": iou_sum, "tp": int(tp), "fp": int(fp), "fn": int(fn)}


def decode_fields(np, mask_utils, core, flow, distance, info):
    foreground = distance >= 0.0
    if not np.any(foreground):
        return []
    positive = distance[foreground]
    niter = max(1, int(2 * (foreground.ndim + 1) * float(np.mean(positive))))
    output = core.compute_masks(
        flow, distance, bd=np.zeros_like(distance), iscell=foreground,
        niter=niter, flow_threshold=0, omni=True, use_gpu=False,
        device=core.torch_CPU, nclasses=3, dim=2, min_size=MIN_AREA,
        cluster=False, suppress=True, diam_threshold=12.0, eps=2 ** 0.5,
        affinity_seg=False, hdbscan=False,
    )[0]
    predictions = []
    for label in range(1, int(output.max()) + 1):
        value = output == label
        if int(value.sum()) < MIN_AREA:
            continue
        predictions.append(mask_utils.encode(np.asfortranarray(value.astype(np.uint8))))
    return predictions


def normalize_tile(np, image):
    value = image.astype(np.float32)
    low, high = np.percentile(value, (1, 99))
    value = np.clip((value - low) / max(float(high - low), 1.0), 0, 1)
    return value.astype(np.float32, copy=False)


def prepare_records(np, cv2, mask_utils, core, document, records, image_dir: Path):
    images = {str(row["id"]): row for row in document["images"]}
    annotations: dict[str, list[dict]] = defaultdict(list)
    for row in document["annotations"]:
        annotations[str(row["image_id"])].append(row)
    allowed_names = {record["file_name"] for record in records}
    if len(allowed_names) != 9:
        raise RuntimeError(f"expected exactly nine allowed G2 images, got {len(allowed_names)}")
    image_cache = {}
    prepared = []
    for record in records:
        name, image_id = record["file_name"], record["image_id"]
        if name not in image_cache:
            path = image_dir / name
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None or image.shape != (2048, 2048):
                raise RuntimeError(f"cannot read frozen fit image: {path}")
            image_cache[name] = image
        info = images[image_id]
        entries = annotations[image_id]
        masks = [mask_utils.decode(annotation_rle(mask_utils, info, row)).astype(bool)
                 for row in entries]
        occupancy = sum((value.astype(np.uint8) for value in masks),
                        np.zeros((2048, 2048), dtype=np.uint8))
        labels = np.zeros((2048, 2048), dtype=np.int32)
        for label, value in enumerate(masks, 1):
            labels[value & (occupancy == 1)] = label
        target = core.masks_to_flows(
            labels.copy(), use_gpu=False, device=core.torch_CPU, omni=True, dim=2,
        )
        distance = target[3].cpu().numpy().astype(np.float32)
        flow = target[4].cpu().numpy().astype(np.float32) * 5.0
        boundary = np.asarray(target[2], dtype=np.float32)
        distance[labels == 0] = -5.0
        weight = np.where(labels > 0, 1.0, 0.5).astype(np.float32)
        weight[occupancy > 1] = 0.0
        prepared.append({
            "record": record, "info": info, "image": image_cache[name],
            "flow": flow, "distance": distance, "boundary": boundary,
            "weight": weight, "ignore": occupancy > 1,
            "targets": [annotation_rle(mask_utils, info, row) for row in entries],
        })
    return prepared


def sample_tile(np, torch, prepared, sample_index, device):
    record_index = sample_index // 9
    tile_index = sample_index % 9
    y0 = TILE_STARTS[tile_index // 3]
    x0 = TILE_STARTS[tile_index % 3]
    item = prepared[record_index]
    image = normalize_tile(np, item["image"][y0:y0 + TILE_SIZE, x0:x0 + TILE_SIZE])
    image = torch.from_numpy(image).to(device).unsqueeze(0).repeat(3, 1, 1)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[:, None, None]
    image = (image - mean) / std
    flow = torch.from_numpy(item["flow"][:, y0:y0 + TILE_SIZE, x0:x0 + TILE_SIZE]).to(device)
    distance = torch.from_numpy(item["distance"][y0:y0 + TILE_SIZE, x0:x0 + TILE_SIZE]).to(device)
    boundary = torch.from_numpy(item["boundary"][y0:y0 + TILE_SIZE, x0:x0 + TILE_SIZE]).to(device)
    weight = torch.from_numpy(item["weight"][y0:y0 + TILE_SIZE, x0:x0 + TILE_SIZE]).to(device)
    return image.unsqueeze(0), flow.unsqueeze(0), distance.unsqueeze(0), boundary.unsqueeze(0), weight.unsqueeze(0)


def evaluate(np, torch, mask_utils, core, model, prepared, device):
    model.eval()
    blend = hann_window(torch, device)
    totals = np.zeros(4, dtype=np.float64)
    rows = []
    with torch.inference_mode():
        image_predictions = {}
        for name in sorted({item["record"]["file_name"] for item in prepared}):
            source = next(item["image"] for item in prepared if item["record"]["file_name"] == name)
            accumulator = torch.zeros((4, 2048, 2048), device=device)
            weights = torch.zeros((2048, 2048), device=device)
            for y0 in TILE_STARTS:
                for x0 in TILE_STARTS:
                    tile = normalize_tile(np, source[y0:y0 + TILE_SIZE, x0:x0 + TILE_SIZE])
                    tensor = torch.from_numpy(tile).to(device).unsqueeze(0).repeat(3, 1, 1)
                    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[:, None, None]
                    std = torch.tensor([0.229, 0.224, 0.225], device=device)[:, None, None]
                    output = model(((tensor - mean) / std).unsqueeze(0))["out"][0]
                    accumulator[:, y0:y0 + TILE_SIZE, x0:x0 + TILE_SIZE] += output * blend
                    weights[y0:y0 + TILE_SIZE, x0:x0 + TILE_SIZE] += blend
            image_predictions[name] = (accumulator / weights.clamp_min(1e-6)).cpu().numpy()
        for item in prepared:
            output = image_predictions[item["record"]["file_name"]]
            predictions = decode_fields(np, mask_utils, core, output[:2], output[2], item["info"])
            counts = pq_counts(np, mask_utils, predictions, item["targets"])
            totals += counts
            rows.append({**item["record"], "predictions": len(predictions), **metric(counts)})
    return metric(totals), rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=MAX_STEPS)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if not 0 <= args.steps <= MAX_STEPS:
        parser.error(f"--steps must be within [0,{MAX_STEPS}]")
    if sha256(args.annotations) != ANNOTATION_SHA256:
        raise RuntimeError("V32 G2 annotation identity changed")
    if sha256(args.backbone) != BACKBONE_SHA256:
        raise RuntimeError("V32 G2 backbone identity changed")
    if sha256(args.upstream / "omnipose/core.py") != UPSTREAM_CORE_SHA256:
        raise RuntimeError("V32 G2 upstream core changed")
    warnings.filterwarnings("ignore", category=FutureWarning)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import cv2
    import numpy as np
    import torch
    from pycocotools import mask as mask_utils
    torch.manual_seed(20260913); np.random.seed(20260913)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    core = load_upstream(args.upstream)
    core.torch.set_num_threads(4)
    document = json.loads(args.annotations.read_text())
    records = selection_records(document)
    prepared = prepare_records(np, cv2, mask_utils, core, document, records, args.images)
    model = build_model(args.backbone, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    history = []
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()

    def checkpoint(step):
        metrics, rows = evaluate(np, torch, mask_utils, core, model, prepared, device)
        record = {"step": step, "metrics": metrics, "rows": rows,
                  "elapsed_seconds": time.monotonic() - started,
                  "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20}
        history.append(record)
        print(json.dumps({"step": step, **metrics,
                          "peak_allocated_mib": record["peak_allocated_mib"]}), flush=True)
        return metrics["pq"] >= TARGET_PQ

    success = checkpoint(0)
    for step in range(1, args.steps + 1):
        model.train()
        set_batchnorm_eval(torch, model)
        image, target_flow, target_distance, target_boundary, weight = sample_tile(
            np, torch, prepared, (step - 1) % SAMPLES, device
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=True):
            output = model(image)["out"]
            flow_weight = weight[:, None]
            flow_loss = torch.mean(((output[:, :2] - target_flow) / 5.0) ** 2 * flow_weight)
            distance_loss = torch.mean(((output[:, 2] - target_distance) / 5.0) ** 2 * weight)
            valid = weight > 0
            boundary_values = torch.nn.functional.binary_cross_entropy_with_logits(
                output[:, 3], target_boundary, reduction="none"
            )
            boundary_loss = boundary_values[valid].mean()
            loss = 5.0 * flow_loss + distance_loss + boundary_loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite V32 G2 loss at step {step}")
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        if step % 25 == 0:
            print(json.dumps({"train_step": step, "loss": float(loss),
                              "flow_loss": float(flow_loss),
                              "distance_loss": float(distance_loss),
                              "boundary_loss": float(boundary_loss)}), flush=True)
        if step in EVAL_STEPS:
            success = checkpoint(step)
            if success:
                break
    peak = torch.cuda.max_memory_allocated(device) / 2**20
    result = {
        "status": "complete", "selection_id": SELECTION_ID,
        "records": records, "samples": SAMPLES, "requested_steps": args.steps,
        "completed_steps": history[-1]["step"], "history": history,
        "peak_allocated_mib": peak, "memory_gate_mib": MEMORY_GATE_MIB,
        "pq_gate": TARGET_PQ, "overfit_passed": bool(success and peak < MEMORY_GATE_MIB),
        "full_training_authorized": bool(success and peak < MEMORY_GATE_MIB),
        "reads_tune": False, "reads_audit": False, "reads_test": False,
        "exports": False, "submits": False,
        "action": "proceed to V32 resource smoke" if success and peak < MEMORY_GATE_MIB
                  else "close V32 before full training; numerical overfit failed",
    }
    write_json(args.output, result)
    print(json.dumps({key: result[key] for key in (
        "completed_steps", "peak_allocated_mib", "overfit_passed",
        "full_training_authorized", "action",
    )}, indent=2))


if __name__ == "__main__":
    main()
