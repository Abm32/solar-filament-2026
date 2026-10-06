"""V33 G2: bounded balanced-foreground Omnipose FPN overfit gate.

Remote-only runner. It imports the bundled, frozen V32 target utilities, reads
exactly 16 fit annotator records and nine fit images, and runs at most 2,000
steps. It cannot read tune/audit/test, export CSV, or submit.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

try:
    import train_v32_omnipose_overfit as base
except ModuleNotFoundError:  # Local tests import through the scripts namespace.
    from scripts import train_v32_omnipose_overfit as base

MAX_STEPS = 2000
EVAL_STEPS = (0, 250, 500, 1000, 1500, 2000)
TARGET_PQ = 0.85
MEMORY_GATE_MIB = 14336
FPN_CHANNELS = 128
FOREGROUND_THRESHOLD = 0.0
HEAD_BIASES = (-2.0, 0.0, 0.0, 0.0, -2.0)


def build_model(backbone_path: Path, device):
    import torch
    from torchvision.models import resnet50

    class BalancedOmniposeFPN(torch.nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.stem = torch.nn.Sequential(
                backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            )
            self.layer1, self.layer2 = backbone.layer1, backbone.layer2
            self.layer3, self.layer4 = backbone.layer3, backbone.layer4
            widths = (256, 512, 1024, 2048)
            self.lateral = torch.nn.ModuleList([
                torch.nn.Conv2d(width, FPN_CHANNELS, 1) for width in widths
            ])
            self.smooth = torch.nn.ModuleList([
                torch.nn.Sequential(
                    torch.nn.Conv2d(FPN_CHANNELS, FPN_CHANNELS, 3, padding=1, bias=False),
                    torch.nn.GroupNorm(16, FPN_CHANNELS),
                    torch.nn.SiLU(inplace=True),
                ) for _ in widths
            ])
            self.head = torch.nn.Sequential(
                torch.nn.Conv2d(FPN_CHANNELS, FPN_CHANNELS, 3, padding=1, bias=False),
                torch.nn.GroupNorm(16, FPN_CHANNELS),
                torch.nn.SiLU(inplace=True),
                torch.nn.Conv2d(FPN_CHANNELS, 5, 1),
            )
            torch.nn.init.zeros_(self.head[-1].weight)
            torch.nn.init.zeros_(self.head[-1].bias)
            with torch.no_grad():
                self.head[-1].bias.copy_(torch.tensor(HEAD_BIASES))

        def forward(self, value):
            import torch.nn.functional as functional
            size = value.shape[-2:]
            c2 = self.layer1(self.stem(value))
            c3, c4, c5 = self.layer2(c2), None, None
            c4 = self.layer3(c3)
            c5 = self.layer4(c4)
            features = (c2, c3, c4, c5)
            pyramid = [None] * 4
            top = None
            for index in range(3, -1, -1):
                lateral = self.lateral[index](features[index])
                if top is not None:
                    lateral = lateral + functional.interpolate(
                        top, size=lateral.shape[-2:], mode="nearest"
                    )
                top = lateral
                pyramid[index] = self.smooth[index](lateral)
            output = self.head(pyramid[0])
            return functional.interpolate(output, size=size, mode="bilinear", align_corners=False)

    backbone = resnet50(weights=None)
    state = torch.load(backbone_path, map_location="cpu", weights_only=True)
    backbone.load_state_dict(state, strict=True)
    return BalancedOmniposeFPN(backbone).to(device)


def set_batchnorm_eval(torch, model) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


def balanced_bce(torch, logits, targets, valid):
    import torch.nn.functional as functional
    values = functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    positive = valid & (targets >= 0.5)
    negative = valid & ~positive
    terms = []
    if positive.any():
        terms.append(values[positive].mean())
    if negative.any():
        terms.append(values[negative].mean())
    if not terms:
        return logits.sum() * 0.0
    return sum(terms) / len(terms)


def sparse_balanced_loss(torch, output, target_flow, target_distance, target_boundary, weight):
    foreground = weight >= 1.0
    valid = weight > 0
    foreground_target = foreground.to(output.dtype)
    foreground_loss = balanced_bce(torch, output[:, 0], foreground_target, valid)
    if foreground.any():
        probabilities = output[:, 0].sigmoid()
        intersection = (probabilities * foreground_target * valid).sum()
        denominator = (probabilities * valid).sum() + foreground_target.sum()
        foreground_loss = foreground_loss + 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
        flow_error = ((output[:, 1:3] - target_flow) / 5.0) ** 2
        flow_loss = flow_error.permute(0, 2, 3, 1)[foreground].mean()
        distance_loss = (((output[:, 3] - target_distance) / 5.0) ** 2)[foreground].mean()
    else:
        flow_loss = output[:, 1:3].sum() * 0.0
        distance_loss = output[:, 3].sum() * 0.0
    boundary_loss = balanced_bce(torch, output[:, 4], target_boundary, valid)
    total = 2.0 * foreground_loss + 5.0 * flow_loss + distance_loss + boundary_loss
    return total, {
        "foreground_loss": foreground_loss,
        "flow_loss": flow_loss,
        "distance_loss": distance_loss,
        "boundary_loss": boundary_loss,
    }


def sample_tile(np, torch, prepared, sample_index, device):
    image, flow, distance, boundary, weight = base.sample_tile(
        np, torch, prepared, sample_index, device
    )
    return image, flow, distance, boundary, weight


def decode_fields(np, mask_utils, core, foreground_logit, flow, distance):
    foreground = foreground_logit >= FOREGROUND_THRESHOLD
    if not np.any(foreground):
        return [], None
    positive_distance = distance[foreground]
    positive_distance = positive_distance[np.isfinite(positive_distance)]
    mean_distance = max(0.0, float(positive_distance.mean())) if positive_distance.size else 0.0
    niter = max(1, int(2 * (foreground.ndim + 1) * mean_distance))
    try:
        output = core.compute_masks(
            flow, distance, bd=np.zeros_like(distance), iscell=foreground,
            niter=niter, flow_threshold=0, omni=True, use_gpu=False,
            device=core.torch_CPU, nclasses=3, dim=2, min_size=base.MIN_AREA,
            cluster=False, suppress=True, diam_threshold=12.0, eps=2 ** 0.5,
            affinity_seg=False, hdbscan=False,
        )[0]
    except ValueError as error:
        if "contains NaN" not in str(error):
            raise
        return [], "upstream_decoder_nan"
    predictions = []
    for label in range(1, int(output.max()) + 1):
        value = output == label
        if int(value.sum()) >= base.MIN_AREA:
            predictions.append(mask_utils.encode(np.asfortranarray(value.astype(np.uint8))))
    return predictions, None


def evaluate(np, torch, mask_utils, core, model, prepared, device):
    model.eval()
    blend = base.hann_window(torch, device)
    totals = np.zeros(4, dtype=np.float64)
    rows, finite_values, foreground_pixels, decoder_errors = [], True, 0, 0
    total_pixels = 0
    channel_min = np.full(5, np.inf)
    channel_max = np.full(5, -np.inf)
    with torch.inference_mode():
        predictions_by_name = {}
        for name in sorted({item["record"]["file_name"] for item in prepared}):
            source = next(item["image"] for item in prepared if item["record"]["file_name"] == name)
            accumulator = torch.zeros((5, 2048, 2048), device=device)
            weights = torch.zeros((2048, 2048), device=device)
            for y0 in base.TILE_STARTS:
                for x0 in base.TILE_STARTS:
                    tile = base.normalize_tile(np, source[y0:y0 + base.TILE_SIZE, x0:x0 + base.TILE_SIZE])
                    tensor = torch.from_numpy(tile).to(device).unsqueeze(0).repeat(3, 1, 1)
                    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[:, None, None]
                    std = torch.tensor([0.229, 0.224, 0.225], device=device)[:, None, None]
                    output = model(((tensor - mean) / std).unsqueeze(0))[0]
                    accumulator[:, y0:y0 + base.TILE_SIZE, x0:x0 + base.TILE_SIZE] += output * blend
                    weights[y0:y0 + base.TILE_SIZE, x0:x0 + base.TILE_SIZE] += blend
            output = (accumulator / weights.clamp_min(1e-6)).cpu().numpy()
            finite_values = finite_values and bool(np.isfinite(output).all())
            channel_min = np.minimum(channel_min, np.nanmin(output, axis=(1, 2)))
            channel_max = np.maximum(channel_max, np.nanmax(output, axis=(1, 2)))
            foreground_pixels += int((output[0] >= FOREGROUND_THRESHOLD).sum())
            total_pixels += output.shape[1] * output.shape[2]
            predictions_by_name[name] = output
        for item in prepared:
            output = predictions_by_name[item["record"]["file_name"]]
            predictions, decoder_error = decode_fields(
                np, mask_utils, core, output[0], output[1:3], output[3]
            )
            decoder_errors += int(decoder_error is not None)
            counts = base.pq_counts(np, mask_utils, predictions, item["targets"])
            totals += counts
            rows.append({**item["record"], "predictions": len(predictions),
                         "decoder_error": decoder_error, **base.metric(counts)})
    diagnostics = {
        "finite_outputs": bool(finite_values),
        "decoder_errors": decoder_errors,
        "predicted_foreground_fraction": foreground_pixels / total_pixels,
        "channel_min": channel_min.tolist(), "channel_max": channel_max.tolist(),
    }
    return base.metric(totals), rows, diagnostics


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
    if base.sha256(args.annotations) != base.ANNOTATION_SHA256:
        raise RuntimeError("V33 annotation identity changed")
    if base.sha256(args.backbone) != base.BACKBONE_SHA256:
        raise RuntimeError("V33 backbone identity changed")
    if base.sha256(args.upstream / "omnipose/core.py") != base.UPSTREAM_CORE_SHA256:
        raise RuntimeError("V33 upstream core changed")

    warnings.filterwarnings("ignore", category=FutureWarning)
    import cv2
    import numpy as np
    import torch
    from pycocotools import mask as mask_utils

    torch.manual_seed(20260913); np.random.seed(20260913)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    core = base.load_upstream(args.upstream)
    core.torch.set_num_threads(4)
    document = json.loads(args.annotations.read_text())
    records = base.selection_records(document)
    prepared = base.prepare_records(np, cv2, mask_utils, core, document, records, args.images)
    model = build_model(args.backbone, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    history, loss_history = [], []
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()

    def checkpoint(step):
        metrics, rows, diagnostics = evaluate(np, torch, mask_utils, core, model, prepared, device)
        record = {"step": step, "metrics": metrics, "diagnostics": diagnostics, "rows": rows,
                  "elapsed_seconds": time.monotonic() - started,
                  "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20}
        history.append(record)
        print(json.dumps({"step": step, **metrics, **diagnostics,
                          "peak_allocated_mib": record["peak_allocated_mib"]}), flush=True)
        return (metrics["pq"] >= TARGET_PQ and diagnostics["finite_outputs"]
                and diagnostics["decoder_errors"] == 0)

    success = checkpoint(0)
    for step in range(1, args.steps + 1):
        model.train(); set_batchnorm_eval(torch, model)
        batch = sample_tile(np, torch, prepared, (step - 1) % base.SAMPLES, device)
        image, target_flow, target_distance, target_boundary, weight = batch
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=True):
            output = model(image)
            loss, terms = sparse_balanced_loss(
                torch, output, target_flow, target_distance, target_boundary, weight
            )
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite V33 loss at step {step}")
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        if step % 25 == 0:
            row = {"train_step": step, "loss": float(loss),
                   **{name: float(value) for name, value in terms.items()}}
            loss_history.append(row); print(json.dumps(row), flush=True)
        if step in EVAL_STEPS:
            success = checkpoint(step)
            if success:
                break

    peak = torch.cuda.max_memory_allocated(device) / 2**20
    authorized = bool(success and peak < MEMORY_GATE_MIB)
    result = {
        "status": "complete", "experiment": "v33_balanced_omnipose_fpn_g2",
        "selection_id": base.SELECTION_ID, "records": records,
        "samples": base.SAMPLES, "requested_steps": args.steps,
        "completed_steps": history[-1]["step"], "history": history,
        "loss_history": loss_history, "peak_allocated_mib": peak,
        "memory_gate_mib": MEMORY_GATE_MIB, "pq_gate": TARGET_PQ,
        "overfit_passed": authorized, "full_training_design_authorized": authorized,
        "reads_tune": False, "reads_audit": False, "reads_test": False,
        "exports": False, "submits": False,
        "action": "freeze a separate V33 full-training design" if authorized
                  else "close V33 before full training; numerical overfit failed",
    }
    base.write_json(args.output, result)
    print(json.dumps({key: result[key] for key in (
        "completed_steps", "peak_allocated_mib", "overfit_passed",
        "full_training_design_authorized", "action",
    )}, indent=2))


if __name__ == "__main__":
    main()
