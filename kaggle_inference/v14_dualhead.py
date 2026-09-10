"""V14: dual-head refiner attacking BOTH PQ axes in one run.

Design and arithmetic: docs/v14_design_2026-09-07.md, docs/ceiling_analysis_2026-09-07.md.

WHY THIS EXISTS. Three submissions score 0.39 (V11 x2, V13). The local->public
offset is a stable -0.050, so public 0.55 needs local tune PQ ~= 0.60, and that
requires recall >= 0.85 TOGETHER WITH mean matched IoU >= 0.75. We are at recall
0.6049 / meanIoU 0.6855. Every experiment E015-E024 moved one axis by ~0.005.

Two measured, untouched pools:
  * 278 grouped proposals killed by the fixed 0.40 confidence cutoff. Those already
    have IoU>0.5 geometry; detector confidence is simply a poor proxy for mask
    quality. E021 bounded only the CHEAP-FEATURE fix (summary geometry, AUC 0.7252);
    a head reading mask pixels is untested.
  * 84 GT with support coverage >0.5 but raw mask IoU <=0.5 -- object found, mask
    fails.

And one measured NON-lever, checked for free before spending GPU: `refined_free`
(no proposal clamp) ties `refined_constrained` in both historical sweeps
(0.418946 vs 0.418869; 0.420141 vs 0.420230). Removing the dilate(proposal) clamp
is worth ~0 on its own, so this run changes the HEAD and the OBJECTIVE, and treats
unclamping as a necessary-but-insufficient detail.

MECHANISM. One shared ResNet34 encoder (V6/V11 used ResNet18) with two heads:
  * mask head -> 256x256 logits, deep supervision at 1/2 and 1/4 scale. Attacks
    the IoU axis.
  * IoU head  -> scalar, predicts the IoU of its OWN mask against GT. Replaces the
    fixed 0.40 confidence cutoff, attacking the recall axis. Detached from the mask
    branch so IoU supervision cannot corrupt mask gradients.

ARMS. Only the mask loss differs; cohort, sample order, epochs, LR, seed are
identical and asserted:
  * arm_a: 0.5 BCE + 0.5 Dice          (V11's loss, the control objective)
  * arm_b: 0.5 BCE + Focal-Tversky     (beta>alpha, FN-weighted; smallest-30%
           instances sit at mean raw IoU 0.518, one eroded pixel from FN)

PROTOCOL, declared before any V14 number is observed:
  * 21 predeclared candidates per arm: mask_threshold {0.40,0.45,0.50} x iou_gate
    {0.20..0.50 step 0.05}. No expansion after seeing scores.
  * Selection is tune PQ ONLY. Audit is reused screening evidence and never selects.
  * SUBMIT_BAR authorises exporting a CSV: tune >= V13's 0.441777 AND audit >=
    E015's 0.450130.
  * PROMOTION_GATE authorises replacing the baseline: tune >= V13 + 0.010 AND audit
    >= V13's audit. The 0.010 is deliberately 10x V13's delta because V13 proved
    +0.004 local is invisible at two public decimals.
  * The frozen control arm runs FIRST and must reproduce E015's exact tune/audit
    counts. Nothing downstream is trustworthy otherwise.

REPORTING. Every candidate reports recall, mean matched IoU and FP as THREE
SEPARATE numbers. Prior runs reported aggregate PQ only, which hid which axis moved
and is the reason four submissions looked different locally but identical publicly.

Nothing here submits. Export is gated on SUBMIT_BAR; upload stays manual.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

CONTROL = "v14_frozen_control"
ARMS = ("arm_a", "arm_b")
ARM_LOSS = {"arm_a": "bce_dice", "arm_b": "bce_focal_tversky"}
V11_RUN_IDENTITY = "7c45acf26f210acce00f1bff581288bc2eb6121485237c7c028588d1de69164c"
V11_SOURCE_RUN = "abhimanyu122/solar-filament-v11-refiner"
CROP_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
SEED = 42
IOU_LOSS_WEIGHT = 0.2
TVERSKY_ALPHA = 0.3
TVERSKY_BETA = 0.7
TVERSKY_GAMMA = 0.75
DEEP_SUPERVISION_WEIGHTS = (1.0, 0.5, 0.25)
POLICY = "flip_consensus"
MATCH_IOU = 0.5
MIN_AREA = 20
COUNTS = ("iou_sum", "tp", "fp", "fn")
# Predeclared inference grid. mask_threshold binarises the mask logits; iou_gate is
# the minimum PREDICTED IoU to keep an instance, replacing detector confidence.
MASK_THRESHOLDS = (0.40, 0.45, 0.50)
IOU_GATES = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
# Accepted E015 control (== public 0.38); identical to v12/v13 pins.
CONTROL_REFERENCE = dict(
    tune=[426.5473399071961, 626, 211, 508],
    audit=[182.75261599994468, 265, 88, 194],
)
# Measured baselines this run is judged against.
BASELINES = dict(
    e015_control=dict(tune_pq=0.4328232774299302, audit_pq=0.4501295960589769,
                      public_score=0.38),
    v13_compose=dict(tune_pq=0.4417771081401778, audit_pq=0.4646438167892603,
                     public_score=0.39, recall=0.6049, mean_matched_iou=0.6855, fp=309),
)
# E021's best cheap ranker, so the IoU head is compared against a real baseline.
CONSENSUS_SCORE_AUC = 0.7252
SUBMIT_BAR = dict(min_tune_pq=0.4417771081401778, min_audit_pq=0.4501295960589769,
                  basis="V13 tune (best measured) and E015 audit floor")
PROMOTION_GATE = dict(required_tune_delta_vs_v13=0.010, require_audit_at_least_v13=True,
                      basis="10x V13's delta: V13 proved +0.004 local is invisible "
                            "at two public decimals")
CHECKPOINT_SHA256 = None
MODULE_SHA256 = None
RAW_CACHE_SHA256 = None


def pq(counts):
    iou, tp, fp, fn = counts
    denominator = tp + 0.5 * (fp + fn)
    return float(iou / denominator) if denominator else 0.0


def axis_metrics(counts):
    """Recall, mean matched IoU and FP as SEPARATE numbers.

    The whole point of V14 is that PQ alone hides which axis moved. Reaching local
    0.60 needs recall >= 0.85 AND mean IoU >= 0.75; a run that trades one for the
    other looks flat in PQ and must be visible here.
    """
    iou_sum, tp, fp, fn = counts
    n_gt = tp + fn
    return dict(recall=float(tp / n_gt) if n_gt else 0.0,
                mean_matched_iou=float(iou_sum / tp) if tp else 0.0,
                fp=int(fp), tp=int(tp), fn=int(fn), n_gt=int(n_gt), pq=pq(counts))


def candidate_key(mask_threshold, iou_gate):
    return f"m{mask_threshold:.2f}_g{iou_gate:.2f}"


def declared_candidates():
    """All 21 candidates per arm, enumerated up front so the grid cannot widen."""
    return tuple(candidate_key(t, g) for t in MASK_THRESHOLDS for g in IOU_GATES)


def verify_grid():
    if len(declared_candidates()) != len(MASK_THRESHOLDS) * len(IOU_GATES) != 21:
        raise ValueError("Predeclared grid must be exactly 3 thresholds x 7 gates")
    if len(set(declared_candidates())) != 21:
        raise ValueError("Predeclared candidates must be unique")
    if not all(0 < t < 1 for t in MASK_THRESHOLDS):
        raise ValueError("Mask thresholds must be probabilities")
    if not all(0 < g < 1 for g in IOU_GATES):
        raise ValueError("IoU gates must be probabilities")
    return True


def focal_tversky_loss(logits, target, alpha=TVERSKY_ALPHA, beta=TVERSKY_BETA,
                       gamma=TVERSKY_GAMMA, eps=1e-6):
    """FN-weighted Tversky with a focal exponent.

    beta > alpha penalises false negatives harder than false positives, which is
    what the bottleneck decomposition asks for: the smallest 30% of instances sit at
    mean raw IoU 0.518, barely over the 0.5 matching bar, so under-segmentation --
    not over-segmentation -- is what converts them to FN.

    gamma < 1 makes the gradient larger for already-poor predictions.
    """
    import torch
    if not 0 < alpha < 1 or not 0 < beta < 1:
        raise ValueError("Tversky alpha/beta must be in (0, 1)")
    if abs(alpha + beta - 1.0) > 1e-9:
        raise ValueError("Tversky alpha + beta must equal 1 for a calibrated index")
    if beta <= alpha:
        raise ValueError("V14 arm_b requires beta > alpha (FN-weighted)")
    probability = torch.sigmoid(logits)
    dims = tuple(range(1, probability.dim()))
    true_positive = (probability * target).sum(dims)
    false_negative = ((1 - probability) * target).sum(dims)
    false_positive = (probability * (1 - target)).sum(dims)
    index = (true_positive + eps) / (
        true_positive + alpha * false_positive + beta * false_negative + eps)
    return torch.pow(1 - index, gamma).mean()


def mask_loss(arm, logits, target, base):
    """arm_a reproduces V11's objective exactly; arm_b is the FN-weighted variant."""
    import torch
    if arm not in ARMS:
        raise ValueError(f"Unknown arm {arm}")
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    if arm == "arm_a":
        return 0.5 * bce + 0.5 * base.soft_dice_loss(logits, target)
    return 0.5 * bce + focal_tversky_loss(logits, target)


def deep_supervision_loss(arm, outputs, target, base, weights=DEEP_SUPERVISION_WEIGHTS):
    """Full-resolution loss plus downsampled auxiliary terms.

    Auxiliary heads at 1/2 and 1/4 give the decoder gradient signal about coarse
    shape while the full-resolution head handles the boundary, which is where our
    IoU is actually lost.
    """
    import torch
    if len(outputs) != len(weights):
        raise ValueError("One output per deep-supervision scale is required")
    total = 0.0
    for logits, weight in zip(outputs, weights):
        if logits.shape[-2:] != target.shape[-2:]:
            scaled = torch.nn.functional.interpolate(
                target, size=logits.shape[-2:], mode="area")
            scaled = (scaled > 0.5).to(target.dtype)
        else:
            scaled = target
        total = total + weight * mask_loss(arm, logits, scaled, base)
    return total / sum(weights)


def observed_iou(logits, target, threshold=0.5, eps=1e-6):
    """Per-sample IoU of the model's OWN thresholded mask vs GT: the IoU head target."""
    import torch
    with torch.no_grad():
        predicted = (torch.sigmoid(logits) >= threshold).to(target.dtype)
        dims = tuple(range(1, predicted.dim()))
        intersection = (predicted * target).sum(dims)
        union = predicted.sum(dims) + target.sum(dims) - intersection
        return (intersection + eps) / (union + eps)


def iou_head_loss(predicted_iou, actual_iou, weight=IOU_LOSS_WEIGHT):
    """MSE on the IoU scalar, down-weighted so it never dominates the mask objective."""
    import torch
    if not 0 < weight <= 1:
        raise ValueError("IoU loss weight must be in (0, 1]")
    return weight * torch.nn.functional.mse_loss(predicted_iou, actual_iou)


def ranking_auc(scores, labels):
    """AUC of a scalar ranker; compared against E021's consensus_score 0.7252.

    Rank-based Mann-Whitney form with explicit tie handling, so a head that outputs
    a constant scores 0.5 rather than something flattering.
    """
    scores, labels = np.asarray(scores, dtype=float), np.asarray(labels).astype(bool)
    if scores.shape != labels.shape:
        raise ValueError("Scores and labels must align")
    positive, negative = int(labels.sum()), int((~labels).sum())
    if not positive or not negative:
        return float("nan")
    order = scores.argsort(kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)
    # Average ranks within tie groups so constant scores give exactly 0.5.
    unique, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    if (counts > 1).any():
        sums = np.zeros(len(unique))
        np.add.at(sums, inverse, ranks)
        ranks = (sums / counts)[inverse]
    return float((ranks[labels].sum() - positive * (positive + 1) / 2)
                 / (positive * negative))


def calibration_error(predicted, actual, bins=10):
    """Mean absolute gap between predicted and actual IoU, binned on prediction."""
    predicted, actual = np.asarray(predicted, float), np.asarray(actual, float)
    if predicted.shape != actual.shape:
        raise ValueError("Calibration inputs must align")
    if not len(predicted):
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(predicted, edges[1:-1]), 0, bins - 1)
    total, weight = 0.0, 0
    for b in range(bins):
        mask = index == b
        if mask.any():
            total += abs(predicted[mask].mean() - actual[mask].mean()) * mask.sum()
            weight += int(mask.sum())
    return float(total / weight) if weight else float("nan")


def verify_control(counts, reference=CONTROL_REFERENCE):
    """The frozen control must reproduce E015's accepted counts before any claim."""
    for group in ("tune", "audit"):
        expected, actual = reference[group], counts[group]
        if len(expected) != len(actual):
            raise ValueError("Control counts have unexpected shape")
        if abs(expected[0] - actual[0]) > 1e-6 or [int(v) for v in actual[1:]] != expected[1:]:
            raise ValueError(f"Control {group} counts differ from the frozen E015 "
                             f"reference: {actual} vs {expected}")
    return True


def select_candidate(results):
    """Tune-PQ-only selection across both arms over the predeclared grid."""
    declared = set(declared_candidates())
    for key in results:
        arm, candidate = key.split("/", 1)
        if arm not in ARMS or candidate not in declared:
            raise ValueError(f"Result outside the predeclared arm/grid: {key}")
    if not results:
        raise ValueError("No candidates to select from")
    return max(sorted(results), key=lambda key: pq(results[key]["tune"]))


def evaluate_bars(selected, results, baselines=BASELINES, submit_bar=SUBMIT_BAR,
                  gate=PROMOTION_GATE):
    """Both predeclared decisions plus the axis breakdown that PQ alone hides."""
    tune_counts, audit_counts = results[selected]["tune"], results[selected]["audit"]
    tune, audit = pq(tune_counts), pq(audit_counts)
    v13 = baselines["v13_compose"]
    authorised = tune >= submit_bar["min_tune_pq"] and audit >= submit_bar["min_audit_pq"]
    promotable = tune - v13["tune_pq"] >= gate["required_tune_delta_vs_v13"]
    if gate["require_audit_at_least_v13"]:
        promotable = promotable and audit >= v13["audit_pq"]
    axes = axis_metrics(tune_counts)
    return dict(
        selected=selected, selected_tune_pq=tune, selected_audit_pq=audit,
        tune_axes=axes, audit_axes=axis_metrics(audit_counts),
        axis_deltas_vs_v13=dict(recall=axes["recall"] - v13["recall"],
                                mean_matched_iou=axes["mean_matched_iou"] - v13["mean_matched_iou"],
                                fp=axes["fp"] - v13["fp"]),
        deltas_vs_baselines={name: dict(tune=tune - value["tune_pq"],
                                        audit=audit - value["audit_pq"])
                             for name, value in baselines.items()},
        submit_bar=submit_bar, export_authorized=bool(authorised),
        promotion_gate=gate, promotable=bool(promotable),
        recommendation=(
            "promote and submit" if promotable else
            "record only; do NOT submit (V13 proved a small local gain buys nothing)"
            if authorised else
            "below the best measured candidate; do not submit"))


def build_dual_head(base, warm_start=None):
    """ResNet34 encoder + mask head (deep supervision) + IoU head.

    Deliberately a separate class from base.CropUNet so the frozen accepted
    pipeline (V6/V11/V13, ResNet18) is never modified by this experiment. The 4th
    input channel is the proposal HINT: unlike refined_constrained it is not a hard
    boundary, so the head can extend past a truncated proposal.
    """
    import torch
    from torch import nn
    from torchvision.models import resnet34, ResNet34_Weights

    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

        def forward(self, x):
            return self.block(x)

    class DualHeadRefiner(nn.Module):
        def __init__(self, pretrained=True):
            super().__init__()
            encoder = resnet34(weights=ResNet34_Weights.DEFAULT if pretrained else None)
            stem = encoder.conv1
            # Four input channels: RGB-replicated grayscale + proposal hint. The hint
            # weights start at zero so the model begins numerically identical to a
            # 3-channel model and must LEARN to use the hint.
            expanded = nn.Conv2d(4, stem.out_channels, stem.kernel_size,
                                 stem.stride, stem.padding, bias=False)
            with torch.no_grad():
                expanded.weight[:, :3] = stem.weight
                expanded.weight[:, 3:] = 0.0
            self.stem = nn.Sequential(expanded, encoder.bn1, encoder.relu)
            self.pool = encoder.maxpool
            self.layer1, self.layer2 = encoder.layer1, encoder.layer2
            self.layer3, self.layer4 = encoder.layer3, encoder.layer4
            self.up4 = nn.ConvTranspose2d(512, 256, 2, 2)
            self.dec4 = ConvBlock(512, 256)
            self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
            self.dec3 = ConvBlock(256, 128)
            self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
            self.dec2 = ConvBlock(128, 64)
            self.up1 = nn.ConvTranspose2d(64, 32, 2, 2)
            self.dec1 = ConvBlock(96, 32)
            self.up0 = nn.ConvTranspose2d(32, 16, 2, 2)
            self.dec0 = ConvBlock(16, 16)
            self.head = nn.Conv2d(16, 1, 1)
            self.aux_half = nn.Conv2d(32, 1, 1)
            self.aux_quarter = nn.Conv2d(64, 1, 1)
            self.iou_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Linear(512, 128), nn.ReLU(inplace=True),
                nn.Dropout(0.1), nn.Linear(128, 1))
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406, 0.0]).view(1, 4, 1, 1))
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225, 1.0]).view(1, 4, 1, 1))

        def forward(self, image):
            if image.shape[1] != 4:
                raise ValueError("V14 expects 3 grayscale channels plus a proposal hint")
            image = (image - self.mean) / self.std
            x0 = self.stem(image)
            x1 = self.layer1(self.pool(x0))
            x2 = self.layer2(x1)
            x3 = self.layer3(x2)
            x4 = self.layer4(x3)
            d4 = self.dec4(torch.cat([self.up4(x4), x3], 1))
            d3 = self.dec3(torch.cat([self.up3(d4), x2], 1))
            d2 = self.dec2(torch.cat([self.up2(d3), x1], 1))
            d1 = self.dec1(torch.cat([self.up1(d2), x0], 1))
            d0 = self.dec0(self.up0(d1))
            # IoU head reads the deepest encoder features, detached so its gradient
            # cannot reshape the mask representation.
            predicted_iou = self.iou_head(x4.detach()).squeeze(1)
            return (self.head(d0), self.aux_half(d1), self.aux_quarter(d2)), predicted_iou

    model = DualHeadRefiner(pretrained=warm_start is None)
    if warm_start is not None:
        loaded, skipped = load_compatible_weights(model, warm_start)
        print(json.dumps(dict(warm_start_tensors_loaded=loaded,
                              warm_start_tensors_skipped=skipped)), flush=True)
    return model


def load_compatible_weights(model, checkpoint_path):
    """Load only shape-matching tensors, and say how many were skipped.

    V11's checkpoint is a ResNet18 U-Net; V14 is ResNet34 with two heads, so most
    tensors cannot transfer. Reporting the counts keeps the warm start honest
    instead of implying full continuation.
    """
    import torch
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    source = payload.get("model", payload)
    target = model.state_dict()
    transfer = {key: value for key, value in source.items()
                if key in target and target[key].shape == value.shape}
    target.update(transfer)
    model.load_state_dict(target)
    return len(transfer), len(target) - len(transfer)


def run_experiment():
    """Entry point; the heavy lifting lives in v14_runtime for testability."""
    import sys
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working"))
    parser.add_argument("--limit", type=int, default=0,
                        help="Smoke only: cap training files and shrink the grid.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("limit must be nonnegative")
    if not MODULE_SHA256 or not CHECKPOINT_SHA256 or not RAW_CACHE_SHA256:
        raise ValueError("Run scripts/prepare_v14_dualhead.py to stage a pinned run")
    verify_grid()
    runner = sys.modules["v14_runtime"]
    runner.MODULE_SHA256 = MODULE_SHA256
    runner.CHECKPOINT_SHA256 = CHECKPOINT_SHA256
    runner.RAW_CACHE_SHA256 = RAW_CACHE_SHA256
    runner.main(args)


if __name__ == "__main__":
    run_experiment()
