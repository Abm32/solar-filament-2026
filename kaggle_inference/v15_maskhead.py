"""V15: retest the mask-capacity hypothesis with an encoder that is actually pretrained.

Design: docs/v15_design_2026-09-07.md. Diagnosis it responds to: E025 in
docs/experiments.md.

WHY THIS EXISTS. V14 measured three of its own defects, not its hypotheses:

  1. `DualHeadRefiner(pretrained=warm_start is None)` -- supplying a warm start
     SILENTLY DISABLED ImageNet init, and the ResNet18->ResNet34 warm start moved
     only 131 of 298 tensors. The encoder was mostly RANDOM for 8 epochs, which is
     why mean matched IoU regressed to 0.6604 against V13's 0.6855.
  2. The IoU head trained only on positive-matched cohort records (min_iou=0.30), so
     it emitted a near-constant ~0.62-0.66 and every declared gate 0.20..0.50
     filtered 0 of 1773 proposals. FP went 309 -> 2051 and tune PQ collapsed to
     0.2449.
  3. The head diagnostic zipped predicted scores in raw proposal order against
     actual IoU in sorted-and-filtered order, so the reported AUC 0.5087 measured
     the misalignment rather than the head.

Because of (1) the expensive question -- can more mask capacity move mean matched
IoU off 0.686? -- was never actually asked. ceiling_analysis item 1 is UNTESTED,
not falsified, and E-F/Focal-Tversky is not retired either: arm_b lost on every
axis, but both arms sat on the same crippled encoder.

THE CENTRAL DESIGN DECISION. V14 changed the mask head AND replaced detector
confidence with a learned gate in one run; when the gate failed, the mask change
became unmeasurable. V15 changes ONLY the refiner and keeps V13's exact measured
selection: E-A box iou=0.3 proposals, then the frozen threshold x confidence grid in
`refined_constrained` mode. Confidence gating at c0.30 is what held V13 to FP=309,
so V15 STRUCTURALLY CANNOT reproduce V14's FP explosion.

The learned IoU score is still computed and reported, as a KEEP-FRACTION frontier,
but it DOES NOT SELECT. That keeps the diagnostic value of V14's head without
letting an uncalibrated scalar drive predictions again.

WHAT IS FIXED, EACH WITH AN ASSERTION:
  * ImageNet init unconditionally; `build_mask_head` asserts pretrained=True and
    logs transferred/skipped warm-start tensor counts.
  * The scoring cohort adds unmatched proposals (true IoU ~0); asserted >=20%
    negatives or the run raises rather than shipping another constant head.
  * Head rows are built from surviving proposal INDICES so predicted and actual
    refer to the same instance; a permutation check must collapse AUC to ~0.5.
  * The learned score is reported only as a keep-fraction frontier, never an
    absolute cut.

PROTOCOL, declared before any V15 number is observed:
  * 42 predeclared candidates (6 thresholds x 7 confidences), bound at runtime to
    detector_tta.LOCAL_* exactly as V13 does. Tune-PQ-only selection.
  * SUBMIT_BAR (authorises exporting a CSV): tune >= V13's 0.441777 AND audit >=
    E015's 0.450130.
  * PROMOTION_GATE (authorises spending a submission): tune >= V13 + 0.010 AND
    audit >= V13's 0.464644. E024 proved +0.0038 local is invisible at two decimals.
  * The frozen control runs FIRST, must reproduce E015's counts, and its CSV must
    hash to the accepted 0.38 file byte-for-byte.

Nothing here submits. Export is gated on SUBMIT_BAR; upload stays manual.

CPU-importable helpers only at module level; torch loads inside runtime paths so the
local suite can import this without a GPU.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

CONTROL = "v15_frozen_control"
ARM = "v15_maskhead"
# V11 checkpoint used as the warm start, and the run that produced it.
V11_RUN_IDENTITY = "7c45acf26f210acce00f1bff581288bc2eb6121485237c7c028588d1de69164c"
V11_SOURCE_RUN = "abhimanyu122/solar-filament-v11-refiner"
# E-A's winning cell and the frozen control cell, copied from V13 verbatim.
CELL = dict(name="ea_iou030", retina_masks=True, agnostic_nms=True, iou=0.3)
CONTROL_CELL = dict(name="frozen_control", retina_masks=True, agnostic_nms=True, iou=0.7)
DETECTOR_RESOLUTION = 1024
DETECTOR_CONFIDENCE_FLOOR = 0.1
POLICY = "flip_consensus"
MATCH_IOU = 0.5
MODE = "refined_constrained"
MIN_AREA = 20
COUNTS = ("iou_sum", "tp", "fp", "fn")
# V13's frozen selection grid. NOT widened: the refiner is the only new variable.
THRESHOLDS = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60)
CONFIDENCES = (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55)
# Training. One arm only, so there is no arm comparison to confound.
CROP_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
SEED = 42
IOU_LOSS_WEIGHT = 0.2
DEEP_SUPERVISION_WEIGHTS = (1.0, 0.5, 0.25)
# Scoring-cohort health: the V14 failure mode is a head that never saw a bad
# proposal. Below this fraction of near-zero-IoU samples the run must abort.
MIN_NEGATIVE_FRACTION = 0.20
NEGATIVE_IOU_CEILING = 0.10
# Keep fractions for the reported frontier. Fractions, never absolute score cuts.
KEEP_FRACTIONS = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00)
# Accepted E015 control (== public 0.38); identical pin to V12/V13/V14.
CONTROL_REFERENCE = dict(
    candidate="t0.40_c0.40",
    tune=[426.5473399071962, 626, 211, 508],
    audit=[182.75261599994462, 265, 88, 194],
)
ACCEPTED_CONTROL_CSV_SHA256 = (
    "bf11690e9bcce73b0c627096cc1198d9fb0916650f7fa4b3f08f1167bb76bde9")
# Measured baselines V15 is judged against.
BASELINES = dict(
    e015_control=dict(tune_pq=0.4328232774299302, audit_pq=0.4501295960589769,
                      public_score=0.38),
    v13_compose=dict(tune_pq=0.4417771081401778, audit_pq=0.4646438167892603,
                     public_score=0.39, recall=0.6049, mean_matched_iou=0.6855, fp=309),
)
SUBMIT_BAR = dict(min_tune_pq=0.4417771081401778, min_audit_pq=0.4501295960589769,
                  basis="V13 tune (best measured) and E015 audit floor")
PROMOTION_GATE = dict(required_tune_delta_vs_v13=0.010, require_audit_at_least_v13=True,
                      basis="E024: +0.0038 local moved nothing at two public decimals")
# E021's best cheap ranker, so the head has a real comparison point.
CONSENSUS_SCORE_AUC = 0.7252
CHECKPOINT_SHA256 = None
MODULE_SHA256 = None
RAW_CACHE_SHA256 = None


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    """Atomic publication so a killed run never leaves a half-written manifest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def pq(counts):
    iou, tp, fp, fn = counts
    denominator = tp + 0.5 * (fp + fn)
    return float(iou / denominator) if denominator else 0.0


def axis_metrics(counts):
    """Recall, mean matched IoU and FP as SEPARATE numbers.

    PQ alone hides which axis moved, which is how four submissions looked different
    locally and identical publicly. V15's whole question is the meanIoU axis.
    """
    iou_sum, tp, fp, fn = counts
    n_gt = tp + fn
    return dict(recall=float(tp / n_gt) if n_gt else 0.0,
                mean_matched_iou=float(iou_sum / tp) if tp else 0.0,
                fp=int(fp), tp=int(tp), fn=int(fn), n_gt=int(n_gt), pq=pq(counts))


def candidate_key(threshold, confidence):
    return f"t{threshold:.2f}_c{confidence:.2f}"


def declared_candidates():
    """All 42 candidates, enumerated up front so the grid cannot widen after scores."""
    return tuple(candidate_key(t, c) for t in THRESHOLDS for c in CONFIDENCES)


def verify_grid():
    declared = declared_candidates()
    if len(declared) != len(THRESHOLDS) * len(CONFIDENCES) != 42:
        raise ValueError("Predeclared grid must be exactly 6 thresholds x 7 confidences")
    if len(set(declared)) != 42:
        raise ValueError("Predeclared candidates must be unique")
    if not all(0 < t < 1 for t in THRESHOLDS):
        raise ValueError("Thresholds must be probabilities")
    if not all(0 < c < 1 for c in CONFIDENCES):
        raise ValueError("Confidences must be probabilities")
    if CONTROL_REFERENCE["candidate"] not in declared:
        raise ValueError("The control candidate must be inside the declared grid")
    return True


def verify_grid_matches_frozen_constants(tta):
    """Bind the grid to the frozen sweep constants, exactly as V13 does.

    A drifting local grid would silently change what "the same selection" means, and
    the whole V15 claim is that selection is unchanged from V13.
    """
    if tuple(tta.LOCAL_THRESHOLDS) != THRESHOLDS:
        raise ValueError("Thresholds drifted from the frozen sweep constants")
    if tuple(tta.LOCAL_CONFIDENCES) != CONFIDENCES:
        raise ValueError("Confidences drifted from the frozen sweep constants")
    return True


def cache_id(name, split, cell_name):
    """Cell-scoped cache identity: iou=0.3 proposals must never collide with iou=0.7."""
    if Path(name).name != name or split not in ("train", "test"):
        raise ValueError("Cache image must be a basename in train or test")
    token = (f"{cell_name}|{split}|{name}|imgsz{DETECTOR_RESOLUTION}"
             f"|conf{DETECTOR_CONFIDENCE_FLOOR}")
    return hashlib.sha256(token.encode()).hexdigest()[:16] + ".json.gz"


def focal_tversky_disabled_note():
    """V15 trains arm_a's objective only; E-F remains open but is not tested here.

    V14's arm comparison was confounded by the random encoder, so re-running it would
    cost a second arm to answer a question that only matters if capacity helps at
    all. Establish capacity first, on one arm, then revisit the loss.
    """
    return ("Focal-Tversky is NOT retired and NOT tested here: V14's arm_b lost on "
            "every axis but both arms had a mostly-random encoder. V15 spends its "
            "budget establishing whether capacity moves meanIoU with a correct "
            "encoder; the loss comparison is a later, cheaper follow-up.")


def observed_iou(logits, target, threshold=0.5, eps=1e-6):
    """Per-sample IoU of the model's OWN thresholded mask vs GT: the IoU head target."""
    import torch
    with torch.no_grad():
        predicted = (torch.sigmoid(logits) >= threshold).to(target.dtype)
        dims = tuple(range(1, predicted.dim()))
        intersection = (predicted * target).sum(dims)
        union = predicted.sum(dims) + target.sum(dims) - intersection
        return (intersection + eps) / (union + eps)


def mask_loss(logits, target, base):
    """V11's proven objective: 0.5 BCE + 0.5 soft Dice. Unchanged, deliberately."""
    import torch
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    return 0.5 * bce + 0.5 * base.soft_dice_loss(logits, target)


def deep_supervision_loss(outputs, target, base, weights=DEEP_SUPERVISION_WEIGHTS):
    """Full-resolution loss plus 1/2 and 1/4 auxiliary terms."""
    import torch
    if len(outputs) != len(weights):
        raise ValueError("One output per deep-supervision scale is required")
    total = 0.0
    for logits, weight in zip(outputs, weights):
        if logits.shape[-2:] != target.shape[-2:]:
            scaled = torch.nn.functional.interpolate(target, size=logits.shape[-2:],
                                                     mode="area")
            scaled = (scaled > 0.5).to(target.dtype)
        else:
            scaled = target
        total = total + weight * mask_loss(logits, scaled, base)
    return total / sum(weights)


def iou_head_loss(predicted_iou, actual_iou, weight=IOU_LOSS_WEIGHT):
    """MSE on the IoU scalar, down-weighted so it never dominates the mask objective."""
    import torch
    if not 0 < weight <= 1:
        raise ValueError("IoU loss weight must be in (0, 1]")
    return weight * torch.nn.functional.mse_loss(predicted_iou, actual_iou)


def ranking_auc(scores, labels):
    """Tie-aware Mann-Whitney AUC, so a constant head scores exactly 0.5.

    V14's head WAS effectively constant and this function would have exposed it, had
    the inputs been aligned. V15 aligns them and permutation-tests the result.
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
    unique, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    if (counts > 1).any():
        sums = np.zeros(len(unique))
        np.add.at(sums, inverse, ranks)
        ranks = (sums / counts)[inverse]
    return float((ranks[labels].sum() - positive * (positive + 1) / 2)
                 / (positive * negative))


def calibration_error(predicted, actual):
    """Mean absolute error of the predicted IoU against the realised IoU."""
    predicted, actual = np.asarray(predicted, dtype=float), np.asarray(actual, dtype=float)
    if predicted.shape != actual.shape:
        raise ValueError("Predicted and actual IoU must align")
    if not len(predicted):
        return float("nan")
    return float(np.abs(predicted - actual).mean())


def verify_head_alignment(scores, actual, seed=SEED):
    """Permutation check: shuffling scores must destroy the reported AUC.

    This is the direct guard against V14 defect 3. If predicted and actual are
    misaligned, the true AUC is already ~0.5 and shuffling changes nothing, so a
    permuted AUC that stays far from 0.5 -- or an unpermuted AUC that equals the
    permuted one -- means the diagnostic is not measuring the head.
    """
    scores, actual = np.asarray(scores, dtype=float), np.asarray(actual, dtype=float)
    labels = actual > MATCH_IOU
    observed = ranking_auc(scores, labels)
    permuted = ranking_auc(np.random.default_rng(seed).permutation(scores), labels)
    return dict(observed_auc=None if math.isnan(observed) else observed,
                permuted_auc=None if math.isnan(permuted) else permuted,
                permutation_gap=(None if math.isnan(observed) or math.isnan(permuted)
                                 else abs(observed - permuted)),
                samples=int(len(scores)),
                positives=int(labels.sum()), negatives=int((~labels).sum()))


def verify_scoring_cohort(actual_iou, minimum=MIN_NEGATIVE_FRACTION,
                          ceiling=NEGATIVE_IOU_CEILING):
    """The scoring head must see a realistic share of bad proposals.

    This is the direct guard against V14 defect 2: a cohort of positives only
    produces a constant head, which silently disables any gate built on it.
    """
    actual = np.asarray(actual_iou, dtype=float)
    if not len(actual):
        raise ValueError("Scoring cohort is empty")
    fraction = float((actual < ceiling).mean())
    if fraction < minimum:
        raise ValueError(
            f"Scoring cohort has only {fraction:.3f} negatives (<{ceiling}); "
            f"V14 shipped a constant head this way. Need >={minimum}.")
    return dict(samples=int(len(actual)), negative_fraction=fraction,
                negative_ceiling=ceiling, minimum_required=minimum)


def keep_fraction_frontier(scores, actual, n_gt, fractions=KEEP_FRACTIONS):
    """What PQ the learned score WOULD reach if it selected, at each keep fraction.

    Reported as a diagnostic only -- selection stays with V13's confidence grid. Uses
    fractions rather than absolute cuts so a head with any output range still yields
    a meaningful curve, which is the reporting half of the V14 defect-2 fix.
    """
    scores, actual = np.asarray(scores, dtype=float), np.asarray(actual, dtype=float)
    if scores.shape != actual.shape:
        raise ValueError("Scores and actual IoU must align")
    if not len(scores):
        return []
    order = np.argsort(-scores, kind="mergesort")
    rows = []
    for fraction in fractions:
        keep = int(round(len(order) * fraction))
        if keep <= 0:
            continue
        chosen = actual[order[:keep]]
        matched = chosen > MATCH_IOU
        tp = int(matched.sum())
        iou_sum = float(chosen[matched].sum())
        fp = keep - tp
        fn = max(int(n_gt) - tp, 0)
        rows.append(dict(keep_fraction=fraction, kept=keep, tp=tp, fp=fp, fn=fn,
                         iou_sum=iou_sum, pq=pq((iou_sum, tp, fp, fn)),
                         mean_matched_iou=float(iou_sum / tp) if tp else 0.0))
    return rows


def verify_control(results, reference=CONTROL_REFERENCE):
    """The frozen control must reproduce E015's accepted counts before any claim."""
    observed = results[reference["candidate"]]
    for group in ("tune", "audit"):
        expected, actual = reference[group], observed[group]
        if abs(expected[0] - actual[0]) > 1e-6 or [int(v) for v in actual[1:]] != expected[1:]:
            raise ValueError(f"Control {group} counts differ from the frozen E015 "
                             f"reference: {actual} vs {expected}")
    return {group: observed[group] for group in ("tune", "audit")}


def select_candidate(results):
    """Tune-only selection over the predeclared grid; ties broken by candidate name."""
    declared = set(declared_candidates())
    if not set(results) <= declared:
        raise ValueError("Results contain a candidate outside the predeclared grid")
    if not results:
        raise ValueError("No candidates to select from")
    return max(sorted(results), key=lambda key: pq(results[key]["tune"]))


def evaluate_bars(selected, results, baselines=BASELINES, submit_bar=SUBMIT_BAR,
                  gate=PROMOTION_GATE):
    """Both decisions, computed from predeclared constants only."""
    tune, audit = pq(results[selected]["tune"]), pq(results[selected]["audit"])
    v13 = baselines["v13_compose"]
    authorized = tune >= submit_bar["min_tune_pq"] and audit >= submit_bar["min_audit_pq"]
    promotable = tune - v13["tune_pq"] >= gate["required_tune_delta_vs_v13"]
    if gate["require_audit_at_least_v13"]:
        promotable = promotable and audit >= v13["audit_pq"]
    axes = axis_metrics(results[selected]["tune"])
    return dict(selected=selected, selected_tune_pq=tune, selected_audit_pq=audit,
                tune_axes=axes, audit_axes=axis_metrics(results[selected]["audit"]),
                axis_deltas_vs_v13=dict(
                    recall=axes["recall"] - v13["recall"],
                    mean_matched_iou=axes["mean_matched_iou"] - v13["mean_matched_iou"],
                    fp=axes["fp"] - v13["fp"]),
                deltas_vs_baselines={
                    name: dict(tune=tune - value["tune_pq"], audit=audit - value["audit_pq"])
                    for name, value in baselines.items()},
                submit_bar=submit_bar, export_authorized=bool(authorized),
                promotion_gate=gate, promotable=bool(promotable),
                capacity_hypothesis=dict(
                    v13_mean_matched_iou=v13["mean_matched_iou"],
                    v15_mean_matched_iou=axes["mean_matched_iou"],
                    supported=bool(axes["mean_matched_iou"] > v13["mean_matched_iou"]),
                    note="The run's actual question: does a correctly pretrained "
                         "ResNet34 beat V11's ResNet18 on the meanIoU axis under "
                         "identical selection?"),
                recommendation=(
                    "promote and submit" if promotable else
                    "record; do not spend a submission (E024: +0.004 is invisible)"
                    if authorized else
                    "below the best measured candidate; do not submit"))


def build_mask_head(base, warm_start=None, pretrained=True):
    """ResNet34 encoder (ALWAYS ImageNet-init) + mask head + IoU head.

    THE V14 FIX. V14 did `DualHeadRefiner(pretrained=warm_start is None)`, so
    supplying a warm start disabled ImageNet initialisation, and the ResNet18->34
    warm start then moved only 131 of 298 tensors -- leaving a mostly random encoder.
    Here `pretrained` defaults to True and is INDEPENDENT of the warm start: ImageNet
    weights load first, then any shape-matching warm-start tensors layer on top.

    Kept as a separate class from base.CropUNet so the frozen accepted pipeline
    (V6/V11/V13, ResNet18) is never modified by this experiment. The 4th input
    channel is the proposal HINT, not a hard boundary.
    """
    # Checked BEFORE importing torch so the guard is fast and CPU-testable.
    if not pretrained:
        raise ValueError(
            "V15 requires ImageNet initialisation: training a ResNet34 encoder from "
            "random weights for 8 epochs is exactly the V14 defect being fixed.")
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

    class MaskHeadRefiner(nn.Module):
        def __init__(self):
            super().__init__()
            # Unconditional ImageNet weights. No flag, so no way to lose them.
            encoder = resnet34(weights=ResNet34_Weights.DEFAULT)
            stem = encoder.conv1
            # Four input channels: RGB-replicated grayscale + proposal hint. The hint
            # weights start at zero so the model begins numerically identical to the
            # 3-channel ImageNet stem and must LEARN to use the hint.
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
            self.register_buffer("mean",
                                 torch.tensor([0.485, 0.456, 0.406, 0.0]).view(1, 4, 1, 1))
            self.register_buffer("std",
                                 torch.tensor([0.229, 0.224, 0.225, 1.0]).view(1, 4, 1, 1))

        def forward(self, image):
            if image.shape[1] != 4:
                raise ValueError("V15 expects 3 grayscale channels plus a proposal hint")
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

    model = MaskHeadRefiner()
    transfer = dict(pretrained=True, warm_start=None)
    if warm_start is not None:
        loaded, skipped = load_compatible_weights(model, warm_start)
        transfer.update(warm_start=str(Path(warm_start).name),
                        tensors_loaded=loaded, tensors_skipped=skipped,
                        total_tensors=loaded + skipped)
        print(json.dumps(dict(imagenet_init=True, **transfer)), flush=True)
    return model, transfer


def load_compatible_weights(model, checkpoint_path):
    """Load only shape-matching tensors, and say how many were skipped.

    V11's checkpoint is a ResNet18 U-Net; V15 is ResNet34, so most tensors cannot
    transfer -- but unlike V14 the untransferred ones keep their ImageNet values
    instead of staying random.
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
    """Entry point; GPU work lives in v15_runtime so this stays CPU-importable."""
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
        raise ValueError("Run scripts/prepare_v15_maskhead.py to stage a pinned run")
    verify_grid()
    runner = sys.modules["v15_runtime"]
    runner.MODULE_SHA256 = MODULE_SHA256
    runner.CHECKPOINT_SHA256 = CHECKPOINT_SHA256
    runner.RAW_CACHE_SHA256 = RAW_CACHE_SHA256
    runner.main(args)


if __name__ == "__main__":
    run_experiment()
