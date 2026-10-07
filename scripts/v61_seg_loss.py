"""V61 Phase B loss: Focal + Dice mask objective for Ultralytics v8SegmentationLoss.

Why: Ultralytics 8.4.138's mask loss is `BCE(pred_mask, gt_mask)` cropped to the box and
normalised by box area (single_mask_loss). It contains NO Dice and NO Focal term. The public
discussion reports Focal + Dice as the key ingredient; this makes that a real, isolated change.

Patched `single_mask_loss` keeps the original BCE and ADDS two terms on the same cropped region:

    L = BCE_term + w_focal * Focal_term + w_dice * Dice_term

  BCE_term   = sum_i  crop(BCE_i).mean(hw) / area_i            (byte-identical to upstream)
  Focal_term = sum_i  crop((1-p_t)^gamma * BCE_i).mean(hw) / area_i          (gamma = 2)
  Dice_term  = sum_i  dice_loss_i                                              (per instance)
                dice_i = 1 - (2*sum(p*g) + s) / (sum(p) + sum(g) + s), over the box crop,
                s = 1 smoothing

`area` is the GT box area in [0,1]; upstream divides the per-pixel-mean BCE by it. Dice is already
scale-free (a ratio in [0,1]), so it is NOT divided by area; it is summed per instance so the
caller's `/ fg_mask.sum()` normalises it on the same footing as the BCE term.

With w_focal = w_dice = 0 the function returns EXACTLY the upstream value (tested bit-for-bit
against the original staticmethod). Weights are fixed BEFORE any held-out number is seen.

Usage: call `install(w_focal, w_dice)` in the training process BEFORE building the trainer.
It patches the class in place and is idempotent. Single-process (single GPU) training only;
with DDP the subprocess would not inherit the patch, so the fold runs use one GPU each.
"""
from __future__ import annotations

# weights chosen a priori (recorded here, not tuned on held-out data):
#   focal 1.0 : same scale as the BCE term it reweights
#   dice  1.0 : Dice loss lives in [0,1] per instance; BCE term is ~O(0.1-1) per instance
W_FOCAL_DEFAULT = 1.0
W_DICE_DEFAULT = 1.0
GAMMA = 2.0
SMOOTH = 1.0
DEFAULT_CHUNK = 16        # instances per checkpointed block; memory-only knob, not a hyperparameter

_ORIGINAL = None


def make_single_mask_loss(torch, functional, crop_mask, original, w_focal, w_dice,
                          gamma=GAMMA, smooth=SMOOTH, chunk=None):
    """Build the replacement staticmethod (gt_mask, pred, proto, xyxy, area) -> scalar.

    chunk=None : extra terms computed on all instances at once (reference path, used by tests).
    chunk=K    : extra terms computed K instances at a time under activation checkpointing, so
                 peak memory is bounded by K regardless of instances/image. The result is the
                 same sum over instances (tested allclose incl. gradients); only memory differs.
                 Needed at mask_ratio=1 where masks are 1024x1024 (B OOMed un-chunked).
    """

    def extra_terms(pred, gt_mask, proto, xyxy, area):
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)
        bce = functional.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        p = torch.sigmoid(pred_mask)
        total = pred_mask.new_zeros(())
        if w_focal:
            p_t = p * gt_mask + (1 - p) * (1 - gt_mask)
            focal = (1 - p_t).pow(gamma) * bce
            total = total + w_focal * (crop_mask(focal, xyxy).mean(dim=(1, 2)) / area).sum()
        if w_dice:
            # crop_mask multiplies IN PLACE: clone anything reused/saved for backward.
            pc = crop_mask(p.clone(), xyxy)
            gc = crop_mask(gt_mask.clone(), xyxy)
            inter = (pc * gc).sum(dim=(1, 2))
            den = pc.sum(dim=(1, 2)) + gc.sum(dim=(1, 2))
            total = total + w_dice * (1 - (2 * inter + smooth) / (den + smooth)).sum()
        return total

    def single_mask_loss(gt_mask, pred, proto, xyxy, area):
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)
        bce = functional.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        total = (crop_mask(bce, xyxy).mean(dim=(1, 2)) / area).sum()
        if w_focal == 0 and w_dice == 0:
            return total
        del pred_mask, bce
        if chunk is None:
            return total + extra_terms(pred, gt_mask, proto, xyxy, area)
        from torch.utils.checkpoint import checkpoint
        for s in range(0, pred.shape[0], chunk):
            sl = slice(s, s + chunk)
            total = total + checkpoint(extra_terms, pred[sl], gt_mask[sl], proto, xyxy[sl],
                                       area[sl], use_reentrant=False)
        return total

    return single_mask_loss


def make_chunked_upstream(torch, functional, crop_mask, chunk=DEFAULT_CHUNK):
    """Memory-only replacement for the UPSTREAM BCE mask loss (no Focal/Dice).

    Upstream computes BCE for all instances at once; at mask_ratio=1 (1024x1024 masks) an image
    with many instances (e.g. with copy-paste) exceeds a 16 GB T4. This computes the identical
    per-instance sum `crop(BCE_i).mean(hw) / area_i` in `chunk`-instance checkpointed blocks.
    Value and gradient equal upstream (tested); only peak memory changes. Not a hyperparameter.
    """
    from torch.utils.checkpoint import checkpoint

    def block(pred, gt_mask, proto, xyxy, area):
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)
        bce = functional.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        return (crop_mask(bce, xyxy).mean(dim=(1, 2)) / area).sum()

    def single_mask_loss(gt_mask, pred, proto, xyxy, area):
        total = pred.new_zeros(())
        for s in range(0, pred.shape[0], chunk):
            sl = slice(s, s + chunk)
            total = total + checkpoint(block, pred[sl], gt_mask[sl], proto, xyxy[sl], area[sl],
                                       use_reentrant=False)
        return total

    return single_mask_loss


def install_chunked_upstream(chunk=DEFAULT_CHUNK):
    """Patch v8SegmentationLoss.single_mask_loss with the chunked upstream BCE. Returns info."""
    global _ORIGINAL
    import torch
    import torch.nn.functional as functional
    from ultralytics.utils import loss as L
    from ultralytics.utils.ops import crop_mask

    if _ORIGINAL is None:
        _ORIGINAL = L.v8SegmentationLoss.__dict__["single_mask_loss"].__func__
    fn = make_chunked_upstream(torch, functional, crop_mask, chunk)
    L.v8SegmentationLoss.single_mask_loss = staticmethod(fn)
    return {"loss": "upstream_bce_chunked", "chunk": chunk}


def install(w_focal=W_FOCAL_DEFAULT, w_dice=W_DICE_DEFAULT, chunk=DEFAULT_CHUNK):
    """Patch v8SegmentationLoss.single_mask_loss in this process. Returns the weights used."""
    global _ORIGINAL
    import torch
    import torch.nn.functional as functional
    from ultralytics.utils import loss as L
    from ultralytics.utils.ops import crop_mask

    if _ORIGINAL is None:
        _ORIGINAL = L.v8SegmentationLoss.__dict__["single_mask_loss"].__func__
    fn = make_single_mask_loss(torch, functional, crop_mask, _ORIGINAL, w_focal, w_dice,
                               chunk=chunk)
    L.v8SegmentationLoss.single_mask_loss = staticmethod(fn)
    return {"w_focal": w_focal, "w_dice": w_dice, "gamma": GAMMA, "smooth": SMOOTH,
            "chunk": chunk}


def original():
    """The upstream function captured before patching (None until install() was called)."""
    return _ORIGINAL
