"""V18 detector retrain: native imgsz=2048 training AND inference, mask_ratio=1.

Design rationale (not a doc, a code comment, per the pivot away from doc-driven
incremental tuning on 2026-09-08):

  * Native train/test images are 2048x2048 (verified via PIL). Every prior
    detector run (V6 through V17) trained at imgsz=1024, which downsamples every
    image by 2x before the detector ever sees it, and mask_ratio=2 halves the
    training mask target resolution again. That is an 8x effective upsample gap
    between the YOLO proto layer and the final mask (ceiling_analysis section 4),
    and it is the single largest untested lever in this project.
  * HDJoJo's actual 0.55-scoring inference notebook (hdjojo/solar-filament-seg-
    inference, re-fetched live 2026-09-08) is bare YOLOv8l-seg at imgsz=2048,
    conf=0.3, iou=0.0, no crop refiner at all. The edge is almost entirely in the
    detector's native training resolution, not inference-side postprocessing.
  * E022/V12 (2026-09-06) only ever varied imgsz between 1024 and 1536, always
    at mask_ratio=2, and it warm-started for only 20 epochs -- it never tested
    2048, never varied mask_ratio, and its own negative result does not bear on
    this combination. V16/V17 (2026-09-08) validated mask_ratio=1 as a real,
    small, replicated lever at imgsz=1024 -- this run moves that same lever to
    a higher resolution instead of stacking more inference-side grid search on a
    detector everyone agrees is capped (V11-V17 all scored 0.39 regardless of
    local PQ gains up to +0.017 over the accepted baseline).
  * CORRECTION, measured 2026-09-08 via the smoke grid (module name kept as
    "v18_native2048" for continuity with the run history, but the module trains
    at TRAIN_RESOLUTION, not literally 2048): full native imgsz=2048 training
    OOMs on a single Kaggle T4 (14.56GB) even at batch=1, with
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True set. imgsz=1536, batch=2
    DID fit (peak 11.9GB). This module therefore trains at 1536, the largest
    resolution empirically confirmed to fit on the available hardware -- still a
    real, 2.25x pixel-area increase over the 1024 baseline, and higher than
    E022/V12's 1536 test (which never varied mask_ratio and warm-started for
    only 20 epochs at a DIFFERENT, matched-1024-inference comparison).

This is a single arm, not a two-arm sanity-check pair like V16/V12: the harness
(control reproduction, cache identity, tune-only selection, submit/promotion
bars) is already validated by V11-V17 and E026's noise-floor pricing, so a
second "identical settings" sanity arm would only re-measure a noise floor
already known (E026: 0.007677 tune PQ at imgsz=1024; expected to differ at
1536, but a single well-verified control-vs-arm comparison is the right first
probe, not a second full training run before any 1536 result exists at all).

Protocol, declared BEFORE any V18 number is observed:

  * ONE arm: fresh continuation from the pinned V6 yolo11s-seg checkpoint
    (architecture-compatible warm start -- backbone size is NOT varied in this
    run, only resolution and mask_ratio, to keep the comparison to one axis
    change at a time), imgsz=TRAIN_RESOLUTION (1536, the measured-feasible
    value), mask_ratio=1, single_cls, overlap_mask=False, same augmentation
    recipe as V12/V16 (audited, unchanged).
  * Inference is ALSO at imgsz=TRAIN_RESOLUTION for the arm (matched train/infer resolution,
    unlike E022 which mismatched resolutions in some cells). The frozen control
    still evaluates at imgsz=1024 (its native, accepted setting) -- the two are
    not directly comparable at the raw-detector level, only via full PQ after
    the SAME frozen refiner and grid, exactly as V16 compared arms at 1024.
  * A `--smoke` mode trains 1 epoch at fraction=0.02 to empirically measure
    peak VRAM and per-epoch wall time BEFORE any full-budget commitment. This is
    the primary purpose of the first launch of this module.
  * Selection is tune-PQ only across {control, v18_arm}; audit never selects.
  * SUBMIT_BAR / PROMOTION_GATE identical in form to V16/V17 (tune >= V13's
    0.441777 and audit >= E015's 0.450130 to export; +0.010 tune over V13 and
    audit >= V13's 0.464644 to promote/spend a submission).

Nothing here submits. Export is a separate pinned step, gated on the frozen
control reproducing the accepted 0.38 CSV byte-for-byte.

CPU-importable helpers only at module level; torch/ultralytics load inside the
runtime paths so the local suite can import this without a GPU.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import random

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils


CONTROL = "v18_frozen_control"
ARM = "v18_native2048"
ARM_MASK_RATIO = 1
CONTROL_MASK_RATIO = 2
# Native images are 2048x2048, but the smoke grid (2026-09-08) measured that
# imgsz=2048 OOMs on a single Kaggle T4 (14.56GB) even at batch=1, with
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True set. imgsz=1536, batch=2 DID
# fit (peak 11.9GB). 1536 is not fully native, but it is a real, measured,
# feasible resolution increase over every prior detector run's 1024 (2.25x the
# pixel area vs the 4x of full 2048), and is the value this run actually trains
# at. TRAIN_RESOLUTION is kept as one named constant (not hardcoded at each
# call site) specifically so this value can be revisited if larger Kaggle
# accelerators or a smaller/quantized training path become available later.
TRAIN_RESOLUTION = 1536
CONTROL_INFERENCE_RESOLUTION = 1024  # The frozen control's native, accepted setting.
ARM_INFERENCE_RESOLUTION = 1536      # Matched to the arm's training resolution.
EPOCHS = 15  # Not V16's 20: imgsz=1536 costs more per epoch (measured ~0.55s/image vs
             # 1024's cheaper marginal cost), and the ops script caps full-run launches at
             # 10,800s. 15 epochs fits within that cap with an eval-time safety margin,
             # decided BEFORE any V18 training/eval number exists.
# CORRECTION 2026-09-09: the v18_full_kernel run at GLOBAL_BATCH=2 reproduced the
# control correctly, then died with torch.OutOfMemoryError ~18.5 minutes into
# epoch 1 (step ~555/937), inside the segmentation loss's mask interpolate call.
# Ultralytics' own batch=2->1 auto-retry had already fired once (step 7) and
# training continued at batch=1 for hundreds of steps before still OOMing --
# GPU_mem climbed 10.2G -> 12.4G -> 13.2G -> 14G (T4 ceiling 14.56G) over that
# span, an allocator-fragmentation ratchet ("5.00 GiB reserved by PyTorch but
# unallocated" in the error), not a single pathological batch. The smoke probe
# never caught this because smoke_probe() sets PYTORCH_CUDA_ALLOC_CONF but
# run_v18()'s full-training path never did -- the full run trained without the
# expandable-segments allocator the smoke test's "fits" verdict relied on. Fixed
# below: the env var is now set at the top of run_v18() before any CUDA/torch
# import, and GLOBAL_BATCH starts at 1 (removing the wasted, fragmentation-
# causing bs=2 attempt) rather than depending on Ultralytics' in-run fallback.
GLOBAL_BATCH = 1     # Fixed batch=1 from the start; bs=2 measured to OOM under sustained training.
FALLBACK_BATCH = 1
SPLIT_SEED = 42  # Fixed: reconstructs the pinned 566/141 V6 provenance split. Never vary.
DEFAULT_TRAIN_SEED = 42  # Overridable: Ultralytics weight-init/augmentation RNG only.
CONFIDENCE = 0.4
THRESHOLD = 0.4
MIN_AREA = 20
MATCH_IOU = 0.5
POLICY = "flip_consensus"
COUNTS = ("iou_sum", "tp", "fp", "fn")
ULTRALYTICS = "8.4.138"
V6_DETECTOR_SHA256 = "0abf568f5ffa504979a44c6ac4997c231adc38cb3c6d998461295c4dad3a38b8"
# Frozen V6 tuning/audit reference for the pinned control (E015 == accepted 0.38).
V6_REFERENCE = {
    "candidate": "flip_consensus_r1024_refined_constrained_t0.40_tta1_c0.40",
    "tune": [426.54733990719626, 626, 211, 508],
    "audit": [182.75261599994465, 265, 88, 194],
}
V13_REFERENCE = {"tune_pq": 0.441777, "audit_pq": 0.464644}
SUBMIT_BAR = {"min_tune_pq": V13_REFERENCE["tune_pq"], "min_audit_pq": 0.450130}
PROMOTION_GATE = {"required_tune_delta_vs_v13": 0.01, "require_audit_at_least_v13": True}
MODULE_SHA256 = None


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def pq(counts):
    iou, tp, fp, fn = counts
    denominator = tp + 0.5 * (fp + fn)
    return float(iou / denominator) if denominator else 0.0


def training_arguments(imgsz, mask_ratio, batch, epochs=EPOCHS, train_seed=DEFAULT_TRAIN_SEED):
    """The V12/V16-audited matched-budget continuation config. imgsz and
    mask_ratio both change vs the V6 baseline here -- that is the point of this
    run -- everything else is held to the same recipe already used by V11-V17."""
    if batch not in (GLOBAL_BATCH, FALLBACK_BATCH):
        raise ValueError(f"Unexpected global batch {batch}")
    if epochs < 1:
        raise ValueError("Epochs must be positive")
    return dict(
        epochs=epochs, patience=0, resume=False, optimizer="AdamW", lr0=1e-4, lrf=0.1,
        momentum=0.9, weight_decay=5e-4, cos_lr=False, warmup_epochs=1.0, warmup_bias_lr=0.0,
        nbs=64, seed=train_seed, deterministic=True, single_cls=True, mask_ratio=mask_ratio,
        overlap_mask=False, amp=True, rect=False, multi_scale=0.0, fraction=1.0, freeze=None,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.3, degrees=10, translate=0.1, scale=0.3, shear=0.0,
        perspective=0.0, fliplr=0.0, flipud=0.0, bgr=0.0, mosaic=0.3, close_mosaic=5,
        mixup=0.0, cutmix=0.0, copy_paste=0.0, save_period=5,
        imgsz=imgsz, batch=batch,
    )


def declared_candidates():
    return {
        CONTROL: dict(arm="frozen", mask_ratio=CONTROL_MASK_RATIO,
                      infer_resolution=CONTROL_INFERENCE_RESOLUTION, priority=0),
        ARM: dict(arm=ARM, mask_ratio=ARM_MASK_RATIO,
                 infer_resolution=ARM_INFERENCE_RESOLUTION, priority=1),
    }


def polygon_to_yolo_line(polygon, width, height, class_id=0):
    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    if len(points) < 3:
        raise ValueError("A polygon needs at least three points")
    points[:, 0] = np.clip(points[:, 0] / width, 0, 1)
    points[:, 1] = np.clip(points[:, 1] / height, 0, 1)
    return f"{class_id} " + " ".join(f"{value:.6f}" for value in points.flatten())


def best_annotator_ids(images, annotations):
    by_file = {}
    for image_id, info in images.items():
        by_file.setdefault(info["file_name"], []).append(image_id)

    def completeness(image_id):
        entries = annotations.get(image_id, [])
        return len(entries), sum(entry.get("area", 0) for entry in entries)

    return {name: max(ids, key=completeness) for name, ids in by_file.items()}


def dataset_partitions(images, annotations, provenance):
    by_file = {}
    for image_id, info in images.items():
        by_file.setdefault(info["file_name"], []).append(image_id)
    best = best_annotator_ids(images, annotations)
    files = sorted(best)
    random.Random(SPLIT_SEED).shuffle(files)
    held = files[:round(len(files) * 0.2)]
    train_files = sorted(set(files[round(len(files) * 0.2):]))
    if train_files != sorted(provenance["training_files"]):
        raise ValueError("Reconstructed training split differs from pinned V6 provenance")
    if sorted(held) != sorted(provenance["validation_files"]):
        raise ValueError("Reconstructed validation split differs from pinned V6 provenance")
    train_ids = [image_id for name in train_files for image_id in sorted(by_file[name])]
    valid_ids = [best[name] for name in sorted(held)]
    instances = sum(len(annotations.get(image_id, [])) for image_id in train_ids)
    if (len(train_files), len(held)) != (566, 141):
        raise ValueError("Expected 566 training and 141 validation filenames")
    if (len(train_ids), instances) != (937, 6606):
        raise ValueError(f"Expected 937 training entries and 6606 instances, got {len(train_ids)}/{instances}")
    if set(train_files) & set(held):
        raise ValueError("Training and validation filenames overlap")
    return dict(train_files=train_files, valid_files=sorted(held),
                train_ids=train_ids, valid_ids=valid_ids,
                train_entries=len(train_ids), train_instances=instances)


def write_yolo_dataset(root, images, annotations, partitions, train_dir):
    root = Path(root)
    if root.exists():
        raise ValueError(f"Refusing to overwrite an existing dataset directory: {root}")
    counts = {}
    for split, image_ids, unique in (("train", partitions["train_ids"], True),
                                     ("val", partitions["valid_ids"], False)):
        image_root, label_root = root / "images" / split, root / "labels" / split
        image_root.mkdir(parents=True, exist_ok=True)
        label_root.mkdir(parents=True, exist_ok=True)
        written = 0
        for image_id in image_ids:
            info = images[image_id]
            source = Path(train_dir) / info["file_name"]
            if not source.is_file():
                raise ValueError(f"Missing source image {source}")
            stem = Path(info["file_name"]).stem
            suffix = Path(info["file_name"]).suffix
            name = f"{stem}_ann{image_id}{suffix}" if unique else info["file_name"]
            target = image_root / name
            if not target.exists():
                target.symlink_to(source)
            lines = [polygon_to_yolo_line(polygon, info["width"], info["height"])
                     for entry in annotations.get(image_id, [])
                     for polygon in entry["segmentation"] if len(polygon) >= 6]
            (label_root / f"{Path(name).stem}.txt").write_text("\n".join(lines) + "\n" if lines else "")
            written += 1
        counts[split] = written
    if counts["train"] != len(partitions["train_ids"]) or counts["val"] != len(partitions["valid_ids"]):
        raise ValueError("Dataset export wrote an unexpected number of examples")
    (root / "data.yaml").write_text(
        f"path: {root}\ntrain: images/train\nval: images/val\nnames:\n  0: filament\n")
    return counts


def verify_yolo_dataset(root, partitions):
    root = Path(root)
    result = {}
    for split, expected in (("train", len(partitions["train_ids"])),
                            ("val", len(partitions["valid_ids"]))):
        images = sorted(p.name for p in (root / "images" / split).glob("*.jpeg"))
        labels = sorted(p.name for p in (root / "labels" / split).glob("*.txt"))
        if len(images) != expected or len(labels) != expected:
            raise ValueError(f"{split}: expected {expected} images and labels, "
                             f"found {len(images)}/{len(labels)}")
        if {Path(n).stem for n in images} != {Path(n).stem for n in labels}:
            raise ValueError(f"{split}: image and label stems differ")
        result[split] = expected
    train_stems = {Path(p).stem.split("_ann")[0] for p in (root / "images/train").glob("*.jpeg")}
    valid_stems = {Path(p).stem for p in (root / "images/val").glob("*.jpeg")}
    if train_stems & valid_stems:
        raise ValueError("Dataset leaks validation filenames into training")
    instances = sum(len([line for line in path.read_text().splitlines() if line.strip()])
                    for path in (root / "labels/train").glob("*.txt"))
    if instances != partitions["train_instances"]:
        raise ValueError(f"Expected {partitions['train_instances']} training instances, found {instances}")
    result["train_instances"] = instances
    return result


def cache_id(name, split, mask_ratio, resolution, detector_sha):
    """Cache key includes resolution (unlike V16, which fixed it) since this run
    varies resolution between control (1024) and arm (2048) -- the two
    populations must never collide."""
    if Path(name).name != name or split not in ("train", "test"):
        raise ValueError("Cache image must be a basename in train or test")
    key = f"{detector_sha}|mr{mask_ratio}|r{resolution}|{split}|{name}"
    return hashlib.sha256(key.encode()).hexdigest()[:16] + ".json.gz"


def serialized_rle(rle):
    return dict(size=list(rle["size"]), counts=rle["counts"].decode("utf-8"))


def restored_rle(rle):
    return dict(size=list(rle["size"]), counts=rle["counts"].encode("utf-8"))


def read_views(path):
    with gzip.open(path, "rt") as stream:
        raw = json.load(stream)
    if not isinstance(raw, list) or len(raw) != 4:
        raise ValueError(f"Incomplete four-view cache: {path}")
    views = []
    for view in raw:
        rles = [restored_rle(r) for r in view["rles"]]
        scores = view["scores"]
        if len(rles) != len(scores) or any(not np.isfinite(s) or not 0 <= s <= 1 for s in scores):
            raise ValueError(f"Invalid cache scores: {path}")
        if any(r["size"] != [2048, 2048] or mask_utils.area(r) == 0 for r in rles):
            raise ValueError(f"Invalid proposal geometry: {path}")
        views.append((rles, scores))
    return views


def detector_views(tta, core, detector, image_path, resolution, cache_path):
    """Four flip views at the given resolution, always in original coordinates."""
    if cache_path.exists():
        return read_views(cache_path)
    image = cv2.imread(str(image_path))
    if image is None or image.shape[:2] != (2048, 2048):
        raise ValueError(f"Cannot read full-resolution image {image_path}")
    views = []
    for axes in ((), (1,), (0,), (0, 1)):
        source = str(image_path) if not axes else tta.flip_array(image, axes)
        result = detector.predict(source, imgsz=resolution, conf=0.1, agnostic_nms=True,
                                  retina_masks=True, verbose=False)[0]
        rles, scores = [], []
        if result.masks is not None:
            scores = result.boxes.conf.detach().cpu().numpy().tolist()
            for mask in result.masks.data:
                restored = tta.flip_array(mask.cpu().numpy().astype(np.uint8), axes)
                if restored.shape != (2048, 2048):
                    raise ValueError("Detector masks must be in original image coordinates")
                rles.append(core.encode(restored))
        keep = [i for i, rle in enumerate(rles) if mask_utils.area(rle) > 0]
        views.append(([rles[i] for i in keep], [scores[i] for i in keep]))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(cache_path, "wt") as stream:
        json.dump([dict(rles=[serialized_rle(r) for r in rles], scores=scores)
                   for rles, scores in views], stream)
    return views


def predict_image(runtime, tta, core, detector, image_path, resolution, cache_path):
    """Frozen postprocess: consensus grouping, frozen refiner, ownership, min area."""
    views = detector_views(tta, core, detector, image_path, resolution, cache_path)
    rles, scores = tta.cluster_views(views, POLICY, MATCH_IOU)
    if not rles:
        return [], np.asarray([]), views
    masks = [mask_utils.decode(rle) for rle in rles]
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Cannot read {image_path}")
    bounds, _, averaged = core.crop_probabilities(runtime["base"], runtime["refiner"],
                                                  gray, masks, True)
    final = core.clean_refined_predictions(masks, scores, bounds, averaged,
                                          THRESHOLD, "refined_constrained")
    return final[0], final[1], views


def raw_proposal_iou(views, gt):
    """Diagnostic only: max IoU of any single raw detector proposal (before
    grouping/refinement) against each GT instance. This is the number V16
    moved a little at 1024; this run asks whether native 2048 moves it more."""
    rles = views[0][0] if views else []
    if not rles or not gt:
        return [0.0] * len(gt)
    iou = mask_utils.iou(rles, gt, [0] * len(gt))
    return iou.max(axis=0).tolist() if iou.size else [0.0] * len(gt)


def evaluate_candidate(runtime, tta, core, detector, candidate, config, names,
                       cache_root, output, progress_every=20):
    """Canonical PQ: every annotator entry separately, strict IoU > 0.5. Also
    records raw-proposal IoU per GT instance as a resolution diagnostic."""
    base = runtime["base"]
    images, annotations = runtime["images"], runtime["annotations"]
    _, tune, audit, _, by_file = runtime["split"]
    resolution = config["infer_resolution"]
    mask_ratio = config["mask_ratio"]
    detector_sha = config.get("detector_sha256", V6_DETECTOR_SHA256)
    totals = {group: np.zeros(4) for group in ("tune", "audit", "all")}
    diagnostics = []
    raw_ious = []
    for index, name in enumerate(sorted(names), 1):
        cache_path = cache_root / cache_id(name, "train", mask_ratio, resolution, detector_sha)
        rles, scores, views = predict_image(runtime, tta, core, detector,
                                            base.CFG.train_dir / name, resolution, cache_path)
        keep = np.asarray(scores) >= CONFIDENCE if len(scores) else np.zeros(0, dtype=bool)
        kept = [rle for rle, flag in zip(rles, keep) if flag]
        group = "tune" if name in tune else "audit"
        for image_id in sorted(by_file[name]):
            info = images[image_id]
            gt = [core.annotation_rle(info, entry) for entry in annotations.get(image_id, [])]
            gt = [rle for rle in gt if mask_utils.area(rle) > 0]
            iou = (mask_utils.iou(kept, gt, [0] * len(gt)).T if kept and gt
                   else np.zeros((len(gt), len(kept))))
            counts = core.pq_counts(iou, len(gt), len(kept))
            totals[group] += counts
            totals["all"] += counts
            raw_ious.extend(raw_proposal_iou(views, gt))
            diagnostics.append(dict(candidate=candidate, file_name=name, image_id=image_id,
                                    group=group, target="coco",
                                    **dict(zip(COUNTS, counts))))
        if index % progress_every == 0 or index == len(names):
            print(f"eval {candidate} {index}/{len(names)} "
                  f"tune_pq={pq(totals['tune']):.6f}", flush=True)
    raw_summary = dict(
        n=len(raw_ious),
        mean=float(np.mean(raw_ious)) if raw_ious else 0.0,
        max=float(np.max(raw_ious)) if raw_ious else 0.0,
        p90=float(np.percentile(raw_ious, 90)) if raw_ious else 0.0,
        above_0_8734=int(sum(1 for v in raw_ious if v > 0.8734)),
    )
    return {group: total.tolist() for group, total in totals.items()}, diagnostics, raw_summary


def verify_control(counts, reference=V6_REFERENCE):
    for group in ("tune", "audit"):
        expected, actual = reference[group], counts[group]
        if len(expected) != len(actual):
            raise ValueError("Control counts have unexpected shape")
        if abs(expected[0] - actual[0]) > 1e-6 or [int(v) for v in actual[1:]] != expected[1:]:
            raise ValueError(f"Control {group} counts differ from the frozen V6 reference: "
                             f"{actual} vs {expected}")
    return True


def select_arm(results):
    """With one arm this is deterministic, but kept as a function for symmetry
    with V16/V17 and to leave room for future arms without restructuring."""
    arms = {name: value for name, value in results.items() if name != CONTROL}
    if not arms:
        raise ValueError("No V18 candidates to select from")
    return max(sorted(arms), key=lambda name: pq(results[name]["tune"]))


def submission_and_promotion(selected, results):
    tune_pq, audit_pq = pq(results[selected]["tune"]), pq(results[selected]["audit"])
    export_authorized = (tune_pq >= SUBMIT_BAR["min_tune_pq"]
                         and audit_pq >= SUBMIT_BAR["min_audit_pq"])
    tune_delta_vs_v13 = tune_pq - V13_REFERENCE["tune_pq"]
    promotable = (export_authorized
                 and tune_delta_vs_v13 >= PROMOTION_GATE["required_tune_delta_vs_v13"]
                 and (audit_pq >= V13_REFERENCE["audit_pq"]
                      if PROMOTION_GATE["require_audit_at_least_v13"] else True))
    return dict(selected=selected, selected_tune_pq=tune_pq, selected_audit_pq=audit_pq,
               submit_bar=SUBMIT_BAR, export_authorized=bool(export_authorized),
               promotion_gate=PROMOTION_GATE, tune_delta_vs_v13=tune_delta_vs_v13,
               promotable=bool(promotable),
               recommendation=("clears promotion gate; consider spending a submission"
                               if promotable else
                               "clears submit bar but not promotion gate; record, do not submit"
                               if export_authorized else
                               "below submit bar; do not export"))


def load_runtime(args):
    """Resolve and hash every pinned input before a single GPU second is spent."""
    import optimize_inference as core
    import detector_tta as tta
    import v10_quality as v10
    import torch
    sources = list(args.input_root.rglob("v6_provenance.json"))
    refiners = list(args.input_root.rglob("best_refiner_coco.pt"))
    if len(sources) != 1 or len(refiners) != 1:
        raise ValueError("Expected exactly one pinned V6 source and canonical refiner")
    provenance_path, refiner_path = sources[0], refiners[0]
    if sha256(provenance_path) != v10.PROVENANCE_SHA256:
        raise ValueError("Changed V6 provenance")
    if sha256(refiner_path) != v10.REFINER_SHA256:
        raise ValueError("Changed canonical refiner")
    provenance = json.loads(provenance_path.read_text())
    for name, digest in provenance["sha256"].items():
        if sha256(provenance_path.parent / name) != digest:
            raise ValueError(f"Changed V6 source artifact {name}")
    if provenance["sha256"]["yolo_best.pt"] != V6_DETECTOR_SHA256:
        raise ValueError("V6 warm-start checkpoint is not the pinned detector")
    base = core.load_base(provenance_path.parent / "solar_base.py")
    base.seed_everything(args.train_seed)
    if sha256(base.CFG.train_json) != provenance["annotation_sha256"]:
        raise ValueError("Changed canonical annotations")
    _, images, annotations = base.load_coco()
    split = core.validation_split(base, images, annotations)
    held, tune, audit, _, by_file = split
    core.verify_split(provenance, held, images)
    if (len(provenance["training_files"]), len(tune), len(audit),
            sum(map(len, by_file.values()))) != (566, 94, 47, 217):
        raise ValueError("Unexpected training/tuning/audit/annotator split")
    if (base.CFG.crop_size, base.CFG.crop_context, base.CFG.min_crop) != (256, 1.8, 96):
        raise ValueError("Changed crop geometry")
    if base.version("ultralytics") != ULTRALYTICS:
        raise ValueError(f"Expected pinned Ultralytics {ULTRALYTICS}")
    partitions = dataset_partitions(images, annotations, provenance)
    if not torch.cuda.is_available():
        raise RuntimeError("V18 requires a CUDA GPU")
    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    refiner = base.CropUNet(pretrained=False).to("cuda:0")
    refiner.load_state_dict(torch.load(refiner_path, map_location="cuda:0", weights_only=True))
    refiner.eval()
    return dict(base=base, core=core, tta=tta, v10=v10, images=images, annotations=annotations,
                split=split, provenance=provenance, partitions=partitions,
                source_root=provenance_path.parent, refiner=refiner,
                refiner_path=refiner_path, devices=torch.cuda.device_count(),
                versions={name: base.version(name) for name in
                          ("ultralytics", "torch", "torchvision", "pycocotools", "numpy")})


def smoke_probe(runtime, imgsz, mask_ratio, batch, output, train_seed, tag, fraction=0.02):
    """Confirm training at this imgsz/batch actually fits before the real run.
    Reports peak VRAM and wall time for ONE epoch at the given data fraction so
    the full run's per-epoch cost can be estimated before committing the full
    GPU budget. The default 0.02 fraction is for a pure fit/OOM check; a larger
    fraction (e.g. 0.2) gives a much more reliable per-image time estimate,
    since a 2%-fraction probe's fixed overhead (model load, dataset indexing,
    first-batch warmup) dominates its total wall time and badly overestimates
    marginal per-image cost when naively extrapolated to a full epoch.
    Catches CUDA OOM so a probe grid can try several configs in one launch
    without one failure aborting the rest."""
    import os
    import time
    import torch
    from ultralytics import YOLO
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    arguments = training_arguments(imgsz, mask_ratio, batch, epochs=1, train_seed=train_seed)
    arguments.update(fraction=fraction, save_period=-1, close_mosaic=0, plots=False, val=False,
                     project=str(output / "smoke"), name=tag, exist_ok=False)
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    model = YOLO(str(runtime["source_root"] / "yolo_best.pt"))
    try:
        model.train(data=str(runtime["dataset"] / "data.yaml"), device=runtime["device"], **arguments)
    except torch.OutOfMemoryError as error:
        del model
        torch.cuda.empty_cache()
        return dict(tag=tag, imgsz=imgsz, mask_ratio=mask_ratio, batch=batch, fraction=fraction,
                   fits=False, error=str(error)[:300])
    elapsed = time.time() - start
    peak = max(torch.cuda.max_memory_allocated(i) for i in range(runtime["devices"]))
    del model
    torch.cuda.empty_cache()
    return dict(tag=tag, imgsz=imgsz, mask_ratio=mask_ratio, batch=batch, fraction=fraction,
               peak_bytes=int(peak), elapsed_seconds=float(elapsed), fits=True)



def train_arm(runtime, imgsz, mask_ratio, batch, output, identity, train_seed, tag):
    """Warm-start from the pinned V6 weights; never resume its stripped optimizer.
    Architecture-compatible warm start: this run does not change backbone size,
    only imgsz and mask_ratio, so the V6 yolo11s-seg checkpoint loads cleanly."""
    from ultralytics import YOLO
    import torch
    marker = output / f"{tag}_training.json"
    weights = output / "checkpoints" / f"{tag}_best.pt"
    arguments = training_arguments(imgsz, mask_ratio, batch, train_seed=train_seed)
    signature = dict(run_identity=identity, tag=tag, arguments=arguments,
                     dataset_sha256=runtime["dataset_digest"],
                     warm_start=V6_DETECTOR_SHA256)
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved["signature"] != signature:
            raise ValueError(f"Cannot resume a different {tag} configuration")
        if sha256(weights) != saved["checkpoint_sha256"]:
            raise ValueError(f"Changed {tag} checkpoint during resume")
        return saved
    if arguments["resume"] is not False:
        raise ValueError("V6 checkpoints are stripped; resume must be False")
    model = YOLO(str(runtime["source_root"] / "yolo_best.pt"))
    model.train(data=str(runtime["dataset"] / "data.yaml"), device=runtime["device"],
                project=str(output / "runs"), name=tag, exist_ok=False,
                plots=False, **arguments)
    save_dir = Path(model.trainer.save_dir)
    trainer_best = Path(model.trainer.best)
    weights.parent.mkdir(parents=True, exist_ok=True)
    weights.write_bytes(trainer_best.read_bytes())
    results = pd.read_csv(save_dir / "results.csv")
    del model
    torch.cuda.empty_cache()
    record = dict(signature=signature, tag=tag, imgsz=imgsz, mask_ratio=mask_ratio, batch=batch,
                  epochs_completed=int(len(results)),
                  elapsed_seconds=float(results["time"].iloc[-1]) if "time" in results else None,
                  checkpoint_sha256=sha256(weights),
                  checkpoint=str(weights.relative_to(output)),
                  results_sha256=sha256(save_dir / "results.csv"),
                  save_dir=str(save_dir))
    write_json(marker, record)
    return record


def run_v18():
    # Must be set before torch/ultralytics ever touch the GPU (load_runtime()
    # imports torch and allocates the refiner on cuda:0). The 2026-09-08 smoke
    # probe set this and reported "fits"; the 2026-09-09 full run did not set it
    # anywhere in this function and OOM'd under sustained training via allocator
    # fragmentation that expandable_segments specifically targets. Fixed here so
    # the full run actually uses the configuration the smoke test validated.
    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--batch", type=int, default=GLOBAL_BATCH)
    parser.add_argument("--train-seed", type=int, default=DEFAULT_TRAIN_SEED)
    args = parser.parse_args()
    if args.batch not in (GLOBAL_BATCH, FALLBACK_BATCH):
        parser.error("Batch must be the declared global batch or its fallback")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    runtime = load_runtime(args)
    # Unlike V16 (batch=4, cleanly divisible across 2 GPUs), V18's batch must stay
    # small (1-2) at imgsz=2048 due to VRAM. DDP splits the global batch across
    # devices, so batch=1 with 2 GPUs gives batch_per_device=0 and Ultralytics
    # raises. Force single-GPU training here; VRAM is the binding constraint
    # regardless of device count at this resolution.
    runtime["device"] = 0
    partitions = runtime["partitions"]
    dataset = output / "v18_yolo_dataset"
    if not dataset.exists():
        write_yolo_dataset(dataset, runtime["images"], runtime["annotations"],
                           partitions, runtime["base"].CFG.train_dir)
    inventory = verify_yolo_dataset(dataset, partitions)
    runtime["dataset"] = dataset
    runtime["dataset_digest"] = digest_json(
        {"labels": sorted((p.name, sha256(p)) for split in ("train", "val")
                          for p in (dataset / "labels" / split).glob("*.txt")),
         "inventory": inventory})
    arm_arguments = training_arguments(TRAIN_RESOLUTION, ARM_MASK_RATIO, args.batch,
                                       train_seed=args.train_seed)
    identity = digest_json(dict(experiment="v18_native2048", arm=ARM,
                                arguments=arm_arguments,
                                dataset=runtime["dataset_digest"],
                                warm_start=V6_DETECTOR_SHA256,
                                refiner=runtime["v10"].REFINER_SHA256,
                                train_seed=args.train_seed,
                                versions=runtime["versions"], smoke=args.smoke))
    manifest = dict(experiment="v18_native2048", run_identity=identity, arm=ARM,
                    train_resolution=TRAIN_RESOLUTION,
                    control_inference_resolution=CONTROL_INFERENCE_RESOLUTION,
                    arm_inference_resolution=ARM_INFERENCE_RESOLUTION,
                    arm_mask_ratio=ARM_MASK_RATIO, control_mask_ratio=CONTROL_MASK_RATIO,
                    train_seed=args.train_seed,
                    declared_candidates=declared_candidates(),
                    submit_bar=SUBMIT_BAR, promotion_gate=PROMOTION_GATE,
                    training_arguments=arm_arguments, dataset=inventory,
                    dataset_sha256=runtime["dataset_digest"],
                    warm_start_sha256=V6_DETECTOR_SHA256,
                    refiner_sha256=runtime["v10"].REFINER_SHA256,
                    tune_files=sorted(runtime["split"][1]), audit_files=sorted(runtime["split"][2]),
                    versions=runtime["versions"], devices=runtime["devices"],
                    global_batch=args.batch, smoke=args.smoke, complete=False)
    write_json(output / "v18_manifest.json", manifest)
    print(json.dumps({k: manifest[k] for k in
                      ("run_identity", "dataset", "global_batch", "devices", "smoke")}, indent=2), flush=True)
    if args.smoke:
        # 2048 is already confirmed infeasible (OOM at batch=1 on a single T4,
        # measured 2026-09-08); the grid starts at TRAIN_RESOLUTION (1536) and
        # only falls back to a smaller batch, not back up to 2048. Each fit
        # check uses fraction=0.02 (cheap); once a config fits, a second,
        # fraction=0.2 timing probe on the SAME config gives a trustworthy
        # per-epoch cost estimate -- a 2%-fraction probe's fixed overhead
        # (model load, dataset indexing, first-batch warmup) badly
        # overestimates marginal per-image cost if naively extrapolated.
        grid = [(TRAIN_RESOLUTION, args.batch)]
        for candidate in ((TRAIN_RESOLUTION, FALLBACK_BATCH),):
            if candidate not in grid:
                grid.append(candidate)
        probes = []
        for imgsz, batch in grid:
            probe = smoke_probe(runtime, imgsz, ARM_MASK_RATIO, batch, output,
                               args.train_seed, f"probe_r{imgsz}_b{batch}_fit", fraction=0.02)
            probes.append(probe)
            print(json.dumps(probe, indent=2), flush=True)
            manifest.update(smoke_probes=probes, complete=False)
            write_json(output / "v18_manifest.json", manifest)
            if probe["fits"]:
                timing = smoke_probe(runtime, imgsz, ARM_MASK_RATIO, batch, output,
                                     args.train_seed, f"probe_r{imgsz}_b{batch}_timing",
                                     fraction=0.2)
                probes.append(timing)
                print(json.dumps(timing, indent=2), flush=True)
                manifest.update(smoke_probes=probes, complete=False)
                write_json(output / "v18_manifest.json", manifest)
                break
        manifest.update(smoke_probes=probes, complete=True)
        write_json(output / "v18_manifest.json", manifest)
        return
    core, tta = runtime["core"], runtime["tta"]
    cache_root = output / "v18_proposal_cache"
    names = sorted(runtime["split"][0])
    candidates = declared_candidates()
    results, diagnostics, raw_summaries = {}, [], {}

    # 1. The frozen control first, at its native accepted resolution (1024). If
    #    it cannot reproduce the accepted 0.38 pipeline's counts, nothing else
    #    measured in this run is trustworthy.
    control_detector = runtime["base"].YOLO(str(runtime["source_root"] / "yolo_best.pt"))
    control_config = dict(candidates[CONTROL], detector_sha256=V6_DETECTOR_SHA256)
    counts, rows, raw_summary = evaluate_candidate(runtime, tta, core, control_detector, CONTROL,
                                                   control_config, names, cache_root, output)
    verify_control(counts)
    results[CONTROL] = counts
    diagnostics.extend(rows)
    raw_summaries[CONTROL] = raw_summary
    del control_detector
    manifest.update(control_counts=counts, control_verified=True, control_raw_proposal=raw_summary)
    write_json(output / "v18_manifest.json", manifest)
    print(f"control verified tune_pq={pq(counts['tune']):.6f} "
          f"audit_pq={pq(counts['audit']):.6f} "
          f"raw_mean_iou={raw_summary['mean']:.4f} raw_max_iou={raw_summary['max']:.4f}", flush=True)

    # 2. Train the arm at native 2048 / mask_ratio=1, then evaluate it at
    #    matched inference resolution 2048.
    training = train_arm(runtime, TRAIN_RESOLUTION, ARM_MASK_RATIO, args.batch, output,
                         identity, args.train_seed, ARM)
    manifest["training"] = training
    write_json(output / "v18_manifest.json", manifest)
    print(f"trained {ARM} imgsz={TRAIN_RESOLUTION} mask_ratio={ARM_MASK_RATIO} "
          f"epochs={training['epochs_completed']} elapsed={training['elapsed_seconds']}s", flush=True)

    detector = runtime["base"].YOLO(str(output / training["checkpoint"]))
    arm_config = dict(candidates[ARM], detector_sha256=training["checkpoint_sha256"])
    counts, rows, raw_summary = evaluate_candidate(runtime, tta, core, detector, ARM,
                                                   arm_config, names, cache_root, output)
    results[ARM] = counts
    diagnostics.extend(rows)
    raw_summaries[ARM] = raw_summary
    write_v18_results(output, results, candidates, diagnostics, raw_summaries)
    print(f"{ARM} tune_pq={pq(counts['tune']):.6f} audit_pq={pq(counts['audit']):.6f} "
          f"raw_mean_iou={raw_summary['mean']:.4f} raw_max_iou={raw_summary['max']:.4f} "
          f"raw_above_0.8734={raw_summary['above_0_8734']}", flush=True)
    del detector

    selected = select_arm(results)
    gate = submission_and_promotion(selected, results)
    manifest.update(results=results, selection=gate, raw_proposal_summaries=raw_summaries,
                    complete=True)
    write_json(output / "v18_manifest.json", manifest)
    write_v18_results(output, results, candidates, diagnostics, raw_summaries)
    write_json(output / "v18_selection.json", dict(
        run_identity=identity, selected=selected, gate=gate,
        frozen_before_audit=True, candidate_counts=results, raw_proposal_summaries=raw_summaries,
        note="Selection uses tuning PQ only; audit is reused screening evidence. Control "
             "runs at imgsz=1024 (its native setting); the arm runs at imgsz=2048 "
             "(matched to its own training resolution) -- they are compared only via "
             "final PQ after the same frozen refiner and grid, not at the raw-detector level."))
    print(json.dumps(gate, indent=2), flush=True)


def write_v18_results(output, results, candidates, diagnostics, raw_summaries):
    rows = []
    for candidate, counts in results.items():
        row = dict(candidate=candidate, **candidates[candidate])
        for group, total in counts.items():
            row[f"{group}_pq"] = pq(total)
            row.update({f"{group}_{field}": value for field, value in zip(COUNTS, total)})
        row.update({f"raw_{k}": v for k, v in raw_summaries.get(candidate, {}).items()})
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values("tune_pq", ascending=False)
    frame.to_csv(output / "v18_candidates.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(output / "v18_image_diagnostics.csv", index=False)
    return frame


if __name__ == "__main__":
    run_v18()
