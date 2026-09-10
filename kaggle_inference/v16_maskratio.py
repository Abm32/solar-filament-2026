"""V16 detector training-target-resolution comparison: mask_ratio=1 vs 2.

Design: docs/v16_design_2026-09-08.md. Directly adapts kaggle_inference/
v12_resolution.py (E022, imgsz 1024 vs 1536, mask_ratio=2 fixed throughout) by
swapping the varying axis: imgsz is now FIXED at 1024 for both arms and at
inference, and mask_ratio varies {1, 2}. No prior experiment varied mask_ratio;
E022 only ever tested image resolution.

Protocol, declared BEFORE any V16 score is observed:

  * Three candidates: frozen control (shipped V6, mask_ratio=2), arm_r2 (fresh
    20-epoch continuation at mask_ratio=2, a harness sanity check), arm_r1
    (identical continuation at mask_ratio=1, the actual test).
  * Both trained arms warm-started independently from the pinned V6 best
    checkpoint with resume=False. Identical 20-epoch schedule, AdamW lr0=1e-4
    lrf=0.1, warmup 1 epoch, nbs=64, train_seed 42 (overridable via --train-seed for
    noise-pricing reruns; the dataset split seed is separately fixed and never
    varies), single_cls, overlap_mask=False,
    V6's augmentation -- copied verbatim from V12's audited recipe.
  * Identical global batch for both arms (fallback to 2 for both if 4 OOMs).
  * Inference is always at imgsz=1024 for every candidate; only the detector's
    training-time mask_ratio differs between arm_r1 and arm_r2.
  * The refiner, grouping, thresholds, confidence, min area, and pixel
    ownership stay frozen at V12/V13 settings.
  * Selection is tune-PQ only across {control, arm_r1, arm_r2}; audit never
    selects.

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


CONTROL = "v16_frozen_control"
ARMS = ("arm_r2", "arm_r1")
ARM_MASK_RATIO = {"arm_r2": 2, "arm_r1": 1}
TRAIN_RESOLUTION = 1024
INFERENCE_RESOLUTION = 1024
EPOCHS = 20
GLOBAL_BATCH = 4
FALLBACK_BATCH = 2
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
# V13, the best measured candidate to date, and its bars. Not cleared just by
# beating the control -- promotion requires clearing these, per E024's lesson
# that a small local gain (+0.0038) moved nothing at two public decimals.
V13_REFERENCE = {"tune_pq": 0.441777, "audit_pq": 0.464644}
SUBMIT_BAR = {"min_tune_pq": V13_REFERENCE["tune_pq"], "min_audit_pq": 0.450130}
PROMOTION_GATE = {"required_tune_delta_vs_v13": 0.01, "require_audit_at_least_v13": True}
RAW_CACHE_SHA256 = None
MODULE_SHA256 = None
DATASET_SHA256 = None


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


def training_arguments(mask_ratio, batch, epochs=EPOCHS, train_seed=DEFAULT_TRAIN_SEED):
    """The V12-audited matched-budget continuation config; only mask_ratio may
    differ between arms. imgsz is fixed at TRAIN_RESOLUTION for both arms, unlike
    V12 where imgsz was the varying axis. train_seed controls only Ultralytics'
    weight-init/augmentation RNG -- never the dataset split (SPLIT_SEED, fixed).
    """
    if mask_ratio not in ARM_MASK_RATIO.values():
        raise ValueError(f"Unexpected mask_ratio {mask_ratio}")
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
        imgsz=TRAIN_RESOLUTION, batch=batch,
    )


def verify_matched_budget(arguments):
    """Arms may differ ONLY in mask_ratio; anything else breaks the comparison.

    imgsz is checked FIRST and separately: unlike V12 (where imgsz was the
    legitimate varying axis), here it must be identical across arms, so a
    smuggled imgsz difference must be reported as its own specific failure
    rather than folded into the generic "differ outside mask_ratio" message.
    """
    if set(arguments) != set(ARMS):
        raise ValueError("Expected exactly the declared arms")
    resolutions = {arguments[arm]["imgsz"] for arm in ARMS}
    if resolutions != {TRAIN_RESOLUTION}:
        raise ValueError("Both arms must train at the fixed imgsz")
    reference = dict(arguments[ARMS[0]])
    for arm in ARMS[1:]:
        other = dict(arguments[arm])
        for key in ("mask_ratio",):
            reference.pop(key, None)
            other.pop(key, None)
        if other != reference:
            differing = {k for k in set(other) | set(reference)
                         if other.get(k) != reference.get(k)}
            raise ValueError(f"Arms differ outside mask_ratio: {sorted(differing)}")
    ratios = {arm: arguments[arm]["mask_ratio"] for arm in ARMS}
    if ratios != ARM_MASK_RATIO:
        raise ValueError("Arm mask_ratio values do not match their declared names")
    batches = {arguments[arm]["batch"] for arm in ARMS}
    if len(batches) != 1:
        raise ValueError("Both arms must use the same global batch")
    return True


def declared_candidates():
    """Every candidate this run may evaluate, declared up front (control + 2 arms).

    Predeclaring the full grid prevents expanding the search after seeing scores.
    """
    candidates = {CONTROL: dict(arm="frozen", mask_ratio=2,
                                infer_resolution=INFERENCE_RESOLUTION, priority=0)}
    for arm in ARMS:
        candidates[f"v16_{arm}"] = dict(
            arm=arm, mask_ratio=ARM_MASK_RATIO[arm],
            infer_resolution=INFERENCE_RESOLUTION, priority=1)
    return candidates


def polygon_to_yolo_line(polygon, width, height, class_id=0):
    """Match the established exporter: normalized, clipped, six decimal places."""
    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    if len(points) < 3:
        raise ValueError("A polygon needs at least three points")
    points[:, 0] = np.clip(points[:, 0] / width, 0, 1)
    points[:, 1] = np.clip(points[:, 1] / height, 0, 1)
    return f"{class_id} " + " ".join(f"{value:.6f}" for value in points.flatten())


def best_annotator_ids(images, annotations):
    """Most annotations, then largest total area, matching the frozen helper."""
    by_file = {}
    for image_id, info in images.items():
        by_file.setdefault(info["file_name"], []).append(image_id)

    def completeness(image_id):
        entries = annotations.get(image_id, [])
        return len(entries), sum(entry.get("area", 0) for entry in entries)

    return {name: max(ids, key=completeness) for name, ids in by_file.items()}


def dataset_partitions(images, annotations, provenance):
    """Rebuild the pinned V6 grouped split and assert it, never re-shuffle blindly."""
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
    by_file_ids = by_file
    train_ids = [image_id for name in train_files for image_id in sorted(by_file_ids[name])]
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
    """Regenerate the export: the downloaded one is incomplete (no val labels).

    Training uses unique ``_ann<image_id>`` names so every annotator variant of a
    filename is its own example, exactly as V6 was trained.
    """
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
    """Both splits must be complete before any GPU time is spent."""
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


def cache_id(name, split, mask_ratio, detector_sha):
    """Cache key includes checkpoint and mask_ratio so V6/V12 caches can never be
    reused across a different detector's proposals (imgsz is fixed here, so it is
    not part of the key, unlike V12 which varied it)."""
    if Path(name).name != name or split not in ("train", "test"):
        raise ValueError("Cache image must be a basename in train or test")
    key = f"{detector_sha}|mr{mask_ratio}|{INFERENCE_RESOLUTION}|{split}|{name}"
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
    grouping/refinement) against each GT instance, from the un-flipped view.

    This is the number V16 exists to move -- the structural cap ceiling_analysis
    measured at 0.8734 -- independent of anything the refiner does downstream.
    """
    rles = views[0][0] if views else []
    if not rles or not gt:
        return [0.0] * len(gt)
    iou = mask_utils.iou(rles, gt, [0] * len(gt))
    return iou.max(axis=0).tolist() if iou.size else [0.0] * len(gt)


def evaluate_candidate(runtime, tta, core, detector, candidate, config, names,
                       cache_root, output, progress_every=20):
    """Canonical PQ: every annotator entry separately, strict IoU > 0.5. Also
    records raw-proposal IoU per GT instance as a mask_ratio diagnostic."""
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
        cache_path = cache_root / cache_id(name, "train", mask_ratio, detector_sha)
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
    """The frozen control must reproduce E015's accepted counts before any claim."""
    for group in ("tune", "audit"):
        expected, actual = reference[group], counts[group]
        if len(expected) != len(actual):
            raise ValueError("Control counts have unexpected shape")
        if abs(expected[0] - actual[0]) > 1e-6 or [int(v) for v in actual[1:]] != expected[1:]:
            raise ValueError(f"Control {group} counts differ from the frozen V6 reference: "
                             f"{actual} vs {expected}")
    return True


def select_arm(results):
    """Rank by tuning PQ only. Audit never selects; it is reused screening evidence."""
    arms = {name: value for name, value in results.items() if name != CONTROL}
    if not arms:
        raise ValueError("No V16 candidates to select from")
    return max(sorted(arms), key=lambda name: pq(results[name]["tune"]))


def submission_and_promotion(selected, results):
    """Two independent bars: SUBMIT_BAR (may this be exported at all) and
    PROMOTION_GATE (should a submission actually be spent on it), both fixed
    before this run's numbers existed."""
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
        raise RuntimeError("V16 requires a CUDA GPU")
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


def smoke_mask_ratio(runtime, mask_ratio, batch, output, train_seed):
    """Confirm training with this mask_ratio actually runs before the real run."""
    import torch
    from ultralytics import YOLO
    arguments = training_arguments(mask_ratio, batch, epochs=1, train_seed=train_seed)
    arguments.update(fraction=0.02, save_period=-1, close_mosaic=0, plots=False, val=False,
                     project=str(output / "smoke"), name=f"probe_mr{mask_ratio}", exist_ok=False)
    torch.cuda.reset_peak_memory_stats()
    model = YOLO(str(runtime["source_root"] / "yolo_best.pt"))
    model.train(data=str(runtime["dataset"] / "data.yaml"), device=runtime["device"], **arguments)
    peak = max(torch.cuda.max_memory_allocated(i) for i in range(runtime["devices"]))
    del model
    torch.cuda.empty_cache()
    return dict(mask_ratio=mask_ratio, batch=batch, peak_bytes=int(peak), fits=True)


def train_arm(runtime, arm, batch, output, identity, train_seed):
    """Warm-start from the pinned V6 weights; never resume its stripped optimizer."""
    import torch
    from ultralytics import YOLO
    mask_ratio = ARM_MASK_RATIO[arm]
    marker = output / f"{arm}_training.json"
    weights = output / "checkpoints" / f"{arm}_best.pt"
    arguments = training_arguments(mask_ratio, batch, train_seed=train_seed)
    signature = dict(run_identity=identity, arm=arm, arguments=arguments,
                     dataset_sha256=runtime["dataset_digest"],
                     warm_start=V6_DETECTOR_SHA256)
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved["signature"] != signature:
            raise ValueError(f"Cannot resume a different {arm} configuration")
        if sha256(weights) != saved["checkpoint_sha256"]:
            raise ValueError(f"Changed {arm} checkpoint during resume")
        return saved
    if arguments["resume"] is not False:
        raise ValueError("V6 checkpoints are stripped; resume must be False")
    model = YOLO(str(runtime["source_root"] / "yolo_best.pt"))
    model.train(data=str(runtime["dataset"] / "data.yaml"), device=runtime["device"],
                project=str(output / "runs"), name=arm, exist_ok=False,
                plots=False, **arguments)
    save_dir = Path(model.trainer.save_dir)
    trainer_best = Path(model.trainer.best)
    weights.parent.mkdir(parents=True, exist_ok=True)
    weights.write_bytes(trainer_best.read_bytes())
    results = pd.read_csv(save_dir / "results.csv")
    del model
    torch.cuda.empty_cache()
    record = dict(signature=signature, arm=arm, mask_ratio=mask_ratio, batch=batch,
                  epochs_completed=int(len(results)),
                  elapsed_seconds=float(results["time"].iloc[-1]) if "time" in results else None,
                  checkpoint_sha256=sha256(weights),
                  checkpoint=str(weights.relative_to(output)),
                  results_sha256=sha256(save_dir / "results.csv"),
                  save_dir=str(save_dir))
    write_json(marker, record)
    return record


def run_v16():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--batch", type=int, default=GLOBAL_BATCH)
    parser.add_argument("--train-seed", type=int, default=DEFAULT_TRAIN_SEED,
                        help="Ultralytics weight-init/augmentation RNG seed only; "
                             "never affects the pinned dataset split (SPLIT_SEED).")
    args = parser.parse_args()
    if args.batch not in (GLOBAL_BATCH, FALLBACK_BATCH):
        parser.error("Batch must be the declared global batch or its fallback")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    runtime = load_runtime(args)
    runtime["device"] = list(range(runtime["devices"])) if runtime["devices"] > 1 else 0
    partitions = runtime["partitions"]
    dataset = output / "v16_yolo_dataset"
    if not dataset.exists():
        write_yolo_dataset(dataset, runtime["images"], runtime["annotations"],
                           partitions, runtime["base"].CFG.train_dir)
    inventory = verify_yolo_dataset(dataset, partitions)
    runtime["dataset"] = dataset
    runtime["dataset_digest"] = digest_json(
        {"labels": sorted((p.name, sha256(p)) for split in ("train", "val")
                          for p in (dataset / "labels" / split).glob("*.txt")),
         "inventory": inventory})
    identity = digest_json(dict(experiment="v16_detector_mask_ratio", arms=list(ARMS),
                                arguments={arm: training_arguments(ARM_MASK_RATIO[arm], args.batch,
                                                                    train_seed=args.train_seed)
                                           for arm in ARMS},
                                dataset=runtime["dataset_digest"],
                                warm_start=V6_DETECTOR_SHA256,
                                refiner=runtime["v10"].REFINER_SHA256,
                                train_seed=args.train_seed,
                                versions=runtime["versions"], smoke=args.smoke))
    arguments = {arm: training_arguments(ARM_MASK_RATIO[arm], args.batch, train_seed=args.train_seed)
                for arm in ARMS}
    verify_matched_budget(arguments)
    manifest = dict(experiment="v16_detector_mask_ratio", run_identity=identity,
                    arms=list(ARMS), arm_mask_ratio=ARM_MASK_RATIO,
                    inference_resolution=INFERENCE_RESOLUTION,
                    train_seed=args.train_seed,
                    declared_candidates=declared_candidates(),
                    submit_bar=SUBMIT_BAR, promotion_gate=PROMOTION_GATE,
                    training_arguments=arguments, dataset=inventory,
                    dataset_sha256=runtime["dataset_digest"],
                    warm_start_sha256=V6_DETECTOR_SHA256,
                    refiner_sha256=runtime["v10"].REFINER_SHA256,
                    tune_files=sorted(runtime["split"][1]), audit_files=sorted(runtime["split"][2]),
                    versions=runtime["versions"], devices=runtime["devices"],
                    global_batch=args.batch, smoke=args.smoke, complete=False)
    write_json(output / "v16_manifest.json", manifest)
    print(json.dumps({k: manifest[k] for k in
                      ("run_identity", "dataset", "global_batch", "devices", "smoke")}, indent=2), flush=True)
    if args.smoke:
        probes = [smoke_mask_ratio(runtime, ARM_MASK_RATIO[arm], args.batch, output, args.train_seed)
                 for arm in ARMS]
        manifest.update(smoke_probes=probes, complete=True)
        write_json(output / "v16_manifest.json", manifest)
        print(json.dumps(probes, indent=2), flush=True)
        return
    core, tta = runtime["core"], runtime["tta"]
    cache_root = output / "v16_proposal_cache"
    names = sorted(runtime["split"][0])
    candidates = declared_candidates()
    results, diagnostics, raw_summaries = {}, [], {}

    # 1. The frozen control first. If it cannot reproduce the accepted 0.38
    #    pipeline's counts, nothing else measured in this run is trustworthy.
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
    write_json(output / "v16_manifest.json", manifest)
    print(f"control verified tune_pq={pq(counts['tune']):.6f} "
          f"audit_pq={pq(counts['audit']):.6f} "
          f"raw_mean_iou={raw_summary['mean']:.4f} raw_max_iou={raw_summary['max']:.4f}", flush=True)

    # 2. Train both arms, then evaluate each at the fixed inference resolution.
    training = {}
    for arm in ARMS:
        training[arm] = train_arm(runtime, arm, args.batch, output, identity, args.train_seed)
        manifest["training"] = training
        write_json(output / "v16_manifest.json", manifest)
        print(f"trained {arm} mask_ratio={ARM_MASK_RATIO[arm]} "
              f"epochs={training[arm]['epochs_completed']} "
              f"elapsed={training[arm]['elapsed_seconds']}s", flush=True)
    for arm in ARMS:
        detector = runtime["base"].YOLO(str(output / training[arm]["checkpoint"]))
        candidate = f"v16_{arm}"
        config = dict(candidates[candidate], detector_sha256=training[arm]["checkpoint_sha256"])
        counts, rows, raw_summary = evaluate_candidate(runtime, tta, core, detector, candidate,
                                                       config, names, cache_root, output)
        results[candidate] = counts
        diagnostics.extend(rows)
        raw_summaries[candidate] = raw_summary
        write_v16_results(output, results, candidates, diagnostics, raw_summaries)
        print(f"{candidate} tune_pq={pq(counts['tune']):.6f} "
              f"audit_pq={pq(counts['audit']):.6f} "
              f"raw_mean_iou={raw_summary['mean']:.4f} raw_max_iou={raw_summary['max']:.4f} "
              f"raw_above_0.8734={raw_summary['above_0_8734']}", flush=True)
        del detector

    selected = select_arm(results)
    gate = submission_and_promotion(selected, results)
    harness_check = dict(
        control_tune_pq=pq(results[CONTROL]["tune"]),
        arm_r2_tune_pq=pq(results["v16_arm_r2"]["tune"]),
        arm_r2_vs_control_delta=pq(results["v16_arm_r2"]["tune"]) - pq(results[CONTROL]["tune"]),
        note="arm_r2 uses the shipped mask_ratio=2; large unexplained deltas vs "
             "control would indicate a harness issue, not a mask_ratio effect.")
    manifest.update(results=results, selection=gate, raw_proposal_summaries=raw_summaries,
                    harness_check=harness_check, complete=True)
    write_json(output / "v16_manifest.json", manifest)
    write_v16_results(output, results, candidates, diagnostics, raw_summaries)
    write_json(output / "v16_selection.json", dict(
        run_identity=identity, selected=selected, gate=gate, harness_check=harness_check,
        frozen_before_audit=True, candidate_counts=results, raw_proposal_summaries=raw_summaries,
        note="Selection uses tuning PQ only; audit is reused screening evidence."))
    print(json.dumps(gate, indent=2), flush=True)
    print(json.dumps(harness_check, indent=2), flush=True)


def write_v16_results(output, results, candidates, diagnostics, raw_summaries):
    rows = []
    for candidate, counts in results.items():
        row = dict(candidate=candidate, **candidates[candidate])
        for group, total in counts.items():
            row[f"{group}_pq"] = pq(total)
            row.update({f"{group}_{field}": value for field, value in zip(COUNTS, total)})
        row.update({f"raw_{k}": v for k, v in raw_summaries.get(candidate, {}).items()})
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values("tune_pq", ascending=False)
    frame.to_csv(output / "v16_candidates.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(output / "v16_image_diagnostics.csv", index=False)
    return frame


if __name__ == "__main__":
    run_v16()
