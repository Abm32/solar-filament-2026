"""V17: compose the V16 `mask_ratio=1` detector with V13's winning inference recipe.

This run TRAINS NOTHING. Both `arm_r1` detectors already exist as completed V16 kernel
outputs; V17 only runs inference and selection over them. That is deliberate and is the
central design point -- see "Why inference-only matters" below.

## What is actually untested here

V16 (E026) established two things about `mask_ratio=1`:

  * Raw proposal quality genuinely improves, replicated across two independent training
    seeds on both mean (+0.0109 / +0.0153) and max (+0.0110 / +0.0084) raw IoU, breaking
    `ceiling_analysis` §4's 0.8734 cap that had stood since V15.
  * It is NOT redundant with V13's shipped box `iou=0.3` lever. At the grouped
    pre-refiner stage on identical files and GT (`analyze_lever_overlap.py`):

        A  frozen V6 @ iou=0.7             recall 0.7416  meanIoU 0.6371  oraclePQ 0.5426
        B  frozen V6 @ iou=0.3  (V13's)    recall 0.7002  meanIoU 0.6366  oraclePQ 0.5243
        C  arm_r1 mask_ratio=1 @ iou=0.7   recall 0.7795  meanIoU 0.6442  oraclePQ 0.5644

    C > A > B on all three axes simultaneously.

But V16 then evaluated its own detector at exactly ONE inference configuration:
`t0.40/c0.40` at the Ultralytics DEFAULT box `iou=0.7`, using the CANONICAL (frozen)
refiner. It never swept the 42-candidate threshold/confidence grid, never applied box
`iou=0.3`, and never used V11's `proposal_centered_e4` refiner -- i.e. it never combined
the better detector with any part of the recipe that actually produced the 0.39 public
score. V13 = (iou=0.3 + V11 refiner + swept grid) on the OLD detector. V17 = the same
recipe on the BETTER detector. That composition has never been measured.

## Why inference-only matters

E026's other finding was a noise floor: re-running V16's *training* with an unchanged
configuration moved tune PQ by up to 0.0077 (seed 1's arm_r2 vs its control), which is
several times any arm-vs-control effect in that experiment. That floor is a property of
**detector training**, and it is why V16's end-to-end PQ deltas flipped sign between
seeds and could not be interpreted.

V17 trains nothing. The detectors are fixed files pinned by SHA-256, the refiner is a
fixed file pinned by SHA-256, and every proposal is cached and signed. The comparison is
therefore deterministic and exactly reproducible -- the noise floor that made V16
uninterpretable does not apply to this run at all. This is the cheapest available way to
convert V16's replicated proposal-quality gain into a measured end-to-end number.

## Protocol, declared before any V17 number is observed

  * **Frozen control runs FIRST**, exactly as V12/V13/V15/V16 did. It uses the frozen V6
    detector, the 321 frozen `iou=0.7` caches and the canonical refiner, must reproduce
    E015's exact tune/audit counts, and its exported CSV must hash to the accepted 0.38
    submission byte-for-byte (`bf11690e…`). If that fails, nothing else in this run is
    trustworthy and it aborts before any claim.
  * **Predeclared grid**: 6 thresholds x 7 confidences = 42 candidates, bound at runtime
    to `detector_tta.LOCAL_*` so it cannot silently widen, identical to E-A/V13.
  * **Box `iou` is a declared 2-cell axis** ({0.3, 0.7}), not a free parameter. E026
    measured the two levers as non-redundant but at DIFFERENT box `iou` values, so which
    one pairs best with `mask_ratio=1` is genuinely open and must be measured rather than
    assumed. Both cells are declared up front; 2 x 42 = 84 candidates total per arm.
  * **Selection is by tuning PQ ONLY**, across the declared candidates. Audit is reused
    screening evidence and never selects. Same as every prior run.
  * **SUBMIT_BAR** (authorises exporting a CSV at all): tune >= V13's 0.441777 AND audit
    >= E015's 0.450130. Clearing it authorises *measurement*, not promotion.
  * **PROMOTION_GATE** (authorises calling this a real gain): tune >= V13 + 0.010 with
    audit >= V13's 0.464644. Per E024, a local gain in the +0.003-0.010 band moved the
    public score not at all, so anything smaller is not worth a promotion claim.
  * **Seed-agreement check**: both V16 seeds' `arm_r1` detectors are evaluated. They are
    independent draws of the same configuration, so if the selected candidate's tune PQ
    disagrees sharply between them, the result is seed-specific and must be reported as
    such rather than as a property of `mask_ratio=1`. Selection uses seed 1 (the
    pre-registered primary); seed 2 is corroboration and never selects.

Nothing here trains, and nothing here submits. Submission stays a separate manual step
gated on SUBMIT_BAR.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

# Declared cells. `iou` is the axis; retina_masks/agnostic_nms match E-A/V13 byte-for-byte.
# E023 established agnostic_nms is INERT here (single_cls=True makes it a no-op); it is
# kept only so the proposal path is identical to the measured 0.39 recipe.
CELLS = (
    dict(name="mr1_iou030", retina_masks=True, agnostic_nms=True, iou=0.3),
    dict(name="mr1_iou070", retina_masks=True, agnostic_nms=True, iou=0.7),
)
CONTROL_CELL = dict(name="frozen_control", retina_masks=True, agnostic_nms=True, iou=0.7)
CONTROL = "v17_frozen_control"
# V16 arm_r1 == mask_ratio=1. Seed 1 is the pre-registered primary; seed 2 corroborates.
PRIMARY_SEED = "seed1"
ARM_DETECTORS = {
    "seed1": dict(train_seed=42, run="abhimanyu122/solar-filament-v16-mask-ratio",
                  detector_sha256="ae476e663039fbbbfda77ec519635ca15e2d50d1fe9f1af54735f84dbdc251ef"),
    "seed2": dict(train_seed=20260908, run="abhimanyu122/solar-filament-v16-mask-ratio-seed2",
                  detector_sha256="097a57ed7b5fe56909393ab05897ba5c41b96f7c343cb88d2ba782dd9dfabbe7"),
}
MASK_RATIO = 1
ARM = "proposal_centered"
V11_RUN_IDENTITY = "7c45acf26f210acce00f1bff581288bc2eb6121485237c7c028588d1de69164c"
V11_SOURCE_RUN = "abhimanyu122/solar-filament-v11-refiner"
V6_DETECTOR_SHA256 = "0abf568f5ffa504979a44c6ac4997c231adc38cb3c6d998461295c4dad3a38b8"
DETECTOR_RESOLUTION = 1024
DETECTOR_CONFIDENCE_FLOOR = 0.1
POLICY = "flip_consensus"
MATCH_IOU = 0.5
MODE = "refined_constrained"
MIN_AREA = 20
COUNTS = ("iou_sum", "tp", "fp", "fn")
# Predeclared grid, identical to E023/V13's, bound to detector_tta's frozen constants.
THRESHOLDS = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60)
CONFIDENCES = (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55)
# Accepted E015 control (== public 0.38).
CONTROL_REFERENCE = dict(
    candidate="t0.40_c0.40",
    tune=[426.5473399071961, 626, 211, 508],
    audit=[182.75261599994468, 265, 88, 194],
)
# Measured predecessors, for explicit like-for-like reporting.
PARENTS = dict(
    e015_control=dict(tune_pq=0.4328232774299302, audit_pq=0.4501295960589769,
                      public_score=0.38),
    v11_proposal_centered_e4=dict(tune_pq=0.438001, audit_pq=0.454884, public_score=0.39),
    v13_compose=dict(tune_pq=0.441777, audit_pq=0.464644, candidate="t0.45_c0.30",
                     public_score=0.39),
    v16_arm_r1_seed1=dict(tune_pq=0.43160351518532336, audit_pq=0.452409789693012,
                          candidate="t0.40_c0.40", public_score=None),
    v16_arm_r1_seed2=dict(tune_pq=0.4347952943716984, audit_pq=0.4481401986170993,
                          candidate="t0.40_c0.40", public_score=None),
)
SUBMIT_BAR = dict(min_tune_pq=0.441777, min_audit_pq=0.450130,
                  basis="V13 tune (best measured to date); audit floor is E015")
PROMOTION_GATE = dict(required_tune_delta_vs_v13=0.010, require_audit_at_least_v13=True,
                      v13_audit_pq=0.464644)
# E026's measured detector-TRAINING noise floor. Recorded so the report can state
# explicitly that it does not apply to this inference-only run.
V16_TRAINING_NOISE_FLOOR = 0.007677316654374711
CHECKPOINT_SHA256 = None
MODULE_SHA256 = None
RAW_CACHE_SHA256 = None
ARM_DETECTOR_SHA256 = None


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


def candidate_key(threshold, confidence):
    return f"t{threshold:.2f}_c{confidence:.2f}"


def cell_candidate_key(cell_name, threshold, confidence):
    """Cell-scoped so iou=0.3 and iou=0.7 candidates can never be conflated."""
    return f"{cell_name}/{candidate_key(threshold, confidence)}"


def declared_candidates():
    """Every candidate this run may ever consider, enumerated before it starts."""
    return tuple(cell_candidate_key(cell["name"], t, c)
                 for cell in CELLS for t in THRESHOLDS for c in CONFIDENCES)


def verify_grid_matches_frozen_constants(tta):
    """The grid must be E-A/V13's, taken from detector_tta's frozen constants."""
    if tuple(tta.LOCAL_THRESHOLDS) != THRESHOLDS:
        raise ValueError("Threshold grid differs from the frozen E-A/E015 grid")
    if tuple(tta.LOCAL_CONFIDENCES) != CONFIDENCES:
        raise ValueError("Confidence grid differs from the frozen E-A/E015 grid")
    if len(declared_candidates()) != 84:
        raise ValueError("Expected exactly 84 predeclared candidates (2 cells x 42)")
    if {cell["iou"] for cell in CELLS} != {0.3, 0.7}:
        raise ValueError("Declared cells must be exactly box iou 0.3 and 0.7")
    return True


def cache_id(name, split, cell_name, detector_sha, mask_ratio=MASK_RATIO):
    """Cache identity spans detector, mask_ratio AND cell.

    Three distinct proposal populations now exist for the same image: V6 @ iou=0.7
    (frozen), V6 @ iou=0.3 (V13's), and arm_r1 @ {0.3, 0.7} per seed. Including the
    detector hash, mask_ratio and the cell name makes collision impossible by
    construction, which is the discipline V13 introduced and E026 relied on.
    """
    if Path(name).name != name or split not in ("train", "test"):
        raise ValueError("Cache image must be a basename in train or test")
    token = (f"{detector_sha}|mr{mask_ratio}|{cell_name}|{split}|{name}"
             f"|imgsz{DETECTOR_RESOLUTION}|conf{DETECTOR_CONFIDENCE_FLOOR}")
    return hashlib.sha256(token.encode()).hexdigest()[:16] + ".json.gz"


def serialized_rle(rle):
    return dict(size=list(rle["size"]), counts=rle["counts"].decode())


def restored_rle(raw):
    return dict(size=list(raw["size"]), counts=raw["counts"].encode())


def generate_views(tta, core, detector, image_path, cell, cache_path, signature):
    """Four flip views at the cell's flags, signed and cached.

    Copied from V13 rather than parameterising it, so the accepted 0.38/0.39 code path
    is never edited by this composition.
    """
    cache_path = Path(cache_path)
    marker = cache_path.with_suffix(".signature.json")
    if cache_path.exists():
        if not marker.exists():
            raise ValueError(f"Unsigned existing cache: {cache_path}")
        recorded = json.loads(marker.read_text())
        if recorded["signature"] != signature:
            raise ValueError(f"Stale cache signature: {cache_path}")
        if recorded["sha256"] != sha256(cache_path):
            raise ValueError(f"Changed cache contents: {cache_path}")
        return read_views(cache_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read {image_path}")
    if image.shape[:2] != (2048, 2048):
        raise ValueError(f"Unexpected image geometry for {image_path}")
    views = []
    for axes in ((), (1,), (0,), (0, 1)):
        result = detector.predict(
            str(image_path) if not axes else tta.flip_array(image, axes),
            imgsz=DETECTOR_RESOLUTION, conf=DETECTOR_CONFIDENCE_FLOOR,
            agnostic_nms=cell["agnostic_nms"], retina_masks=cell["retina_masks"],
            iou=cell["iou"], verbose=False)[0]
        rles, scores = [], []
        if result.masks is not None:
            scores = result.boxes.conf.detach().cpu().numpy().tolist()
            for mask in result.masks.data:
                restored = tta.flip_array(mask.cpu().numpy().astype(np.uint8), axes)
                if restored.shape != image.shape[:2]:
                    raise ValueError("Detector masks must be in original image coordinates")
                rles.append(core.encode(restored))
        keep = [i for i, rle in enumerate(rles) if mask_utils.area(rle) > 0]
        views.append(([rles[i] for i in keep], [scores[i] for i in keep]))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(cache_path, "wt") as stream:
        json.dump([dict(rles=[serialized_rle(r) for r in rles], scores=scores)
                   for rles, scores in views], stream)
    write_json(marker, dict(signature=signature, sha256=sha256(cache_path)))
    return read_views(cache_path)


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
        if any(list(r["size"]) != [2048, 2048] or mask_utils.area(r) == 0 for r in rles):
            raise ValueError(f"Invalid proposal geometry: {path}")
        views.append((rles, scores))
    return views


def refine(runtime, core, model, views, image_path, thresholds):
    """One refiner pass per image, reused across every threshold in the grid."""
    proposals, scores = runtime["tta"].cluster_views(views, POLICY, MATCH_IOU)
    if not proposals:
        return {threshold: ([], np.asarray([])) for threshold in thresholds}
    masks = [mask_utils.decode(rle) for rle in proposals]
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None or gray.shape != (2048, 2048):
        raise ValueError(f"Cannot read {image_path}")
    bounds, _, probabilities = core.crop_probabilities(runtime["base"], model, gray, masks, True)
    return {threshold: core.clean_refined_predictions(masks, scores, bounds, probabilities,
                                                     threshold, MODE)
            for threshold in thresholds}


def raw_proposal_summary(runtime, core, views_by_file):
    """Pre-refiner IoU of the best single raw proposal per GT instance.

    This is E026's mechanistic measurement -- the one that replicated across seeds --
    so V17 reports it for its own proposal population as a consistency check against
    the +0.0109/+0.0153 mean and 0.8844/0.8818 max already established.
    """
    images, annotations, by_file = runtime["images"], runtime["annotations"], runtime["split"][4]
    best = []
    for name, views in views_by_file.items():
        rles, _ = runtime["tta"].cluster_views(views, POLICY, MATCH_IOU)
        for image_id in sorted(by_file[name]):
            info = images[image_id]
            gt = [core.annotation_rle(info, entry) for entry in annotations.get(image_id, [])]
            gt = [rle for rle in gt if mask_utils.area(rle) > 0]
            if not gt:
                continue
            if not rles:
                best.extend([0.0] * len(gt))
                continue
            iou = np.asarray(mask_utils.iou(rles, gt, [0] * len(gt)), dtype=np.float64)
            best.extend(iou.max(axis=0).tolist() if iou.size else [0.0] * len(gt))
    array = np.asarray(best, dtype=np.float64)
    if not array.size:
        return dict(n=0)
    return dict(n=int(array.size), mean=float(array.mean()), max=float(array.max()),
                p90=float(np.percentile(array, 90)),
                above_0_8734=int((array > 0.8734303008401842).sum()))


def evaluate_arm(runtime, core, model, detector, cell, cache_root, thresholds,
                 confidences, arm_name, detector_sha, progress_every=20,
                 collect_views=False):
    """Canonical PQ: every annotator entry scored separately, strict IoU > 0.5."""
    base, images, annotations = runtime["base"], runtime["images"], runtime["annotations"]
    held_out, tune, audit, _, by_file = runtime["split"]
    keys = [cell_candidate_key(cell["name"], t, c) for t in thresholds for c in confidences]
    totals = {key: {group: np.zeros(4) for group in ("tune", "audit")} for key in keys}
    diagnostics, collected = [], {}
    for index, name in enumerate(sorted(held_out), 1):
        views = arm_views(runtime, core, detector, cell, cache_root, name, "train", detector_sha)
        if collect_views:
            collected[name] = views
        refined = refine(runtime, core, model, views, base.CFG.train_dir / name, thresholds)
        group = "tune" if name in tune else "audit"
        gt_by_image = {}
        for image_id in sorted(by_file[name]):
            info = images[image_id]
            entries = [core.annotation_rle(info, entry) for entry in annotations.get(image_id, [])]
            gt_by_image[image_id] = [rle for rle in entries if mask_utils.area(rle) > 0]
        for threshold in thresholds:
            rles, scores = refined[threshold]
            for confidence in confidences:
                keep = np.asarray(scores) >= confidence if len(scores) else np.zeros(0, dtype=bool)
                kept = [rle for rle, flag in zip(rles, keep) if flag]
                key = cell_candidate_key(cell["name"], threshold, confidence)
                for image_id, gt in gt_by_image.items():
                    iou = (mask_utils.iou(kept, gt, [0] * len(gt)).T if kept and gt
                           else np.zeros((len(gt), len(kept))))
                    counts = core.pq_counts(iou, len(gt), len(kept))
                    totals[key][group] += counts
                    diagnostics.append(dict(arm=arm_name, cell=cell["name"], candidate=key,
                                            file_name=name, image_id=image_id, group=group,
                                            target="coco", **dict(zip(COUNTS, counts))))
        if index % progress_every == 0 or index == len(held_out):
            best = max(totals, key=lambda k: pq(totals[k]["tune"]))
            print(f"eval {arm_name}/{cell['name']} {index}/{len(held_out)} best={best} "
                  f"tune_pq={pq(totals[best]['tune']):.6f}", flush=True)
    results = {key: {group: total.tolist() for group, total in groups.items()}
               for key, groups in totals.items()}
    return (results, diagnostics, collected) if collect_views else (results, diagnostics)


def arm_views(runtime, core, detector, cell, cache_root, name, split, detector_sha):
    """Frozen caches for the control; freshly generated, signed caches for the arms."""
    if cell["name"] == CONTROL_CELL["name"]:
        return read_views(runtime["inventory"][(split, name)])
    signature = dict(cell=cell, split=split, file_name=name, mask_ratio=MASK_RATIO,
                     detector_sha256=detector_sha, resolution=DETECTOR_RESOLUTION,
                     confidence_floor=DETECTOR_CONFIDENCE_FLOOR, views=4)
    path = Path(cache_root) / cache_id(name, split, cell["name"], detector_sha)
    image_root = runtime["base"].CFG.train_dir if split == "train" else runtime["base"].CFG.test_dir
    return generate_views(runtime["tta"], core, detector, image_root / name, cell, path,
                          signature)


def verify_control(counts, reference=CONTROL_REFERENCE):
    """The frozen control must reproduce E015's accepted counts before any claim."""
    key = cell_candidate_key(CONTROL_CELL["name"], 0.40, 0.40)
    observed = counts[key]
    for group in ("tune", "audit"):
        expected, actual = reference[group], observed[group]
        if abs(expected[0] - actual[0]) > 1e-6 or [int(v) for v in actual[1:]] != expected[1:]:
            raise ValueError(f"Control {group} counts differ from the frozen E015 reference: "
                             f"{actual} vs {expected}")
    return {group: observed[group] for group in ("tune", "audit")}


def select_candidate(results):
    """Tune-only selection over the predeclared grid; ties broken by candidate name."""
    declared = set(declared_candidates())
    if not set(results) <= declared:
        outside = sorted(set(results) - declared)
        raise ValueError(f"Results contain candidates outside the predeclared grid: {outside[:3]}")
    if not results:
        raise ValueError("No candidates to select from")
    return max(sorted(results), key=lambda key: pq(results[key]["tune"]))


def evaluate_bars(selected, results, parents=PARENTS, submit_bar=SUBMIT_BAR,
                  gate=PROMOTION_GATE):
    """Both decisions, computed from predeclared constants only."""
    tune, audit = pq(results[selected]["tune"]), pq(results[selected]["audit"])
    authorized = tune >= submit_bar["min_tune_pq"] and audit >= submit_bar["min_audit_pq"]
    promotable = tune - parents["v13_compose"]["tune_pq"] >= gate["required_tune_delta_vs_v13"]
    if gate["require_audit_at_least_v13"]:
        promotable = promotable and audit >= gate["v13_audit_pq"]
    deltas = {name: dict(tune=tune - value["tune_pq"], audit=audit - value["audit_pq"])
              for name, value in parents.items()}
    return dict(selected=selected, selected_tune_pq=tune, selected_audit_pq=audit,
                deltas_vs_parents=deltas, submit_bar=submit_bar,
                export_authorized=bool(authorized), promotion_gate=gate,
                promotable=bool(promotable),
                inference_only=True,
                training_noise_floor_note=(
                    "E026 measured a detector-TRAINING noise floor of "
                    f"{V16_TRAINING_NOISE_FLOOR:.6f} tune PQ. V17 trains nothing: the "
                    "detectors, refiner and every cached proposal are pinned by SHA-256, "
                    "so that floor does not apply and this comparison is deterministic."),
                recommendation=(
                    "promote and submit" if promotable else
                    "export and measure one submission" if authorized else
                    "below V13; do not spend a submission"))


def seed_agreement(results_by_seed, selected):
    """Do both independently trained arm_r1 detectors agree at the selected candidate?

    E026's lesson was that a single training draw can mislead. Selection uses the
    pre-registered primary seed only; this reports whether the corroborating seed sees
    the same thing at the same configuration, without ever influencing selection.
    """
    report = {}
    for label, results in results_by_seed.items():
        if selected in results:
            report[label] = dict(tune_pq=pq(results[selected]["tune"]),
                                 audit_pq=pq(results[selected]["audit"]))
    values = [entry["tune_pq"] for entry in report.values()]
    spread = max(values) - min(values) if len(values) > 1 else 0.0
    return dict(per_seed=report, selected=selected, tune_pq_spread=spread,
                both_clear_submit_bar=all(
                    entry["tune_pq"] >= SUBMIT_BAR["min_tune_pq"]
                    and entry["audit_pq"] >= SUBMIT_BAR["min_audit_pq"]
                    for entry in report.values()) if report else False,
                note=("Selection uses the pre-registered primary seed only. A large spread "
                      "means the result is seed-specific, not a property of mask_ratio=1."))


def export(runtime, core, model, detector, cell, cache_root, config, output, filename,
           detector_sha):
    """Identical geometry, ownership, and 180-image coverage as every prior export."""
    base = runtime["base"]
    rows, image_counts = [], {}
    for index, path in enumerate(sorted(base.CFG.test_dir.glob("*.jpeg")), 1):
        views = arm_views(runtime, core, detector, cell, cache_root, path.name, "test",
                          detector_sha)
        rles, scores = refine(runtime, core, model, views, path, (config["threshold"],))[
            config["threshold"]]
        predictions = [rle for rle, score in zip(rles, scores) if score >= config["confidence"]]
        image_counts[path.name] = len(predictions)
        occupied = np.zeros((2048, 2048), dtype=bool)
        for number, rle in enumerate(predictions, 1):
            mask = mask_utils.decode(rle).astype(bool)
            if mask.sum() < MIN_AREA or (mask & occupied).any():
                raise ValueError("Invalid test prediction area or pixel ownership")
            occupied |= mask
            rows.append(dict(filament_id=f"{path.stem}_{number}",
                             segmentation_rle=rle["counts"].decode()))
        if index % 20 == 0 or index == 180:
            print(f"export={filename} images={index}/180 rows={len(rows)}", flush=True)
    if len(image_counts) != 180:
        raise ValueError("Incomplete test coverage during export")
    destination = Path(output) / filename
    pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]).to_csv(destination, index=False)
    return dict(file=filename, rows=len(rows), sha256=sha256(destination), config=config,
                evaluated_images=180,
                empty_images=[name for name, count in image_counts.items() if not count],
                structural_validation="all masks decoded, area>=20, per-image ownership disjoint")


def locate_v11_checkpoint(input_root, digest, arm=ARM):
    matches = [path for path in Path(input_root).rglob(f"{arm}_e4.pt") if sha256(path) == digest]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one pinned {arm}_e4.pt checkpoint")
    return matches[0]


def verify_v11_checkpoint(path, expected_identity=V11_RUN_IDENTITY):
    """The refiner must be the epoch-4 checkpoint from the completed V11 run."""
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("model", "run_identity", "epoch", "cohort_sha256"):
        if key not in payload:
            raise ValueError(f"V11 checkpoint is missing {key}")
    if payload["run_identity"] != expected_identity:
        raise ValueError("Checkpoint run identity does not match the completed V11 run")
    if payload["epoch"] != 4:
        raise ValueError("Expected the epoch-4 checkpoint")
    return dict(run_identity=payload["run_identity"], epoch=payload["epoch"],
                cohort_sha256=payload["cohort_sha256"], arm_file=Path(path).name)


def locate_arm_detector(input_root, digest):
    """The mask_ratio=1 detector must be a completed V16 arm_r1 checkpoint, by hash."""
    matches = [path for path in Path(input_root).rglob("arm_r1_best.pt")
               if sha256(path) == digest]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one arm_r1_best.pt with sha256 {digest[:16]}…")
    return matches[0]


def run_compose():
    import sys
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working"))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("limit must be nonnegative")
    if not MODULE_SHA256 or not CHECKPOINT_SHA256 or not RAW_CACHE_SHA256:
        raise ValueError("Run scripts/prepare_v17_compose.py to stage a pinned run")
    if not ARM_DETECTOR_SHA256 or set(ARM_DETECTOR_SHA256) != set(ARM_DETECTORS):
        raise ValueError("Staging must pin both V16 arm_r1 detector checkpoints")
    core = sys.modules["optimize_inference"]
    tta = sys.modules["detector_tta"]
    v10 = sys.modules["v10_quality"]
    v11 = sys.modules["v11_refiner"]
    import torch
    verify_grid_matches_frozen_constants(tta)
    # Reuse V11's complete input verification: V6 provenance, canonical refiner,
    # canonical annotations, split, crop geometry, all 321 frozen raw caches, and
    # exact library versions.
    v11.RAW_CACHE_SHA256 = RAW_CACHE_SHA256
    v11.MODULE_SHA256 = MODULE_SHA256
    runtime = v11.load_runtime(args)
    runtime["tta"] = tta
    if runtime["provenance"]["sha256"]["yolo_best.pt"] != V6_DETECTOR_SHA256:
        raise ValueError("V6 detector is not the pinned frozen detector")
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = locate_v11_checkpoint(args.input_root, CHECKPOINT_SHA256)
    checkpoint_identity = verify_v11_checkpoint(checkpoint)
    arm_paths = {label: locate_arm_detector(args.input_root, ARM_DETECTOR_SHA256[label])
                 for label in sorted(ARM_DETECTORS)}
    if len({str(path) for path in arm_paths.values()}) != len(arm_paths):
        raise ValueError("Both seeds resolved to the same detector file")

    thresholds, confidences, cells = THRESHOLDS, CONFIDENCES, CELLS
    seeds = sorted(ARM_DETECTORS)
    if args.limit:
        thresholds, confidences = (0.40,), (0.30, 0.40)
        cells, seeds = CELLS[:1], [PRIMARY_SEED]
    identity = digest_json(dict(script_sha256=sha256(Path(__file__)), modules=MODULE_SHA256,
                                checkpoint_sha256=CHECKPOINT_SHA256, cells=list(CELLS),
                                arm_detectors=ARM_DETECTOR_SHA256,
                                raw_cache_sha256=digest_json(RAW_CACHE_SHA256),
                                mask_ratio=MASK_RATIO, limit=args.limit))
    manifest_path = args.output / "v17_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous.get("run_identity") != identity:
            raise ValueError("Output directory belongs to a different V17 run")
        if previous.get("complete"):
            print(json.dumps(dict(already_complete=True)), flush=True)
            return
    manifest = dict(experiment="v17_mask_ratio1_x_v13_recipe", run_identity=identity,
                    cells=list(CELLS), control_cell=CONTROL_CELL, arm=ARM,
                    mask_ratio=MASK_RATIO, primary_seed=PRIMARY_SEED,
                    arm_detectors={label: dict(ARM_DETECTORS[label],
                                               resolved=str(arm_paths[label]))
                                   for label in arm_paths},
                    v11_source_run=V11_SOURCE_RUN, v11_checkpoint_sha256=CHECKPOINT_SHA256,
                    v11_checkpoint_identity=checkpoint_identity,
                    v6_detector_sha256=V6_DETECTOR_SHA256,
                    canonical_refiner_sha256=v10.REFINER_SHA256,
                    control_reference=CONTROL_REFERENCE, parents=PARENTS,
                    submit_bar=SUBMIT_BAR, promotion_gate=PROMOTION_GATE,
                    declared_candidates=list(declared_candidates()),
                    thresholds=list(thresholds), confidences=list(confidences),
                    evaluated_cells=[cell["name"] for cell in cells],
                    evaluated_seeds=seeds,
                    policy=POLICY, mode=MODE, match_iou=MATCH_IOU, min_prediction_area=MIN_AREA,
                    detector_resolution=DETECTOR_RESOLUTION,
                    detector_confidence_floor=DETECTOR_CONFIDENCE_FLOOR,
                    detector_views=4, crop_views=4, crop_size=runtime["base"].CFG.crop_size,
                    smoke_limit=args.limit, trains=False, submits=False,
                    versions=runtime["versions"], complete=False,
                    training_noise_floor_vs_v16=V16_TRAINING_NOISE_FLOOR,
                    note="Composes V16's replicated mask_ratio=1 detector with V13's measured "
                         "recipe (box iou swept over {0.3,0.7}, V11 proposal_centered_e4 "
                         "refiner, 42-candidate grid per cell). Trains nothing, so E026's "
                         "detector-training noise floor does not apply. Control must "
                         "reproduce the accepted 0.38 CSV first. Export gated on SUBMIT_BAR; "
                         "submission stays manual.")
    write_json(manifest_path, manifest)
    cache_root = args.output / "proposal_cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    # 1. Frozen control FIRST: canonical refiner + frozen iou=0.7 caches must reproduce
    #    E015's counts, and its CSV must hash to the accepted 0.38 file. Nothing
    #    downstream is trustworthy unless this passes.
    control_model = v11.build_model(runtime["base"], runtime["refiner"])
    control_results, control_diagnostics = evaluate_arm(
        runtime, core, control_model, None, CONTROL_CELL, cache_root,
        (0.40,), (0.40,), CONTROL, V6_DETECTOR_SHA256)
    control_counts = verify_control(control_results) if not args.limit else None
    print(json.dumps(dict(control_verified=bool(control_counts),
                          control_counts=control_counts)), flush=True)
    control_export = None
    if not args.limit:
        control_export = export(runtime, core, control_model, None, CONTROL_CELL, cache_root,
                                dict(threshold=0.40, confidence=0.40), args.output,
                                "submission_control.csv", V6_DETECTOR_SHA256)
        v10.verify_control_export(args.output / "submission_control.csv")
        print(json.dumps(dict(control_csv_verified=True, **control_export)), flush=True)
    del control_model
    torch.cuda.empty_cache()

    # 2. The composition: mask_ratio=1 proposals (both seeds, both box iou cells) fed to
    #    V11's continuation refiner.
    model = v11.build_model(runtime["base"], runtime["refiner"], False, checkpoint)
    results_by_seed, all_diagnostics, raw_summaries = {}, list(control_diagnostics), {}
    detectors = {}
    for label in seeds:
        detector = runtime["base"].YOLO(str(arm_paths[label]))
        detectors[label] = detector
        detector_sha = ARM_DETECTOR_SHA256[label]
        seed_results = {}
        for cell in cells:
            collect = cell is cells[0]
            outcome = evaluate_arm(runtime, core, model, detector, cell, cache_root,
                                   thresholds, confidences, f"v17_{label}", detector_sha,
                                   collect_views=collect)
            if collect:
                cell_results, diagnostics, views = outcome
                raw_summaries[label] = raw_proposal_summary(runtime, core, views)
                print(json.dumps(dict(seed=label, raw_proposal=raw_summaries[label])), flush=True)
            else:
                cell_results, diagnostics = outcome
            seed_results.update(cell_results)
            all_diagnostics.extend(diagnostics)
        results_by_seed[label] = seed_results
        rows = [dict(seed=label, cell=key.split("/")[0], candidate=key,
                     tune_pq=pq(value["tune"]), audit_pq=pq(value["audit"]),
                     **{f"tune_{f}": v for f, v in zip(COUNTS, value["tune"])},
                     **{f"audit_{f}": v for f, v in zip(COUNTS, value["audit"])})
                for key, value in seed_results.items()]
        frame = pd.DataFrame(rows).sort_values("tune_pq", ascending=False)
        frame.to_csv(args.output / f"v17_candidates_{label}.csv", index=False)
        write_json(args.output / "v17_results.json",
                   dict(control={CONTROL: control_results}, composition=results_by_seed,
                        raw_proposal_summaries=raw_summaries))
        print(f"seed {label} done best_tune_pq={frame['tune_pq'].max():.6f}", flush=True)

    pd.DataFrame(all_diagnostics).to_csv(args.output / "v17_image_diagnostics.csv", index=False)
    primary = results_by_seed[PRIMARY_SEED]
    selected = select_candidate(primary)
    report = evaluate_bars(selected, primary)
    report.update(control_counts=control_counts, control_csv=control_export,
                  raw_proposal_summaries=raw_summaries,
                  seed_agreement=seed_agreement(results_by_seed, selected))
    write_json(args.output / "v17_report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    if args.limit:
        print("Smoke test finished; no export, no gate decision", flush=True)
        manifest.update(complete=True, smoke=True)
        write_json(manifest_path, manifest)
        return

    # 3. Export only if the predeclared SUBMIT_BAR authorises measurement.
    candidate_export = None
    if report["export_authorized"]:
        cell_name, candidate = selected.split("/")
        cell = next(entry for entry in CELLS if entry["name"] == cell_name)
        threshold, confidence = (float(part[1:]) for part in candidate.split("_"))
        candidate_export = export(runtime, core, model, detectors[PRIMARY_SEED], cell,
                                  cache_root, dict(threshold=threshold, confidence=confidence),
                                  args.output, "submission.csv",
                                  ARM_DETECTOR_SHA256[PRIMARY_SEED])
        if candidate_export["sha256"] == control_export["sha256"]:
            raise ValueError("Candidate export is byte-identical to the control; nothing measured")
        write_json(args.output / "submission_candidates.json",
                   [dict(control_export, candidate=CONTROL, control_verified=True),
                    dict(candidate_export, candidate=selected,
                         promotable=report["promotable"],
                         export_authorized=report["export_authorized"])])
        print(json.dumps(dict(candidate_export=candidate_export), indent=2), flush=True)
    else:
        print("SUBMIT_BAR not cleared; no candidate CSV exported by design", flush=True)
    manifest.update(complete=True, smoke=False, exported=bool(candidate_export))
    write_json(manifest_path, manifest)


if __name__ == "__main__":
    run_compose()
