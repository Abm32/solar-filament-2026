"""V13: compose E-A's box `iou=0.3` proposals with V11's continuation-trained refiner.

Both parents are measured and orthogonal. E-A (E023) changed only inference-time
detector flags on the frozen V6 detector and PASSED its +0.003 gate, recovering 58
of 508 missed tuning instances at a net-positive PQ trade. V11 (E019/E020) changed
only refiner weights, failed its +0.01 gate, but scored **0.39 public** twice. This
run is the first time the two are combined.

  * Parent A, E-A best cell: box `iou=0.3`, `retina_masks=True`, `agnostic_nms=True`
    at t0.40/c0.30 -> tune 0.436726 / audit 0.459054 with the CANONICAL refiner.
  * Parent B, V11 `proposal_centered_e4`: tune 0.438001 / audit 0.454884 with
    `iou=0.7` (default) proposals.

Why this needs a fresh run rather than a re-export: V11's export path reads the 321
frozen raw detector caches, and those caches were generated at the Ultralytics
default box `iou=0.7`. `iou=0.3` changes the detector's own NMS, so its proposals
cannot be recovered from them. This run regenerates proposals at `iou=0.3` for all
141 validation and 180 test images, then feeds them to the V11 refiner.

`agnostic_nms` is kept True purely to match E-A's winning cell byte-for-byte. E023
established it is INERT here: the detector is `single_cls=True`, so class-agnostic
NMS is a no-op and the flag cannot affect any output.

Protocol, declared before any V13 number is observed:

  * Predeclared grid is exactly E-A's: 6 thresholds x 7 confidences = 42 candidates.
    No expansion, no new threshold values, no post-hoc cell invention.
  * Selection is by tuning PQ ONLY. Audit is reused screening evidence and never
    selects, matching E015/E018/E019/E022.
  * SUBMIT_BAR (decides whether a CSV is exported at all): the composition must
    reach at least the better parent's tuning PQ, and must not regress audit below
    the accepted E015 control. Clearing this authorizes measurement, not promotion.
  * PROMOTION_GATE (decides whether V13 replaces the accepted baseline): tune
    >= V11 + 0.003 with audit >= V11 audit. Reported honestly either way.
  * The frozen control arm runs FIRST. It uses the frozen `iou=0.7` caches and the
    canonical refiner, must reproduce E015's exact tune/audit counts, and its
    exported CSV must hash to the accepted 0.38 submission byte-for-byte. If that
    fails, nothing else in this run is trustworthy and it aborts before any claim.

Nothing here trains, and nothing here submits. Submission stays a separate manual
step gated on SUBMIT_BAR.
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

# E-A's winning cell. retina_masks/agnostic_nms match E023 exactly; iou is the lever.
CELL = dict(name="ea_iou030", retina_masks=True, agnostic_nms=True, iou=0.3)
CONTROL_CELL = dict(name="frozen_control", retina_masks=True, agnostic_nms=True, iou=0.7)
CONTROL = "v13_frozen_control"
ARM = "proposal_centered"
V11_RUN_IDENTITY = "7c45acf26f210acce00f1bff581288bc2eb6121485237c7c028588d1de69164c"
V11_SOURCE_RUN = "abhimanyu122/solar-filament-v11-refiner"
DETECTOR_RESOLUTION = 1024
DETECTOR_CONFIDENCE_FLOOR = 0.1
POLICY = "flip_consensus"
MATCH_IOU = 0.5
MODE = "refined_constrained"
MIN_AREA = 20
COUNTS = ("iou_sum", "tp", "fp", "fn")
# Predeclared grid, identical to E023's. Bound to detector_tta's frozen constants at
# runtime so it cannot silently widen.
THRESHOLDS = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60)
CONFIDENCES = (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55)
# Accepted E015 control (== public 0.38), the same counts v12_resolution pins.
CONTROL_REFERENCE = dict(
    candidate="t0.40_c0.40",
    tune=[426.5473399071961, 626, 211, 508],
    audit=[182.75261599994468, 265, 88, 194],
)
# Parent results, for an explicit like-for-like comparison in the report.
PARENTS = dict(
    e015_control=dict(tune_pq=0.4328232774299302, audit_pq=0.4501295960589769),
    ea_iou030=dict(tune_pq=0.4367255551286259, audit_pq=0.4590541571224397,
                   candidate="t0.40_c0.30", public_score=None),
    v11_proposal_centered_e4=dict(tune_pq=0.438001, audit_pq=0.454884, public_score=0.39),
)
SUBMIT_BAR = dict(min_tune_pq=0.4367255551286259, min_audit_pq=0.4501295960589769,
                  basis="better parent's tune (E-A 0.436726); audit floor is E015")
PROMOTION_GATE = dict(required_tune_delta_vs_v11=0.003, require_audit_at_least_v11=True)
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


def declared_candidates():
    """The full predeclared grid, enumerated up front so it cannot be widened."""
    return tuple(candidate_key(t, c) for t in THRESHOLDS for c in CONFIDENCES)


def verify_grid_matches_frozen_constants(tta):
    """The grid must be E-A's, taken from detector_tta's frozen tuning constants."""
    if tuple(tta.LOCAL_THRESHOLDS) != THRESHOLDS:
        raise ValueError("Threshold grid differs from the frozen E-A/E015 grid")
    if tuple(tta.LOCAL_CONFIDENCES) != CONFIDENCES:
        raise ValueError("Confidence grid differs from the frozen E-A/E015 grid")
    if len(declared_candidates()) != 42:
        raise ValueError("Expected exactly 42 predeclared candidates")
    return True


def cache_id(name, split, cell_name):
    """Cell-scoped cache identity: iou=0.3 proposals must never collide with iou=0.7."""
    if Path(name).name != name or split not in ("train", "test"):
        raise ValueError("Cache image must be a basename in train or test")
    token = f"{cell_name}|{split}|{name}|imgsz{DETECTOR_RESOLUTION}|conf{DETECTOR_CONFIDENCE_FLOOR}"
    return hashlib.sha256(token.encode()).hexdigest()[:16] + ".json.gz"


def serialized_rle(rle):
    return dict(size=rle["size"], counts=rle["counts"].decode())


def restored_rle(raw):
    return dict(size=raw["size"], counts=raw["counts"].encode())


def generate_views(tta, core, detector, image_path, cell, cache_path, signature):
    """Four flip views at the cell's flags, signed and cached; identical to E-A's path.

    Duplicated from detector_tta.proposal_views rather than parameterizing it, so the
    accepted 0.38/0.39 code path is never edited by this composition.
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
        # Identity view uses the original path, reproducing the accepted runtime.
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
        if any(r["size"] != [2048, 2048] or mask_utils.area(r) == 0 for r in rles):
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


def evaluate_arm(runtime, core, model, detector, cell, cache_root, thresholds,
                 confidences, arm_name, progress_every=20):
    """Canonical PQ: every annotator entry scored separately, strict IoU > 0.5."""
    base, images, annotations = runtime["base"], runtime["images"], runtime["annotations"]
    held_out, tune, audit, _, by_file = runtime["split"]
    totals = {key: {group: np.zeros(4) for group in ("tune", "audit")}
              for key in (candidate_key(t, c) for t in thresholds for c in confidences)}
    diagnostics = []
    for index, name in enumerate(sorted(held_out), 1):
        views = arm_views(runtime, core, detector, cell, cache_root, name, "train")
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
                key = candidate_key(threshold, confidence)
                for image_id, gt in gt_by_image.items():
                    iou = (mask_utils.iou(kept, gt, [0] * len(gt)).T if kept and gt
                           else np.zeros((len(gt), len(kept))))
                    counts = core.pq_counts(iou, len(gt), len(kept))
                    totals[key][group] += counts
                    diagnostics.append(dict(arm=arm_name, candidate=key, file_name=name,
                                            image_id=image_id, group=group, target="coco",
                                            **dict(zip(COUNTS, counts))))
        if index % progress_every == 0 or index == len(held_out):
            best = max(totals, key=lambda k: pq(totals[k]["tune"]))
            print(f"eval {arm_name} {index}/{len(held_out)} best={best} "
                  f"tune_pq={pq(totals[best]['tune']):.6f}", flush=True)
    return ({key: {group: total.tolist() for group, total in groups.items()}
             for key, groups in totals.items()}, diagnostics)


def arm_views(runtime, core, detector, cell, cache_root, name, split):
    """Frozen caches for the control; freshly generated, signed caches for iou=0.3."""
    if cell["name"] == CONTROL_CELL["name"]:
        return read_views(runtime["inventory"][(split, name)])
    signature = dict(cell=cell, split=split, file_name=name,
                     detector_sha256=runtime["detector_sha256"],
                     resolution=DETECTOR_RESOLUTION,
                     confidence_floor=DETECTOR_CONFIDENCE_FLOOR, views=4)
    path = Path(cache_root) / cache_id(name, split, cell["name"])
    image_root = runtime["base"].CFG.train_dir if split == "train" else runtime["base"].CFG.test_dir
    return generate_views(runtime["tta"], core, detector, image_root / name, cell, path, signature)


def verify_control(counts, reference=CONTROL_REFERENCE):
    """The frozen control must reproduce E015's accepted counts before any claim."""
    observed = counts[reference["candidate"]]
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
        raise ValueError("Results contain a candidate outside the predeclared grid")
    if not results:
        raise ValueError("No candidates to select from")
    return max(sorted(results), key=lambda key: pq(results[key]["tune"]))


def evaluate_bars(selected, results, parents=PARENTS, submit_bar=SUBMIT_BAR,
                  gate=PROMOTION_GATE):
    """Both decisions, computed from predeclared constants only."""
    tune, audit = pq(results[selected]["tune"]), pq(results[selected]["audit"])
    v11 = parents["v11_proposal_centered_e4"]
    authorized = tune >= submit_bar["min_tune_pq"] and audit >= submit_bar["min_audit_pq"]
    promotable = tune - v11["tune_pq"] >= gate["required_tune_delta_vs_v11"]
    if gate["require_audit_at_least_v11"]:
        promotable = promotable and audit >= v11["audit_pq"]
    deltas = {name: dict(tune=tune - value["tune_pq"], audit=audit - value["audit_pq"])
              for name, value in parents.items()}
    return dict(selected=selected, selected_tune_pq=tune, selected_audit_pq=audit,
                deltas_vs_parents=deltas, submit_bar=submit_bar,
                export_authorized=bool(authorized), promotion_gate=gate,
                promotable=bool(promotable),
                recommendation=(
                    "promote" if promotable else
                    "export and measure one submission" if authorized else
                    "composition does not reach the better parent; do not spend a submission"))


def export(runtime, core, model, detector, cell, cache_root, config, output, filename):
    """Identical geometry, ownership, and 180-image coverage as every prior export."""
    base = runtime["base"]
    rows, image_counts = [], {}
    for index, path in enumerate(sorted(base.CFG.test_dir.glob("*.jpeg")), 1):
        views = arm_views(runtime, core, detector, cell, cache_root, path.name, "test")
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
        raise ValueError("Run scripts/prepare_v13_compose.py to stage a pinned run")
    core = sys.modules["optimize_inference"]
    tta = sys.modules["detector_tta"]
    v10 = sys.modules["v10_quality"]
    v11 = sys.modules["v11_refiner"]
    import torch
    verify_grid_matches_frozen_constants(tta)
    # Reuse V11's complete input verification: V6 provenance and artifacts, canonical
    # refiner, canonical annotations, split, crop geometry, all 321 frozen raw caches,
    # frozen-control diagnostics, and exact library versions.
    v11.RAW_CACHE_SHA256 = RAW_CACHE_SHA256
    v11.MODULE_SHA256 = MODULE_SHA256
    runtime = v11.load_runtime(args)
    runtime["tta"] = tta
    runtime["detector_sha256"] = v10.SOURCE_PINS["v6"]["detector"]
    if runtime["provenance"]["sha256"]["yolo_best.pt"] != runtime["detector_sha256"]:
        raise ValueError("V6 detector is not the pinned frozen detector")
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = locate_v11_checkpoint(args.input_root, CHECKPOINT_SHA256)
    checkpoint_identity = verify_v11_checkpoint(checkpoint)
    thresholds, confidences = THRESHOLDS, CONFIDENCES
    held_out = sorted(runtime["split"][0])
    if args.limit:
        thresholds, confidences = (0.40,), (0.30, 0.40)
    identity = digest_json(dict(script_sha256=sha256(Path(__file__)), modules=MODULE_SHA256,
                                checkpoint_sha256=CHECKPOINT_SHA256, cell=CELL,
                                raw_cache_sha256=digest_json(RAW_CACHE_SHA256),
                                source_run=V11_SOURCE_RUN, limit=args.limit))
    manifest_path = args.output / "v13_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous.get("run_identity") != identity:
            raise ValueError("Output directory belongs to a different V13 run")
        if previous.get("complete"):
            print(json.dumps(dict(already_complete=True)), flush=True)
            return
    manifest = dict(experiment="v13_ea_flags_x_v11_refiner", run_identity=identity,
                    cell=CELL, control_cell=CONTROL_CELL, arm=ARM,
                    v11_source_run=V11_SOURCE_RUN, v11_checkpoint_sha256=CHECKPOINT_SHA256,
                    v11_checkpoint_identity=checkpoint_identity,
                    detector_sha256=runtime["detector_sha256"],
                    canonical_refiner_sha256=v10.REFINER_SHA256,
                    control_reference=CONTROL_REFERENCE, parents=PARENTS,
                    submit_bar=SUBMIT_BAR, promotion_gate=PROMOTION_GATE,
                    declared_candidates=list(declared_candidates()),
                    thresholds=list(thresholds), confidences=list(confidences),
                    policy=POLICY, mode=MODE, match_iou=MATCH_IOU, min_prediction_area=MIN_AREA,
                    detector_resolution=DETECTOR_RESOLUTION,
                    detector_confidence_floor=DETECTOR_CONFIDENCE_FLOOR,
                    detector_views=4, crop_views=4, crop_size=runtime["base"].CFG.crop_size,
                    smoke_limit=args.limit, trains=False, submits=False,
                    versions=runtime["versions"], complete=False,
                    agnostic_nms_note="Kept True to match E-A's winning cell exactly. E023 "
                                      "established it is inert under single_cls=True.",
                    note="Composition of two measured orthogonal levers; export is gated on "
                         "SUBMIT_BAR and submission remains a separate manual step.")
    write_json(manifest_path, manifest)
    cache_root = args.output / "proposal_cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    # 1. Frozen control first. Canonical refiner + frozen iou=0.7 caches must
    #    reproduce E015's counts, and its CSV must hash to the accepted 0.38 file.
    #    Nothing downstream is trustworthy unless this passes.
    control_model = v11.build_model(runtime["base"], runtime["refiner"])
    control_results, control_diagnostics = evaluate_arm(
        runtime, core, control_model, None, CONTROL_CELL, cache_root,
        (0.40,), (0.40,), CONTROL)
    control_counts = verify_control(control_results) if not args.limit else None
    print(json.dumps(dict(control_verified=bool(control_counts),
                          control_counts=control_counts)), flush=True)
    control_export = None
    if not args.limit:
        control_export = export(runtime, core, control_model, None, CONTROL_CELL, cache_root,
                                dict(threshold=0.40, confidence=0.40), args.output,
                                "submission_control.csv")
        v10.verify_control_export(args.output / "submission_control.csv")
        print(json.dumps(dict(control_csv_verified=True, **control_export)), flush=True)
    del control_model
    torch.cuda.empty_cache()

    # 2. The composition arm: iou=0.3 proposals + the V11 continuation refiner.
    detector = runtime["base"].YOLO(str(runtime["source_root"] / "yolo_best.pt"))
    model = v11.build_model(runtime["base"], runtime["refiner"], False, checkpoint)
    results, diagnostics = evaluate_arm(runtime, core, model, detector, CELL, cache_root,
                                        thresholds, confidences, CELL["name"])
    frame = pd.DataFrame(control_diagnostics + diagnostics)
    frame.to_csv(args.output / "v13_image_diagnostics.csv", index=False)
    rows = [dict(arm=CELL["name"], candidate=key, tune_pq=pq(value["tune"]),
                 audit_pq=pq(value["audit"]),
                 **{f"tune_{f}": v for f, v in zip(COUNTS, value["tune"])},
                 **{f"audit_{f}": v for f, v in zip(COUNTS, value["audit"])})
            for key, value in results.items()]
    pd.DataFrame(rows).sort_values("tune_pq", ascending=False).to_csv(
        args.output / "v13_candidates.csv", index=False)
    selected = select_candidate(results)
    report = evaluate_bars(selected, results)
    report.update(control_counts=control_counts, control_csv=control_export)
    write_json(args.output / "v13_report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    write_json(args.output / "v13_results.json",
               dict(control={CONTROL: control_results}, composition=results))
    if args.limit:
        print("Smoke test finished; no export, no gate decision", flush=True)
        manifest.update(complete=True, smoke=True)
        write_json(manifest_path, manifest)
        return

    # 3. Export only if the predeclared SUBMIT_BAR authorizes measurement.
    candidate_export = None
    if report["export_authorized"]:
        threshold, confidence = (float(part[1:]) for part in selected.split("_"))
        candidate_export = export(runtime, core, model, detector, CELL, cache_root,
                                  dict(threshold=threshold, confidence=confidence),
                                  args.output, "submission.csv")
        if candidate_export["sha256"] == control_export["sha256"]:
            raise ValueError("Candidate export is byte-identical to the control; nothing measured")
        write_json(args.output / "submission_candidates.json",
                   [dict(control_export, candidate=CONTROL, control_verified=True),
                    dict(candidate_export, candidate=selected,
                         promotable=report["promotable"],
                         tune_pq=report["selected_tune_pq"],
                         audit_pq=report["selected_audit_pq"],
                         requires_independent_csv_validation=True)])
    else:
        print("SUBMIT_BAR not cleared; no CSV exported by design", flush=True)
    del model
    torch.cuda.empty_cache()
    manifest.update(complete=True, selected=selected, report=report,
                    control_export=control_export, candidate_export=candidate_export)
    write_json(manifest_path, manifest)
    print(json.dumps(dict(selected=selected, export_authorized=report["export_authorized"],
                          promotable=report["promotable"],
                          rows=(candidate_export or {}).get("rows"),
                          sha256=(candidate_export or {}).get("sha256"),
                          complete=True), indent=2), flush=True)


if __name__ == "__main__":
    run_compose()
