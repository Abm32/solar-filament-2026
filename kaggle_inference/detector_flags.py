"""E-A: cheap inference-flag ablation on the frozen detector (retina_masks,
agnostic_nms, box iou).

Protocol, declared before any score is observed (docs/optimization_plan_055_2026-09-06.md):

  * Sweep retina_masks {False,True} x agnostic_nms {False,True} x box iou
    {default 0.7, aggressive 0.3}, 2x2x2 = 8 cells. No retrain, no threshold-grid
    expansion beyond the existing LOCAL_THRESHOLDS/LOCAL_CONFIDENCES already used
    by detector_tta.py.
  * Everything else stays exactly as accepted: same V6 detector + COCO refiner
    checkpoints (hash-verified), same flip_consensus policy, same crop-refiner
    constrained postprocess, same 94 tune / 47 audit canonical split.
  * The accepted cell is retina_masks=True, agnostic_nms=True, iou=0.7 (the
    Ultralytics default) -- this is what detector_tta.py already hardcodes. It
    is included in the grid as CELL_CONTROL so the ablation reproduces the
    accepted numbers as one of its own rows, not as a separate code path. The
    control is checked against CONTROL_REFERENCE, the flip_consensus E015 counts
    (candidate t0.40/c0.40), because this ablation runs the flip_consensus policy.
  * Success gate: >= +0.003 tune PQ over the control cell, nonnegative audit
    delta. Anything else is measured and dropped, matching E-A's declared gate.

Mechanism: SQ primarily (finer mask raster from retina_masks), minor RQ
(near-misses crossing 0.5 via agnostic_nms suppressing overlapping proposals,
or a lower box iou threshold suppressing more duplicate boxes pre-mask).

Nothing here submits. Export is a separate pinned step gated on whichever cell
(if any) clears the +0.003/nonnegative-audit bar, and on the frozen control
reproducing the accepted 0.38 CSV byte-for-byte -- exactly as v11/v12 required.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mask_utils

import detector_tta as tta
import optimize_inference as core

CELLS = (
    dict(name="control_retina_agnostic_iou070", retina_masks=True, agnostic_nms=True, iou=0.7),
    dict(name="retina_agnostic_iou030", retina_masks=True, agnostic_nms=True, iou=0.3),
    dict(name="retina_noagnostic_iou070", retina_masks=True, agnostic_nms=False, iou=0.7),
    dict(name="retina_noagnostic_iou030", retina_masks=True, agnostic_nms=False, iou=0.3),
)
CONTROL_CELL = "control_retina_agnostic_iou070"
GATE = dict(required_tune_delta=0.003, require_nonnegative_audit=True)
# Frozen E015 (== accepted 0.38) reference for the control cell. E-A's control runs
# the flip_consensus policy, so it must be checked against the flip_consensus arm of
# the accepted detector-TTA run, NOT against the single-view sweep that happens to
# sit next to best_refiner_coco.pt. These are the same aggregate counts v12_resolution
# pins as V6_REFERENCE, verified locally against
# outputs/kaggle_detector_tta/flip_consensus/inference_image_diagnostics.csv.
CONTROL_REFERENCE = dict(
    candidate="r1024_refined_constrained_t0.40_tta1_c0.40",
    tune=[426.5473399071961, 626, 211, 508],
    audit=[182.75261599994468, 265, 88, 194],
)


def cell_by_name(name):
    matches = [cell for cell in CELLS if cell["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"Unknown or ambiguous cell: {name}")
    return matches[0]


def proposal_views_for_cell(base, detector, image_path, cell, cache_root):
    """Same four-view flip cache as detector_tta.proposal_views, keyed per cell.

    Duplicated rather than parameterizing the frozen function so the accepted
    0.38/0.39 code path in detector_tta.py is never edited by this ablation.
    """
    cache_id = hashlib.sha256(str(image_path).encode()).hexdigest()[:16]
    cache = cache_root / f"{cache_id}.json.gz"
    if cache.exists():
        import gzip
        with gzip.open(cache, "rt") as stream:
            serialized = json.load(stream)
        return [([dict(size=r["size"], counts=r["counts"].encode()) for r in view["rles"]],
                 view["scores"]) for view in serialized]
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read {image_path}")
    views = []
    for axes in ((), (1,), (0,), (0, 1)):
        result = detector.predict(
            str(image_path) if not axes else tta.flip_array(image, axes),
            imgsz=1024, conf=0.1, agnostic_nms=cell["agnostic_nms"],
            retina_masks=cell["retina_masks"], iou=cell["iou"], verbose=False)[0]
        rles, scores = [], []
        if result.masks is not None:
            scores = result.boxes.conf.detach().cpu().numpy().tolist()
            for mask in result.masks.data:
                restored = tta.flip_array(mask.cpu().numpy().astype(np.uint8), axes)
                if restored.shape != image.shape[:2]:
                    raise ValueError("Detector masks must be in original image coordinates")
                rles.append(core.encode(restored))
        views.append((rles, scores))
    import gzip
    with gzip.open(cache, "wt") as stream:
        json.dump([dict(rles=[dict(size=r["size"], counts=r["counts"].decode()) for r in rles],
                        scores=scores) for rles, scores in views], stream)
    return views


def evaluate_cell(base, detector, refiner, images, annotations, output, cell, limit):
    """Run the frozen flip_consensus + crop-refiner path for one flag cell."""
    cell_cache = output / "proposal_cache" / cell["name"]
    cell_cache.mkdir(parents=True, exist_ok=True)

    def cell_prediction_variants(base_module, detector_module, refiner_module, image_path,
                                  resolution, selected=None):
        if resolution != 1024:
            raise ValueError("This ablation fixes detector resolution at 1024")
        rles, scores = tta.cluster_views(
            proposal_views_for_cell(base_module, detector_module, image_path, cell, cell_cache),
            "flip_consensus")
        masks = [mask_utils.decode(rle) for rle in rles]
        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f"Cannot read {image_path}")
        bounds, _, averaged = core.crop_probabilities(base_module, refiner_module, gray, masks, True)
        thresholds = (selected["threshold"],) if selected is not None else tta.LOCAL_THRESHOLDS
        for threshold in thresholds:
            cleaned, kept_scores = core.clean_refined_predictions(
                masks, scores, bounds, averaged, threshold, "refined_constrained")
            yield "refined_constrained", threshold, True, cleaned, kept_scores

    core.prediction_variants = cell_prediction_variants
    core.CONFIDENCES = tta.LOCAL_CONFIDENCES
    cell_output = output / cell["name"]
    cell_output.mkdir(exist_ok=True)
    core.sweep(base, detector, refiner, images, annotations, cell_output, [1024], limit)
    candidates = pd.read_csv(cell_output / "inference_candidates.csv")
    candidates["cell"] = cell["name"]
    return candidates


def run_experiment():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working"))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("limit must be nonnegative")
    sources = list(Path("/kaggle/input").rglob("v6_provenance.json"))
    refiners = list(Path("/kaggle/input").rglob("best_refiner_coco.pt"))
    if len(sources) != 1 or len(refiners) != 1:
        raise ValueError(f"Expected one V6 source and one COCO refiner: {sources}, {refiners}")
    source, baseline = sources[0].parent, refiners[0].parent
    provenance = json.loads(sources[0].read_text())
    for name, expected in provenance["sha256"].items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Source hash mismatch: {name}")
    if hashlib.sha256(refiners[0].read_bytes()).hexdigest() != tta.REFINER_SHA256:
        raise ValueError("Refiner differs from accepted 0.36 checkpoint")
    args.output.mkdir(parents=True, exist_ok=True)
    base = core.load_base(source / "solar_base.py")
    base.seed_everything(42)
    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    _, images, annotations = base.load_coco()
    held_out, tune, audit, best_ids, _ = core.validation_split(base, images, annotations)
    core.verify_split(provenance, held_out, images)
    previous = json.loads((baseline / "inference_manifest.json").read_text())
    target = "COCO polygons; rounded OpenCV only for legacy comparison"
    for field, expected in (("tune_files", sorted(tune)), ("audit_files", sorted(audit)),
                            ("target_rasterization", target)):
        if previous[field] != expected:
            raise ValueError(f"Baseline mismatch: {field}")
    if hashlib.sha256(base.CFG.train_json.read_bytes()).hexdigest() != provenance["annotation_sha256"]:
        raise ValueError("Annotation source changed")
    manifest = dict(experiment="e_a_detector_flags", source=provenance,
                    detector=dict(kernel=provenance["kernel"], sha256=provenance["sha256"]["yolo_best.pt"]),
                    helper_sha256=hashlib.sha256((source / "solar_base.py").read_bytes()).hexdigest(),
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    refiner_checkpoint_sha256=tta.REFINER_SHA256,
                    baseline_submission=56026663, baseline_public_score=0.38,
                    target_rasterization=target, cells=CELLS, control_cell=CONTROL_CELL, gate=GATE,
                    control_reference=CONTROL_REFERENCE,
                    tune_files=sorted(tune), audit_files=sorted(audit), smoke_limit=args.limit,
                    thresholds=tta.LOCAL_THRESHOLDS, confidences=tta.LOCAL_CONFIDENCES,
                    resolutions=[1024], mask_match_iou=0.5,
                    versions={name: base.version(name) for name in
                              ("ultralytics", "torch", "torchvision", "pycocotools", "numpy")})
    (args.output / "inference_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    detector = base.YOLO(str(source / "yolo_best.pt"))
    refiner = base.CropUNet(pretrained=False).to("cuda:0")
    refiner.load_state_dict(torch.load(refiners[0], map_location="cuda:0", weights_only=True))
    refiner.eval()
    all_candidates = []
    for cell in CELLS:
        frame = evaluate_cell(base, detector, refiner, images, annotations, args.output, cell, args.limit)
        all_candidates.append(frame)
        if cell["name"] == CONTROL_CELL and not args.limit:
            counts = verify_control_reproduces_accepted(args.output / cell["name"])
            (args.output / "e_a_control_verification.json").write_text(
                json.dumps(dict(reference=CONTROL_REFERENCE, observed=counts), indent=2) + "\n")
            print("Control cell reproduces the accepted 0.38 tune/audit counts", flush=True)
    combined = pd.concat(all_candidates, ignore_index=True)
    combined.to_csv(args.output / "e_a_candidates.csv", index=False)
    if args.limit:
        print("Smoke test finished; no gate evaluated", flush=True)
        return
    report = evaluate_gate(combined)
    (args.output / "e_a_gate_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def verify_control_reproduces_accepted(cell_output, reference=CONTROL_REFERENCE):
    """The control cell must reproduce E015's accepted tune/audit counts exactly.

    Aggregates the canonical COCO-target rows of the control candidate, matching
    v12_resolution.verify_control. Only ``target == "coco"`` rows count: the
    ``rounded_legacy`` rows are a separate legacy comparison and are excluded from
    tune/audit PQ by construction.
    """
    diagnostics = pd.read_csv(cell_output / "inference_image_diagnostics.csv")
    rows = diagnostics[(diagnostics.candidate == reference["candidate"])
                       & (diagnostics.target == "coco")]
    if rows.empty:
        raise ValueError(f"Control candidate {reference['candidate']} is missing from the sweep")
    actual = {}
    for group in ("tune", "audit"):
        subset = rows[rows.group == group]
        if subset.empty:
            raise ValueError(f"Control candidate has no {group} rows")
        actual[group] = [float(subset.iou_sum.sum()), int(subset.tp.sum()),
                         int(subset.fp.sum()), int(subset.fn.sum())]
    for group in ("tune", "audit"):
        expected = reference[group]
        if abs(expected[0] - actual[group][0]) > 1e-6 or actual[group][1:] != expected[1:]:
            raise ValueError(f"Control {group} counts differ from the frozen E015 reference: "
                             f"{actual[group]} vs {expected}")
    return actual


def evaluate_gate(combined):
    """Best (tune_pq-maximizing) candidate per cell, control-relative gate per cell."""
    control_best = combined[combined.cell == CONTROL_CELL].sort_values(
        "tune_pq", ascending=False).iloc[0]
    rows = []
    for cell_name, group in combined.groupby("cell"):
        best = group.sort_values("tune_pq", ascending=False).iloc[0]
        tune_delta = float(best.tune_pq - control_best.tune_pq)
        audit_delta = float(best.audit_pq - control_best.audit_pq)
        passed = (cell_name != CONTROL_CELL and tune_delta >= GATE["required_tune_delta"]
                  and (not GATE["require_nonnegative_audit"] or audit_delta >= 0))
        rows.append(dict(cell=cell_name, candidate=best.candidate, tune_pq=float(best.tune_pq),
                         audit_pq=float(best.audit_pq), tune_delta=tune_delta,
                         audit_delta=audit_delta, gate_passed=bool(passed)))
    rows.sort(key=lambda row: -row["tune_delta"])
    any_passed = any(row["gate_passed"] for row in rows)
    return dict(control_cell=CONTROL_CELL, control_tune_pq=float(control_best.tune_pq),
                control_audit_pq=float(control_best.audit_pq), gate=GATE,
                cells=rows, any_cell_passed_gate=any_passed,
                recommendation=("promote " + rows[0]["cell"] if any_passed else
                                "no cell clears the gate; E-A measured and dropped"))


if __name__ == "__main__":
    run_experiment()
