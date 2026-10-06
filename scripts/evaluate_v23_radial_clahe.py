"""V23 canonical evaluator: transformed detector pixels, raw V17 refiner pixels.

Modes:
  evaluate-tune --arm {control,transformed}  Evaluate one arm on tune only.
  select-tune                              Compare tune reports and authorize audit.
  evaluate-audit                           Evaluate the frozen transformed candidate once.

No mode reads competition test images, exports a CSV, or submits to Kaggle.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

BASE = Path(os.environ.get("V23_BASE", "/mnt/solar-v23-radial-clahe"))
DATA_ROOT = Path(os.environ.get(
    "V23_DATA_ROOT",
    "/home/azureuser/solar-filament-2026/data/raw/filament-segmentation-2026/"
    "MAGFiLO_1.0_Kaggle_2026",
))
RAW_IMAGES = DATA_ROOT / "train/train_images"
TRANSFORMED_IMAGES = BASE / "transformed/train_images"
ANNOTATIONS = DATA_ROOT / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
INPUTS = BASE / "inputs"
RUNTIME = BASE / "runtime"
OUTPUT = BASE / "evaluation"

V6_PROVENANCE = INPUTS / "v6_source/v6_provenance.json"
SOLAR_BASE = INPUTS / "v6_source/solar_base.py"
CANONICAL_REFINER = INPUTS / "best_refiner_coco.pt"
V11_REFINER = INPUTS / "proposal_centered_e4.pt"
CONTROL_DETECTOR = BASE / "outputs/train/control/seed1/final.pt"
ARM_DETECTOR = BASE / "outputs/train/transformed/seed1/final.pt"
CONTROL_REPLICA = BASE / "outputs/train/control/seed2/final.pt"
ARM_REPLICA = BASE / "outputs/train/transformed/seed2/final.pt"


ARM_PATHS = {
    "control": CONTROL_DETECTOR,
    "transformed": ARM_DETECTOR,
    "control-replica": CONTROL_REPLICA,
    "transformed-replica": ARM_REPLICA,
}
ANNOTATION_SHA256 = "5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1"
PROVENANCE_SHA256 = "ceb09f980d5be195db35f2954c26fd19ea7ffc0f318b2bae5b7c803504b17122"
CANONICAL_REFINER_SHA256 = "8d00805318233a4196c706c1ec4eba9136ca1cb80d62298f082e82f7a2d01fb1"
V11_REFINER_SHA256 = "fae496b9d00819f49c5278e81e41a7917b31d18ea80f827e162746d840d14cd2"
SOURCE_DETECTOR_SHA256 = "ae476e663039fbbbfda77ec519635ca15e2d50d1fe9f1af54735f84dbdc251ef"
V17_TUNE_PQ = 0.448702717279432
V17_AUDIT_PQ = 0.4707634285309831
TUNE_REQUIRED = 0.458702717279432
MATCHED_CONTROL_DELTA = 0.010

CELLS = (
    {"name": "iou030", "retina_masks": True, "agnostic_nms": True, "iou": 0.3},
    {"name": "iou070", "retina_masks": True, "agnostic_nms": True, "iou": 0.7},
)
THRESHOLDS = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60)
CONFIDENCES = (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55)
DETECTOR_FLOOR = 0.10
MATCH_IOU = 0.5
MIN_AREA = 20


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_json(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def pq(counts) -> float:
    denominator = counts[1] + 0.5 * (counts[2] + counts[3])
    return float(counts[0] / denominator) if denominator else 0.0


def metric_record(counts) -> dict:
    denominator = counts[1] + 0.5 * (counts[2] + counts[3])
    return {
        "pq": pq(counts),
        "sq": float(counts[0] / counts[1]) if counts[1] else 0.0,
        "rq": float(counts[1] / denominator) if denominator else 0.0,
        "tp_iou": float(counts[0]), "tp": int(counts[1]),
        "fp": int(counts[2]), "fn": int(counts[3]),
    }


def candidate_key(cell: str, threshold: float, confidence: float) -> str:
    return f"{cell}/t{threshold:.2f}_c{confidence:.2f}"


def parse_candidate(value: str) -> tuple[dict, float, float]:
    cell_name, settings = value.split("/")
    cell = next((entry for entry in CELLS if entry["name"] == cell_name), None)
    if cell is None:
        raise ValueError(f"unknown cell {cell_name}")
    threshold, confidence = (float(part[1:]) for part in settings.split("_"))
    if threshold not in THRESHOLDS or confidence not in CONFIDENCES:
        raise ValueError("candidate outside frozen grid")
    return cell, threshold, confidence


def load_modules():
    sys.path.insert(0, str(RUNTIME))
    import detector_tta as tta
    import optimize_inference as core
    import v11_refiner as v11
    import v17_compose as v17
    return core, tta, v11, v17


def verify_inputs() -> dict:
    expected = {
        V6_PROVENANCE: PROVENANCE_SHA256,
        CANONICAL_REFINER: CANONICAL_REFINER_SHA256,
        V11_REFINER: V11_REFINER_SHA256,
        ANNOTATIONS: ANNOTATION_SHA256,
    }
    for path, digest in expected.items():
        if not path.is_file() or sha256(path) != digest:
            raise RuntimeError(f"missing or changed pinned input {path}")
    preprocessing = json.loads((BASE / "preprocessing_manifest.json").read_text())
    if preprocessing["images"] != 707:
        raise RuntimeError("transformed image population mismatch")
    checkpoints = {}
    for arm, path in ARM_PATHS.items():
        if not path.is_file():
            raise RuntimeError(f"missing trained {arm} detector")
        checkpoints[arm] = sha256(path)
    return {"preprocessing": preprocessing, "checkpoints": checkpoints}


def runtime_state():
    import torch

    core, tta, v11, v17 = load_modules()
    pins = verify_inputs()
    base = core.load_base(SOLAR_BASE)
    base.CFG.root = DATA_ROOT
    base.CFG.train_dir = RAW_IMAGES
    base.CFG.test_dir = DATA_ROOT / "test/test_images"
    base.CFG.train_json = ANNOTATIONS
    base.CFG.imgsz = 1024
    base.CFG.crop_size = 256
    base.CFG.crop_context = 1.8
    base.CFG.min_crop = 96
    base.seed_everything(42)
    provenance = json.loads(V6_PROVENANCE.read_text())
    _, images, annotations = base.load_coco()
    split = core.validation_split(base, images, annotations)
    held, tune, audit, _, by_file = split
    core.verify_split(provenance, held, images)
    if (len(held), len(tune), len(audit), sum(map(len, by_file.values()))) != (141, 94, 47, 217):
        raise RuntimeError("canonical split mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("V23 evaluation requires CUDA")
    model = v11.build_model(base, CANONICAL_REFINER, False, V11_REFINER)
    return {
        "base": base, "core": core, "tta": tta, "v11": v11, "v17": v17,
        "images": images, "annotations": annotations, "split": split,
        "refiner": model, "pins": pins,
    }


def cache_path(arm: str, detector_sha: str, name: str, cell: dict) -> Path:
    preprocessing = "raw" if arm == "control" else json.loads(
        (BASE / "preprocessing_manifest.json").read_text()
    )["transform_id"]
    token = f"{arm}|{preprocessing}|{detector_sha}|{cell['name']}|{name}|1024|0.10"
    return OUTPUT / "proposal_cache" / arm / (hashlib.sha256(token.encode()).hexdigest()[:20] + ".json.gz")


def generate_views(runtime: dict, detector, image_path: Path, cell: dict,
                   destination: Path, signature: dict):
    marker = destination.with_suffix(".signature.json")
    if destination.is_file():
        recorded = json.loads(marker.read_text())
        if recorded["signature"] != signature or recorded["sha256"] != sha256(destination):
            raise RuntimeError(f"stale proposal cache {destination}")
        return read_views(destination)
    image = cv2.imread(str(image_path))
    if image is None or image.shape[:2] != (2048, 2048):
        raise RuntimeError(f"cannot read detector image {image_path}")
    views = []
    for axes in ((), (1,), (0,), (0, 1)):
        source = str(image_path) if not axes else runtime["tta"].flip_array(image, axes)
        result = detector.predict(
            source, imgsz=1024, conf=DETECTOR_FLOOR,
            agnostic_nms=cell["agnostic_nms"], retina_masks=cell["retina_masks"],
            iou=cell["iou"], verbose=False,
        )[0]
        rles, scores = [], []
        if result.masks is not None:
            scores = result.boxes.conf.detach().cpu().numpy().tolist()
            for mask in result.masks.data:
                restored = runtime["tta"].flip_array(mask.cpu().numpy().astype(np.uint8), axes)
                if restored.shape != (2048, 2048):
                    raise RuntimeError("detector mask geometry changed")
                rles.append(runtime["core"].encode(restored))
        keep = [index for index, rle in enumerate(rles) if mask_utils.area(rle) > 0]
        views.append(([rles[index] for index in keep], [scores[index] for index in keep]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(destination, "wt") as handle:
        json.dump([{
            "rles": [{"size": list(rle["size"]), "counts": rle["counts"].decode()}
                     for rle in rles],
            "scores": scores,
        } for rles, scores in views], handle)
    write_json(marker, {"signature": signature, "sha256": sha256(destination)})
    return views


def read_views(path: Path):
    with gzip.open(path, "rt") as handle:
        raw = json.load(handle)
    views = []
    if len(raw) != 4:
        raise RuntimeError("proposal cache does not contain four views")
    for view in raw:
        rles = [{"size": list(item["size"]), "counts": item["counts"].encode()}
                for item in view["rles"]]
        if len(rles) != len(view["scores"]):
            raise RuntimeError("proposal cache score mismatch")
        views.append((rles, view["scores"]))
    return views


def refine(runtime: dict, views, raw_image: Path, thresholds: tuple[float, ...]):
    proposals, scores = runtime["tta"].cluster_views(views, "flip_consensus", MATCH_IOU)
    if not proposals:
        return {threshold: ([], np.asarray([])) for threshold in thresholds}, proposals
    masks = [mask_utils.decode(rle) for rle in proposals]
    gray = cv2.imread(str(raw_image), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError(f"cannot read raw refiner image {raw_image}")
    bounds, _, probabilities = runtime["core"].crop_probabilities(
        runtime["base"], runtime["refiner"], gray, masks, True,
    )
    result = {
        threshold: runtime["core"].clean_refined_predictions(
            masks, scores, bounds, probabilities, threshold, "refined_constrained",
        )
        for threshold in thresholds
    }
    return result, proposals


def add_raw_matches(values: list[float], proposals: list[dict], gt: list[dict]) -> None:
    if not gt:
        return
    if not proposals:
        values.extend([0.0] * len(gt)); return
    iou = np.asarray(mask_utils.iou(proposals, gt, [0] * len(gt)), dtype=np.float64)
    values.extend(iou.max(axis=0).tolist() if iou.size else [0.0] * len(gt))


def evaluate_arm(arm: str, group: str, selected: str | None = None) -> dict:
    from ultralytics import YOLO

    if arm not in ARM_PATHS or group not in ("tune", "audit"):
        raise ValueError("invalid arm/group")
    runtime = runtime_state()
    held, tune, audit, _, by_file = runtime["split"]
    names = sorted(tune if group == "tune" else audit)
    detector_path = ARM_PATHS[arm]
    detector_sha = sha256(detector_path)
    is_transformed = arm.startswith("transformed")
    detector_images = TRANSFORMED_IMAGES if is_transformed else RAW_IMAGES
    detector = YOLO(str(detector_path))
    if selected:
        cell, threshold, confidence = parse_candidate(selected)
        cells, thresholds, confidences = (cell,), (threshold,), (confidence,)
    else:
        cells, thresholds, confidences = CELLS, THRESHOLDS, CONFIDENCES
    totals = {
        candidate_key(cell["name"], threshold, confidence): np.zeros(4, dtype=np.float64)
        for cell in cells for threshold in thresholds for confidence in confidences
    }
    raw_by_cell = {cell["name"]: [] for cell in cells}
    diagnostics = []
    started = time.monotonic()
    for position, name in enumerate(names, 1):
        for cell in cells:
            signature = {
                "arm": arm, "detector_sha256": detector_sha,
                "transform_id": runtime["pins"]["preprocessing"]["transform_id"] if is_transformed else "raw",
                "cell": cell, "file_name": name, "views": 4,
            }
            destination = cache_path(arm, detector_sha, name, cell)
            views = generate_views(runtime, detector, detector_images / name, cell,
                                   destination, signature)
            refined, proposals = refine(runtime, views, RAW_IMAGES / name, thresholds)
            for image_id in sorted(by_file[name]):
                info = runtime["images"][image_id]
                gt = [runtime["core"].annotation_rle(info, entry)
                      for entry in runtime["annotations"].get(image_id, [])]
                gt = [rle for rle in gt if mask_utils.area(rle) > 0]
                add_raw_matches(raw_by_cell[cell["name"]], proposals, gt)
                for threshold in thresholds:
                    rles, scores = refined[threshold]
                    for confidence in confidences:
                        keep = np.asarray(scores) >= confidence if len(scores) else np.zeros(0, bool)
                        predictions = [rle for rle, flag in zip(rles, keep) if flag]
                        iou = (mask_utils.iou(predictions, gt, [0] * len(gt)).T
                               if predictions and gt else np.zeros((len(gt), len(predictions))))
                        counts = runtime["core"].pq_counts(iou, len(gt), len(predictions))
                        key = candidate_key(cell["name"], threshold, confidence)
                        totals[key] += counts
                        diagnostics.append({
                            "arm": arm, "group": group, "file_name": name,
                            "image_id": image_id, "candidate": key,
                            "iou_sum": counts[0], "tp": counts[1],
                            "fp": counts[2], "fn": counts[3],
                        })
        if position % 20 == 0 or position == len(names):
            best = max(totals, key=lambda key: pq(totals[key]))
            print(f"{arm}/{group} {position}/{len(names)} best={best} pq={pq(totals[best]):.6f}", flush=True)
    raw = {}
    for cell, values in raw_by_cell.items():
        array = np.asarray(values, dtype=np.float64)
        raw[cell] = {
            "instances": int(len(array)), "recall_gt_0_5": float((array > 0.5).mean()),
            "mean_best_iou": float(array.mean()), "max_best_iou": float(array.max()),
            "p90_best_iou": float(np.percentile(array, 90)),
            "missed": int((array <= 0.5).sum()),
        }
    result = {
        "status": "complete", "arm": arm, "group": group,
        "detector": str(detector_path), "detector_sha256": detector_sha,
        "detector_pixels": runtime["pins"]["preprocessing"]["transform_id"] if is_transformed else "raw",
        "refiner_pixels": "raw", "selected_only": selected,
        "files": len(names), "records": sum(len(by_file[name]) for name in names),
        "results": {key: metric_record(value) for key, value in totals.items()},
        "raw_proposals": raw, "elapsed_seconds": time.monotonic() - started,
    }
    output = OUTPUT / f"{arm}_{group}.json"
    write_json(output, result)
    pd.DataFrame(diagnostics).to_csv(OUTPUT / f"{arm}_{group}_diagnostics.csv", index=False)
    print(json.dumps({k: result[k] for k in ("arm", "group", "detector_sha256", "raw_proposals")}, indent=2))
    return result


def select_tune() -> dict:
    control = json.loads((OUTPUT / "control_tune.json").read_text())
    arm = json.loads((OUTPUT / "transformed_tune.json").read_text())
    control_replica = json.loads((OUTPUT / "control-replica_tune.json").read_text())
    arm_replica = json.loads((OUTPUT / "transformed-replica_tune.json").read_text())
    if any(report["group"] != "tune" for report in (control, arm, control_replica, arm_replica)):
        raise RuntimeError("selection inputs are not tune reports")
    control_selected = max(sorted(control["results"]), key=lambda key: control["results"][key]["pq"])
    arm_selected = max(sorted(arm["results"]), key=lambda key: arm["results"][key]["pq"])
    control_replica_selected = max(sorted(control_replica["results"]), key=lambda key: control_replica["results"][key]["pq"])
    arm_replica_selected = max(sorted(arm_replica["results"]), key=lambda key: arm_replica["results"][key]["pq"])
    control_pq = control["results"][control_selected]["pq"]
    arm_pq = arm["results"][arm_selected]["pq"]
    replica_control_pq = control_replica["results"][control_replica_selected]["pq"]
    replica_arm_pq = arm_replica["results"][arm_replica_selected]["pq"]
    replica_delta = replica_arm_pq - replica_control_pq
    audit_authorized = (
        arm_pq >= TUNE_REQUIRED
        and arm_pq >= control_pq + MATCHED_CONTROL_DELTA
        and replica_delta > 0
    )
    report = {
        "status": "complete", "selected_on": "tune only",
        "matched_control": {"candidate": control_selected, **control["results"][control_selected]},
        "transformed": {"candidate": arm_selected, **arm["results"][arm_selected]},
        "replication": {
            "control": {"candidate": control_replica_selected, **control_replica["results"][control_replica_selected]},
            "transformed": {"candidate": arm_replica_selected, **arm_replica["results"][arm_replica_selected]},
            "delta": replica_delta,
            "direction_agrees": replica_delta > 0,
        },
        "v17_tune_pq": V17_TUNE_PQ, "required_tune_pq": TUNE_REQUIRED,
        "required_delta_vs_control": MATCHED_CONTROL_DELTA,
        "delta_vs_v17": arm_pq - V17_TUNE_PQ,
        "delta_vs_matched_control": arm_pq - control_pq,
        "raw_proposal_delta": {
            cell: {
                key: arm["raw_proposals"][cell][key] - control["raw_proposals"][cell][key]
                for key in ("recall_gt_0_5", "mean_best_iou", "max_best_iou")
            } for cell in ("iou030", "iou070")
        },
        "audit_authorized": bool(audit_authorized),
        "action": "evaluate transformed candidate on audit once" if audit_authorized else "close V23 without audit/export",
    }
    write_json(OUTPUT / "tune_selection.json", report)
    print(json.dumps(report, indent=2))
    return report


def evaluate_audit() -> dict:
    selection_path = OUTPUT / "tune_selection.json"
    selection = json.loads(selection_path.read_text())
    if not selection["audit_authorized"]:
        raise RuntimeError("tune gates did not authorize audit")
    final_path = OUTPUT / "v23_evaluation.json"
    if final_path.is_file():
        print(final_path.read_text()); return json.loads(final_path.read_text())
    marker = OUTPUT / "audit_opened.json"
    if marker.exists():
        raise RuntimeError("audit was already opened without a completed result")
    candidate = selection["transformed"]["candidate"]
    write_json(marker, {"candidate": candidate, "checkpoint_sha256": sha256(ARM_DETECTOR),
                        "opened_unix": time.time()})
    audit = evaluate_arm("transformed", "audit", candidate)
    metrics = audit["results"][candidate]
    export_authorized = metrics["pq"] >= V17_AUDIT_PQ
    result = {
        "status": "complete", "tune_selection": selection,
        "audit": {"candidate": candidate, **metrics},
        "required_audit_pq": V17_AUDIT_PQ,
        "export_authorized": bool(export_authorized),
        "action": "prepare test export for explicit authorization" if export_authorized else "close V23 no export",
    }
    write_json(final_path, result)
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("evaluate-tune", "select-tune", "evaluate-audit"))
    parser.add_argument("--arm", choices=tuple(ARM_PATHS))
    args = parser.parse_args()
    cv2.setNumThreads(2)
    if args.mode == "evaluate-tune":
        if not args.arm:
            parser.error("evaluate-tune requires --arm")
        evaluate_arm(args.arm, "tune")
    elif args.mode == "select-tune":
        select_tune()
    else:
        evaluate_audit()


if __name__ == "__main__":
    main()
