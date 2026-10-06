"""V26 native sliced proposal discovery, tune-first and audit-gated.

Modes:
  verify                 Verify pinned assets, split, runtime, and frozen geometry.
  smoke --shard N        Evaluate one deterministic tune file for shard N.
  tune --shard N         Evaluate shard N of the 94 tune files.
  merge-tune             Merge four shards, verify V17 control, and apply gates.
  audit --shard N        Evaluate the frozen hybrid candidate only when authorized.
  merge-audit            Merge authorized audit shards and apply export gate.

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

BASE = Path(os.environ.get("V26_BASE", "/mnt/solar-v26-sliced"))
DATA_ROOT = Path(os.environ.get(
    "V26_DATA_ROOT",
    "/home/azureuser/solar-filament-2026/data/raw/filament-segmentation-2026/"
    "MAGFiLO_1.0_Kaggle_2026",
))
INPUTS = BASE / "inputs"
RUNTIME = BASE / "runtime"
OUTPUT = BASE / "evaluation"
RAW_IMAGES = DATA_ROOT / "train/train_images"
ANNOTATIONS = DATA_ROOT / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
PROVENANCE = INPUTS / "v6_source/v6_provenance.json"
SOLAR_BASE = INPUTS / "v6_source/solar_base.py"
DETECTOR = INPUTS / "v17_detector.pt"
CANONICAL_REFINER = INPUTS / "best_refiner_coco.pt"
V11_REFINER = INPUTS / "proposal_centered_e4.pt"
HISTORICAL_DIAGNOSTICS = INPUTS / "proposal_diagnostics.csv"

ANNOTATION_SHA256 = "5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1"
PROVENANCE_SHA256 = "ceb09f980d5be195db35f2954c26fd19ea7ffc0f318b2bae5b7c803504b17122"
DETECTOR_SHA256 = "ae476e663039fbbbfda77ec519635ca15e2d50d1fe9f1af54735f84dbdc251ef"
CANONICAL_REFINER_SHA256 = "8d00805318233a4196c706c1ec4eba9136ca1cb80d62298f082e82f7a2d01fb1"
V11_REFINER_SHA256 = "fae496b9d00819f49c5278e81e41a7917b31d18ea80f827e162746d840d14cd2"
HISTORICAL_DIAGNOSTICS_SHA256 = "aecdc16648cc7f6e8959770581b6ae8c910403279720378da35842fe7d24e5ad"

TILE_SIZE = 1024
TILE_STARTS = (0, 512, 1024)
TILE_CENTERS = (512, 1024, 1536)
OWNER_BOUNDARIES = (768, 1280)
FULL_INFERENCE_SIZE = 1024
TILE_INFERENCE_SIZE = 1024
DETECTOR_FLOOR = 0.10
NMS_IOU = 0.50
MATCH_IOU = 0.50
MIN_AREA = 20
SHARDS = 4
POLICY = "flip_consensus"
MODE = "refined_constrained"
CELLS = (
    {"name": "iou030", "retina_masks": True, "agnostic_nms": True, "iou": 0.3},
    {"name": "iou070", "retina_masks": True, "agnostic_nms": True, "iou": 0.7},
)
THRESHOLDS = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60)
CONFIDENCES = (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55)
ARMS = ("control", "sliced", "hybrid")
COUNTS = ("iou_sum", "tp", "fp", "fn")
V17_TUNE_PQ = 0.448702717279432
V17_AUDIT_PQ = 0.4707634285309831
TUNE_REQUIRED = V17_TUNE_PQ + 0.010
MATCHED_CONTROL_DELTA = 0.010
RAW_RECALL_DELTA = 0.020
HISTORICAL_RESCUE_REQUIRED = 15
BOTTOM_RECALL_DELTA = 0.030
BOTTOM_AREA_MAX = 800.0
EXPECTED_CONTROL_CANDIDATE = "control/iou030/t0.45_c0.30"
EXPECTED_CONTROL_TUNE = [481.00931292355114, 704, 306, 430]
EXPECTED_TUNE = {"files": 94, "records": 150, "gt": 1134}
EXPECTED_AUDIT = {"files": 47, "records": 67, "gt": 459}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_json(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def serialized_rle(rle: dict) -> dict:
    counts = rle["counts"].decode() if isinstance(rle["counts"], bytes) else rle["counts"]
    return {"size": list(rle["size"]), "counts": counts}


def restored_rle(raw: dict) -> dict:
    counts = raw["counts"].encode() if isinstance(raw["counts"], str) else raw["counts"]
    return {"size": list(raw["size"]), "counts": counts}


def pq(counts) -> float:
    denominator = counts[1] + 0.5 * (counts[2] + counts[3])
    return float(counts[0] / denominator) if denominator else 0.0


def metric_record(counts) -> dict:
    denominator = counts[1] + 0.5 * (counts[2] + counts[3])
    return {
        "pq": pq(counts),
        "sq": float(counts[0] / counts[1]) if counts[1] else 0.0,
        "rq": float(counts[1] / denominator) if denominator else 0.0,
        **dict(zip(COUNTS, (float(counts[0]), int(counts[1]), int(counts[2]), int(counts[3])))),
    }


def candidate_key(arm: str, cell: str, threshold: float, confidence: float) -> str:
    if arm not in ARMS or cell not in {entry["name"] for entry in CELLS}:
        raise ValueError("candidate outside frozen arms/cells")
    if threshold not in THRESHOLDS or confidence not in CONFIDENCES:
        raise ValueError("candidate outside frozen grid")
    return f"{arm}/{cell}/t{threshold:.2f}_c{confidence:.2f}"


def parse_candidate(value: str) -> tuple[str, dict, float, float]:
    arm, cell_name, settings = value.split("/")
    cell = next((entry for entry in CELLS if entry["name"] == cell_name), None)
    if arm not in ARMS or cell is None:
        raise ValueError("unknown candidate")
    threshold, confidence = (float(part[1:]) for part in settings.split("_"))
    candidate_key(arm, cell_name, threshold, confidence)
    return arm, cell, threshold, confidence


def owner_axis(coordinate: float) -> int:
    if not np.isfinite(coordinate) or not 0 <= coordinate < 2048:
        raise ValueError(f"invalid global centroid coordinate {coordinate}")
    if coordinate < OWNER_BOUNDARIES[0]:
        return 0
    if coordinate < OWNER_BOUNDARIES[1]:
        return 1
    return 2


def owner_origin(cx: float, cy: float) -> tuple[int, int]:
    return TILE_STARTS[owner_axis(cx)], TILE_STARTS[owner_axis(cy)]


def mask_centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.where(mask)
    if not len(xs):
        raise ValueError("empty mask has no centroid")
    return float(xs.mean()), float(ys.mean())


def embed_tile_mask(tile_mask: np.ndarray, x0: int, y0: int) -> np.ndarray:
    if tile_mask.shape != (TILE_SIZE, TILE_SIZE):
        raise ValueError(f"unexpected tile mask geometry {tile_mask.shape}")
    if x0 not in TILE_STARTS or y0 not in TILE_STARTS:
        raise ValueError("tile origin outside frozen grid")
    full = np.zeros((2048, 2048), dtype=np.uint8)
    full[y0:y0 + TILE_SIZE, x0:x0 + TILE_SIZE] = tile_mask.astype(np.uint8)
    return full


def owns_mask(mask: np.ndarray, x0: int, y0: int) -> bool:
    cx, cy = mask_centroid(mask)
    return owner_origin(cx, cy) == (x0, y0)


def mask_nms(rles: list[dict], scores: list[float], labels: list[str], threshold: float = NMS_IOU):
    if not (len(rles) == len(scores) == len(labels)):
        raise ValueError("NMS inputs must align")
    if not 0 < threshold <= 1:
        raise ValueError("NMS threshold outside (0,1]")
    order = sorted(range(len(rles)), key=lambda index: (-float(scores[index]), labels[index], index))
    kept = []
    for index in order:
        if kept:
            overlaps = np.asarray(mask_utils.iou([rles[index]], [rles[j] for j in kept],
                                                [0] * len(kept)))[0]
            if np.any(overlaps >= threshold):
                continue
        kept.append(index)
    return ([rles[index] for index in kept], [float(scores[index]) for index in kept],
            [labels[index] for index in kept])


def shard_names(names: list[str], shard: int, limit: int = 0) -> list[str]:
    if shard not in range(SHARDS):
        raise ValueError("shard must be in 0..3")
    selected = sorted(names)[shard::SHARDS]
    return selected[:limit] if limit else selected


def cache_path(branch: str, detector_sha: str, name: str, cell: dict,
               origin: tuple[int, int] | None = None) -> Path:
    if branch not in ("full", "tile") or Path(name).name != name:
        raise ValueError("invalid cache branch or image")
    if branch == "tile" and origin not in {
        (x, y) for y in TILE_STARTS for x in TILE_STARTS
    }:
        raise ValueError("tile cache requires frozen origin")
    token = {
        "experiment": "v26", "branch": branch, "detector": detector_sha,
        "file": name, "cell": cell, "origin": list(origin) if origin else None,
        "views": 4, "policy": POLICY, "full_size": FULL_INFERENCE_SIZE,
        "tile_size": TILE_SIZE, "tile_inference": TILE_INFERENCE_SIZE,
        "floor": DETECTOR_FLOOR,
    }
    digest = digest_json(token)[:24]
    return OUTPUT / "proposal_cache" / branch / f"{digest}.json.gz"


def load_modules():
    sys.path.insert(0, str(RUNTIME))
    import detector_tta as tta
    import optimize_inference as core
    import v11_refiner as v11
    return core, tta, v11


def runtime_state():
    import torch

    core, tta, v11 = load_modules()
    pins = {
        ANNOTATIONS: ANNOTATION_SHA256, PROVENANCE: PROVENANCE_SHA256,
        DETECTOR: DETECTOR_SHA256, CANONICAL_REFINER: CANONICAL_REFINER_SHA256,
        V11_REFINER: V11_REFINER_SHA256,
    }
    if not HISTORICAL_DIAGNOSTICS_SHA256:
        raise RuntimeError("stage a pinned historical diagnostics digest")
    pins[HISTORICAL_DIAGNOSTICS] = HISTORICAL_DIAGNOSTICS_SHA256
    for path, expected in pins.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"missing or changed pinned input {path}")
    base = core.load_base(SOLAR_BASE)
    base.CFG.root = DATA_ROOT
    base.CFG.train_dir = RAW_IMAGES
    base.CFG.train_json = ANNOTATIONS
    base.CFG.imgsz = FULL_INFERENCE_SIZE
    base.CFG.crop_size, base.CFG.crop_context, base.CFG.min_crop = 256, 1.8, 96
    base.seed_everything(42)
    provenance = json.loads(PROVENANCE.read_text())
    _, images, annotations = base.load_coco()
    split = core.validation_split(base, images, annotations)
    held, tune, audit, _, by_file = split
    core.verify_split(provenance, held, images)
    if (len(held), len(tune), len(audit), sum(map(len, by_file.values()))) != (141, 94, 47, 217):
        raise RuntimeError("canonical split mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("V26 requires CUDA")
    model = v11.build_model(base, CANONICAL_REFINER, False, V11_REFINER)
    return {"core": core, "tta": tta, "v11": v11, "base": base, "images": images,
            "annotations": annotations, "split": split, "refiner": model,
            "detector_sha256": sha256(DETECTOR), "historical": pd.read_csv(HISTORICAL_DIAGNOSTICS)}


def write_views(path: Path, views, signature: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt") as handle:
        json.dump([{"rles": [serialized_rle(rle) for rle in rles], "scores": scores}
                   for rles, scores in views], handle)
    temporary.replace(path)
    write_json(path.with_suffix(".signature.json"), {"signature": signature, "sha256": sha256(path)})


def read_views(path: Path, signature: dict):
    marker = path.with_suffix(".signature.json")
    if not path.is_file() or not marker.is_file():
        return None
    recorded = json.loads(marker.read_text())
    if recorded != {"signature": signature, "sha256": sha256(path)}:
        raise RuntimeError(f"stale or changed cache {path}")
    with gzip.open(path, "rt") as handle:
        raw = json.load(handle)
    if len(raw) != 4:
        raise RuntimeError("cache does not contain four views")
    views = []
    for view in raw:
        rles = [restored_rle(item) for item in view["rles"]]
        scores = [float(score) for score in view["scores"]]
        if len(rles) != len(scores) or any(rle["size"] != [2048, 2048] for rle in rles):
            raise RuntimeError("invalid cached view")
        views.append((rles, scores))
    return views


def generate_full_views(runtime, detector, image_path: Path, cell: dict):
    tta, core = runtime["tta"], runtime["core"]
    signature = {"branch": "full", "file": image_path.name, "cell": cell,
                 "detector": runtime["detector_sha256"], "views": 4}
    path = cache_path("full", runtime["detector_sha256"], image_path.name, cell)
    cached = read_views(path, signature)
    if cached is not None:
        return cached
    image = cv2.imread(str(image_path))
    if image is None or image.shape[:2] != (2048, 2048):
        raise RuntimeError(f"cannot read full image {image_path}")
    views = []
    for axes in ((), (1,), (0,), (0, 1)):
        source = str(image_path) if not axes else tta.flip_array(image, axes)
        result = detector.predict(source, imgsz=FULL_INFERENCE_SIZE, conf=DETECTOR_FLOOR,
                                  agnostic_nms=cell["agnostic_nms"],
                                  retina_masks=cell["retina_masks"], iou=cell["iou"],
                                  verbose=False)[0]
        rles, scores = [], []
        if result.masks is not None:
            scores = result.boxes.conf.detach().cpu().numpy().astype(float).tolist()
            for tensor in result.masks.data:
                mask = tta.flip_array(tensor.cpu().numpy().astype(np.uint8), axes)
                if mask.shape != (2048, 2048):
                    raise RuntimeError("full detector mask geometry changed")
                rles.append(core.encode(mask))
        views.append((rles, scores))
    write_views(path, views, signature)
    return read_views(path, signature)


def generate_tile_views(runtime, detector, image: np.ndarray, name: str, cell: dict,
                        x0: int, y0: int):
    tta, core = runtime["tta"], runtime["core"]
    origin = (x0, y0)
    signature = {"branch": "tile", "file": name, "cell": cell,
                 "origin": [x0, y0], "detector": runtime["detector_sha256"], "views": 4}
    path = cache_path("tile", runtime["detector_sha256"], name, cell, origin)
    cached = read_views(path, signature)
    if cached is not None:
        return cached
    tile = image[y0:y0 + TILE_SIZE, x0:x0 + TILE_SIZE]
    if tile.shape[:2] != (TILE_SIZE, TILE_SIZE):
        raise RuntimeError("incomplete frozen tile")
    views = []
    for axes in ((), (1,), (0,), (0, 1)):
        source = tile if not axes else tta.flip_array(tile, axes)
        result = detector.predict(source, imgsz=TILE_INFERENCE_SIZE, conf=DETECTOR_FLOOR,
                                  agnostic_nms=cell["agnostic_nms"],
                                  retina_masks=cell["retina_masks"], iou=cell["iou"],
                                  verbose=False)[0]
        rles, scores = [], []
        if result.masks is not None:
            raw_scores = result.boxes.conf.detach().cpu().numpy().astype(float).tolist()
            for tensor, score in zip(result.masks.data, raw_scores):
                local = tta.flip_array(tensor.cpu().numpy().astype(np.uint8), axes)
                full = embed_tile_mask(local, x0, y0)
                if not full.any() or not owns_mask(full, x0, y0):
                    continue
                rles.append(core.encode(full)); scores.append(float(score))
        views.append((rles, scores))
    write_views(path, views, signature)
    return read_views(path, signature)


def proposals_for_image(runtime, detector, image_path: Path, cell: dict):
    image = cv2.imread(str(image_path))
    if image is None or image.shape[:2] != (2048, 2048):
        raise RuntimeError(f"cannot read {image_path}")
    full_views = generate_full_views(runtime, detector, image_path, cell)
    full_rles, full_scores = runtime["tta"].cluster_views(full_views, POLICY, MATCH_IOU)
    tile_rles, tile_scores, tile_labels = [], [], []
    tile_before_nms = 0
    for y0 in TILE_STARTS:
        for x0 in TILE_STARTS:
            views = generate_tile_views(runtime, detector, image, image_path.name, cell, x0, y0)
            rles, scores = runtime["tta"].cluster_views(views, POLICY, MATCH_IOU)
            tile_before_nms += len(rles)
            tile_rles.extend(rles); tile_scores.extend(scores)
            tile_labels.extend([f"tile_{x0}_{y0}"] * len(rles))
    tile_rles, tile_scores, tile_labels = mask_nms(tile_rles, tile_scores, tile_labels)
    hybrid_rles, hybrid_scores, hybrid_labels = mask_nms(
        list(full_rles) + tile_rles, list(full_scores) + tile_scores,
        ["full"] * len(full_rles) + tile_labels,
    )
    diagnostics = {
        "full": len(full_rles), "tile_before_nms": tile_before_nms,
        "tile_after_nms": len(tile_rles), "hybrid_after_nms": len(hybrid_rles),
        "hybrid_full": sum(label == "full" for label in hybrid_labels),
        "hybrid_tile": sum(label != "full" for label in hybrid_labels),
    }
    return {
        "control": (list(full_rles), list(map(float, full_scores))),
        "sliced": (tile_rles, tile_scores),
        "hybrid": (hybrid_rles, hybrid_scores),
    }, diagnostics


def refine(runtime, rles: list[dict], scores: list[float], image_path: Path):
    if not rles:
        return {threshold: ([], np.asarray([])) for threshold in THRESHOLDS}
    masks = [mask_utils.decode(rle).astype(np.uint8) for rle in rles]
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None or gray.shape != (2048, 2048):
        raise RuntimeError(f"cannot read refiner pixels {image_path}")
    bounds, _, probabilities = runtime["core"].crop_probabilities(
        runtime["base"], runtime["refiner"], gray, masks, True,
    )
    return {threshold: runtime["core"].clean_refined_predictions(
        masks, scores, bounds, probabilities, threshold, MODE,
    ) for threshold in THRESHOLDS}


def raw_match_records(runtime, name: str, rles: list[dict], group: str, cell: str, arm: str):
    _, _, _, _, by_file = runtime["split"]
    rows = []
    historical = runtime["historical"]
    for image_id in sorted(by_file[name]):
        info = runtime["images"][image_id]
        gt = [runtime["core"].annotation_rle(info, entry)
              for entry in runtime["annotations"].get(image_id, [])]
        gt = [rle for rle in gt if mask_utils.area(rle) > 0]
        if rles and gt:
            matrix = np.asarray(mask_utils.iou(rles, gt, [0] * len(gt)), dtype=np.float64)
            best = matrix.max(axis=0) if matrix.size else np.zeros(len(gt))
        else:
            best = np.zeros(len(gt), dtype=np.float64)
        source = historical[(historical.file_name == name) & (historical.image_id == image_id)]
        source = source.sort_values("gt_index")
        if len(source) != len(gt):
            raise RuntimeError(f"historical GT alignment failed for {image_id}")
        for gt_index, value in enumerate(best):
            old = source.iloc[gt_index]
            rows.append({
                "arm": arm, "cell": cell, "group": group, "file_name": name,
                "image_id": image_id, "gt_index": gt_index, "gt_area": float(old.gt_area),
                "best_iou": float(value), "historical_v6_raw_iou": float(old.v6_raw_iou),
            })
    return rows


def evaluate_names(group: str, shard: int, limit: int = 0, selected: str | None = None):
    from ultralytics import YOLO

    runtime = runtime_state()
    held, tune, audit, _, by_file = runtime["split"]
    population = sorted(tune if group == "tune" else audit)
    names = shard_names(population, shard, limit)
    if group == "audit":
        selection = json.loads((OUTPUT / "tune_selection.json").read_text())
        if not selection["audit_authorized"]:
            raise RuntimeError("tune gates did not authorize audit")
        selected = selection["hybrid"]["candidate"]
    selected_parts = parse_candidate(selected) if selected else None
    detector = YOLO(str(DETECTOR))
    totals = {}
    diagnostics, raw_rows, proposal_rows = [], [], []
    started = time.monotonic()
    for position, name in enumerate(names, 1):
        image_path = RAW_IMAGES / name
        cells = (selected_parts[1],) if selected_parts else CELLS
        for cell in cells:
            proposals, proposal_diag = proposals_for_image(runtime, detector, image_path, cell)
            proposal_rows.append({"file_name": name, "group": group, "cell": cell["name"],
                                  **proposal_diag})
            arms = ("hybrid",) if selected_parts else ARMS
            raw_arms = ("control", "hybrid") if selected_parts else ARMS
            for raw_arm in raw_arms:
                raw_rles, _ = proposals[raw_arm]
                raw_rows.extend(raw_match_records(
                    runtime, name, raw_rles, group, cell["name"], raw_arm
                ))
            for arm in arms:
                rles, scores = proposals[arm]
                refined = refine(runtime, rles, scores, image_path)
                settings = ((selected_parts[2], selected_parts[3]),) if selected_parts else (
                    (threshold, confidence) for threshold in THRESHOLDS for confidence in CONFIDENCES
                )
                for threshold, confidence in settings:
                    key = candidate_key(arm, cell["name"], threshold, confidence)
                    total = totals.setdefault(key, np.zeros(4, dtype=np.float64))
                    final_rles, final_scores = refined[threshold]
                    keep = np.asarray(final_scores) >= confidence if len(final_scores) else np.zeros(0, bool)
                    kept = [rle for rle, flag in zip(final_rles, keep) if flag]
                    for image_id in sorted(by_file[name]):
                        info = runtime["images"][image_id]
                        gt = [runtime["core"].annotation_rle(info, entry)
                              for entry in runtime["annotations"].get(image_id, [])]
                        gt = [rle for rle in gt if mask_utils.area(rle) > 0]
                        matrix = (mask_utils.iou(kept, gt, [0] * len(gt)).T
                                  if kept and gt else np.zeros((len(gt), len(kept))))
                        counts = runtime["core"].pq_counts(matrix, len(gt), len(kept))
                        total += counts
                        diagnostics.append({
                            "candidate": key, "file_name": name, "image_id": image_id,
                            "group": group, **dict(zip(COUNTS, counts)),
                        })
        print(f"v26 {group} shard={shard} {position}/{len(names)} elapsed={time.monotonic()-started:.1f}s",
              flush=True)
    result = {
        "status": "complete", "group": group, "shard": shard, "limit": limit,
        "files": names, "file_count": len(names),
        "records": sum(len(by_file[name]) for name in names),
        "gt": int(sum(len(runtime["annotations"].get(image_id, []))
                      for name in names for image_id in by_file[name])),
        "results": {key: metric_record(value) for key, value in totals.items()},
        "elapsed_seconds": time.monotonic() - started,
        "detector_sha256": runtime["detector_sha256"],
        "selected_only": selected,
    }
    stage = "smoke" if limit else group
    destination = OUTPUT / stage / f"shard_{shard}"
    write_json(destination / "result.json", result)
    pd.DataFrame(diagnostics).to_csv(destination / "diagnostics.csv", index=False)
    pd.DataFrame(raw_rows).to_csv(destination / "raw_matches.csv", index=False)
    pd.DataFrame(proposal_rows).to_csv(destination / "proposal_counts.csv", index=False)
    print(json.dumps({k: result[k] for k in ("group", "shard", "file_count", "records", "gt",
                                             "elapsed_seconds")}, indent=2))
    return result


def merge_group(group: str):
    stage = group
    reports, diagnostics, raw, proposal = [], [], [], []
    for shard in range(SHARDS):
        root = OUTPUT / stage / f"shard_{shard}"
        reports.append(json.loads((root / "result.json").read_text()))
        diagnostics.append(pd.read_csv(root / "diagnostics.csv"))
        raw.append(pd.read_csv(root / "raw_matches.csv"))
        proposal.append(pd.read_csv(root / "proposal_counts.csv"))
    names = [name for report in reports for name in report["files"]]
    if len(names) != len(set(names)):
        raise RuntimeError("duplicate physical file across shards")
    expected = EXPECTED_TUNE if group == "tune" else EXPECTED_AUDIT
    actual = {"files": len(names), "records": sum(r["records"] for r in reports),
              "gt": sum(r["gt"] for r in reports)}
    if actual != expected:
        raise RuntimeError(f"merged population mismatch {actual} != {expected}")
    totals = {}
    for report in reports:
        for key, metric in report["results"].items():
            total = totals.setdefault(key, np.zeros(4, dtype=np.float64))
            total += [metric[field] for field in COUNTS]
    metrics = {key: metric_record(total) for key, total in totals.items()}
    diagnostics = pd.concat(diagnostics, ignore_index=True)
    raw = pd.concat(raw, ignore_index=True)
    proposal = pd.concat(proposal, ignore_index=True)
    merged = OUTPUT / f"merged_{group}"
    merged.mkdir(parents=True, exist_ok=True)
    write_json(merged / "results.json", {"population": actual, "results": metrics})
    diagnostics.to_csv(merged / "diagnostics.csv", index=False)
    raw.to_csv(merged / "raw_matches.csv", index=False)
    proposal.to_csv(merged / "proposal_counts.csv", index=False)
    return metrics, raw, proposal, actual


def raw_summary(frame: pd.DataFrame, arm: str, cell: str) -> dict:
    selected = frame[(frame.arm == arm) & (frame.cell == cell)].copy()
    values = selected.best_iou.to_numpy(dtype=float)
    matched = values > MATCH_IOU
    bottom = selected.gt_area <= BOTTOM_AREA_MAX
    historical_miss = selected.historical_v6_raw_iou <= MATCH_IOU
    return {
        "instances": len(selected), "recall": float(matched.mean()),
        "mean_best_iou": float(values.mean()), "max_best_iou": float(values.max()),
        "p90_best_iou": float(np.percentile(values, 90)), "missed": int((~matched).sum()),
        "bottom_instances": int(bottom.sum()),
        "bottom_recall": float(matched[bottom].mean()),
        "historical_misses": int(historical_miss.sum()),
        "historical_rescued": int((matched & historical_miss).sum()),
    }


def merge_tune():
    metrics, raw, proposal, population = merge_group("tune")
    control = max((key for key in metrics if key.startswith("control/")),
                  key=lambda key: (metrics[key]["pq"], key))
    if control != EXPECTED_CONTROL_CANDIDATE:
        raise RuntimeError(f"control selected {control}, expected {EXPECTED_CONTROL_CANDIDATE}")
    observed = [metrics[control][field] for field in COUNTS]
    if abs(observed[0] - EXPECTED_CONTROL_TUNE[0]) > 1e-6 or [int(v) for v in observed[1:]] != EXPECTED_CONTROL_TUNE[1:]:
        raise RuntimeError(f"V17 control counts changed: {observed}")
    hybrid = max((key for key in metrics if key.startswith("hybrid/")),
                 key=lambda key: (metrics[key]["pq"], key))
    sliced = max((key for key in metrics if key.startswith("sliced/")),
                 key=lambda key: (metrics[key]["pq"], key))
    _, hybrid_cell, _, _ = parse_candidate(hybrid)
    control_raw = raw_summary(raw, "control", hybrid_cell["name"])
    hybrid_raw = raw_summary(raw, "hybrid", hybrid_cell["name"])
    sliced_raw = raw_summary(raw, "sliced", hybrid_cell["name"])
    pq_absolute = metrics[hybrid]["pq"] >= TUNE_REQUIRED
    pq_control = metrics[hybrid]["pq"] >= metrics[control]["pq"] + MATCHED_CONTROL_DELTA
    raw_gate = hybrid_raw["recall"] >= control_raw["recall"] + RAW_RECALL_DELTA
    rescue_gate = hybrid_raw["historical_rescued"] >= HISTORICAL_RESCUE_REQUIRED
    bottom_gate = hybrid_raw["bottom_recall"] >= control_raw["bottom_recall"] + BOTTOM_RECALL_DELTA
    fp_limit = metrics[hybrid]["fp"] <= 1.20 * metrics[control]["fp"]
    fp_gate = fp_limit or (pq_absolute and pq_control)
    gates = {"absolute_tune": pq_absolute, "matched_control": pq_control,
             "raw_recall": raw_gate, "historical_rescue": rescue_gate,
             "bottom_recall": bottom_gate, "fp_control": fp_gate}
    authorized = all(gates.values())
    report = {
        "status": "complete", "selected_on": "tune only", "population": population,
        "control": {"candidate": control, **metrics[control], "raw": control_raw},
        "sliced": {"candidate": sliced, **metrics[sliced], "raw_selected_cell": sliced_raw},
        "hybrid": {"candidate": hybrid, **metrics[hybrid], "raw": hybrid_raw},
        "deltas": {"pq_vs_v17": metrics[hybrid]["pq"] - V17_TUNE_PQ,
                   "pq_vs_control": metrics[hybrid]["pq"] - metrics[control]["pq"],
                   "raw_recall": hybrid_raw["recall"] - control_raw["recall"],
                   "bottom_recall": hybrid_raw["bottom_recall"] - control_raw["bottom_recall"]},
        "gates": gates, "audit_authorized": bool(authorized),
        "proposal_counts": proposal.groupby("cell").sum(numeric_only=True).to_dict("index"),
        "action": "evaluate frozen hybrid on audit once" if authorized else "close V26 without audit/export",
    }
    write_json(OUTPUT / "tune_selection.json", report)
    print(json.dumps(report, indent=2))
    return report


def merge_audit():
    selection = json.loads((OUTPUT / "tune_selection.json").read_text())
    if not selection["audit_authorized"]:
        raise RuntimeError("tune gates did not authorize audit")
    metrics, raw, proposal, population = merge_group("audit")
    candidate = selection["hybrid"]["candidate"]
    if set(metrics) != {candidate}:
        raise RuntimeError("audit contains anything beyond frozen hybrid candidate")
    _, cell, _, _ = parse_candidate(candidate)
    audit_control_raw = raw_summary(raw, "control", cell["name"])
    audit_raw = raw_summary(raw, "hybrid", cell["name"])
    raw_delta = audit_raw["recall"] - audit_control_raw["recall"]
    export_authorized = metrics[candidate]["pq"] >= V17_AUDIT_PQ and raw_delta >= 0
    report = {"status": "complete", "population": population,
              "audit_control_raw": audit_control_raw,
              "audit": {"candidate": candidate, **metrics[candidate], "raw": audit_raw},
              "audit_raw_recall_delta": raw_delta,
              "required_audit_pq": V17_AUDIT_PQ,
              "required_raw_recall_delta": 0.0,
              "export_authorized": bool(export_authorized),
              "action": "implement separate test exporter for explicit authorization"
                        if export_authorized else "close V26 without export"}
    write_json(OUTPUT / "v26_evaluation.json", report)
    print(json.dumps(report, indent=2))
    return report


def verify():
    runtime = runtime_state()
    payload = {
        "status": "verified", "detector_sha256": runtime["detector_sha256"],
        "annotation_sha256": sha256(ANNOTATIONS), "provenance_sha256": sha256(PROVENANCE),
        "refiner_sha256": sha256(V11_REFINER), "historical_sha256": sha256(HISTORICAL_DIAGNOSTICS),
        "tile_size": TILE_SIZE, "tile_starts": list(TILE_STARTS),
        "owner_boundaries": list(OWNER_BOUNDARIES), "cells": list(CELLS),
        "thresholds": list(THRESHOLDS), "confidences": list(CONFIDENCES),
        "arms": list(ARMS), "shards": SHARDS, "trains": False, "submits": False,
    }
    write_json(BASE / "verification.json", payload)
    print(json.dumps(payload, indent=2))
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("verify", "smoke", "tune", "merge-tune",
                                         "audit", "merge-audit"))
    parser.add_argument("--shard", type=int, choices=range(SHARDS))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    if args.mode == "verify": verify()
    elif args.mode == "smoke":
        if args.shard is None: parser.error("smoke requires --shard")
        evaluate_names("tune", args.shard, limit=args.limit or 1)
    elif args.mode == "tune":
        if args.shard is None: parser.error("tune requires --shard")
        evaluate_names("tune", args.shard)
    elif args.mode == "merge-tune": merge_tune()
    elif args.mode == "audit":
        if args.shard is None: parser.error("audit requires --shard")
        evaluate_names("audit", args.shard)
    else: merge_audit()


if __name__ == "__main__":
    main()
