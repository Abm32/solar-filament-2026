"""V28 frozen utility scorer training and sealed-split evaluation.

This stage is authorized only by the successful file-grouped OOF feasibility
report. It trains the frozen scorer on all 566 training files, evaluates one
predeclared score on tune, and opens audit only when every tune gate passes.
It never reads competition test images and never creates a submission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

try:
    from scripts import analyze_v28_utility as cohort
except ImportError:  # staged beside analyze_v28_utility.py on Azure
    import analyze_v28_utility as cohort

BASE = Path(os.environ.get("V28_BASE", "/mnt/solar-v28-utility"))
OUTPUT = BASE / "evaluation"
MODEL_PATH = BASE / "model/v28_utility.pt"
TRAIN_REPORT = BASE / "model/training_report.json"
OOF_REPORT = BASE / "cohort/oof_report.json"

SCALAR_COLUMNS = (
    "confidence", "log_area", "area_ratio", "crop_width", "crop_height",
    "source_full", "source_tile",
)
EPOCHS = 6
BATCH_SIZE = 128
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
PRODUCT_THRESHOLD = 0.10
REFINE_THRESHOLD = 0.45
CONTROL_CONFIDENCE = 0.30
HYBRID_CONFIDENCE = 0.45
MIN_AREA = 20
SHARDS = 4
SEED = 20260912
COUNTS = ("iou_sum", "tp", "fp", "fn")
EXPECTED_TUNE = {"files": 94, "records": 150, "gt": 1134}
EXPECTED_AUDIT = {"files": 47, "records": 67, "gt": 459}
EXPECTED_CONTROL = np.asarray([481.0093129235512, 704, 306, 430], dtype=float)
EXPECTED_HYBRID = np.asarray([465.5503995207298, 678, 339, 456], dtype=float)
V17_TUNE_PQ = 0.44870271727943206
V17_AUDIT_PQ = 0.470763
ABSOLUTE_TUNE_REQUIRED = V17_TUNE_PQ + 0.010
HYBRID_DELTA_REQUIRED = 0.010


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def pq(counts) -> float:
    iou_sum, tp, fp, fn = map(float, counts)
    denominator = tp + 0.5 * (fp + fn)
    return iou_sum / denominator if denominator else 0.0


def metric_record(counts) -> dict:
    iou_sum, tp, fp, fn = map(float, counts)
    denominator = tp + 0.5 * (fp + fn)
    return {
        "pq": iou_sum / denominator if denominator else 0.0,
        "sq": iou_sum / tp if tp else 0.0,
        "rq": tp / denominator if denominator else 0.0,
        "iou_sum": iou_sum,
        "tp": int(round(tp)), "fp": int(round(fp)), "fn": int(round(fn)),
    }


def scalar_features(frame: pd.DataFrame) -> np.ndarray:
    values = frame[list(SCALAR_COLUMNS)].astype(float).to_numpy(np.float32)
    values[:, 3:5] /= 2048.0
    return values


def utility_net():
    import torch
    from torch import nn

    class UtilityNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(4, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            )
            self.head = nn.Sequential(
                nn.Linear(64 + len(SCALAR_COLUMNS), 64), nn.ReLU(),
                nn.Dropout(0.15), nn.Linear(64, 2),
            )

        def forward(self, image, values):
            return self.head(torch.cat((self.conv(image).flatten(1), values), 1))

    return UtilityNet()


def require_oof_authorization() -> dict:
    if not OOF_REPORT.is_file():
        raise RuntimeError("missing V28 OOF report")
    report = json.loads(OOF_REPORT.read_text())
    if not report.get("learnable"):
        raise RuntimeError("V28 OOF learnability gates did not authorize this stage")
    baseline = report["metrics"]["confidence"]
    candidate = report["metrics"]["oof_product"]
    observed = {
        "auc": candidate["auc"] - baseline["auc"],
        "correlation": candidate["utility_correlation"] - baseline["utility_correlation"],
        "cross_fitted_pq": (
            candidate["cross_fitted_pq"]["mean_fold_pq"]
            - baseline["cross_fitted_pq"]["mean_fold_pq"]
        ),
    }
    required = report["learnability_gates"]
    if not (
        observed["auc"] > required["auc_delta"]
        and observed["correlation"] > required["correlation_delta"]
        and observed["cross_fitted_pq"] > required["cross_fitted_pq_delta"]
    ):
        raise RuntimeError("OOF report authorization is internally inconsistent")
    thresholds = {
        row["threshold"]
        for row in candidate["cross_fitted_pq"]["folds"]
    }
    if thresholds != {PRODUCT_THRESHOLD}:
        raise RuntimeError(f"OOF folds did not unanimously select {PRODUCT_THRESHOLD}: {thresholds}")
    return {"report_sha256": sha256(OOF_REPORT), "observed_deltas": observed}


def train_full(epochs: int = EPOCHS) -> dict:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    if epochs != EPOCHS:
        raise RuntimeError(f"frozen V28 training requires exactly {EPOCHS} epochs")
    authorization = require_oof_authorization()
    frame, crops = cohort.load_all_samples()
    if len(frame) != 14127 or frame.file_name.nunique() != 566:
        raise RuntimeError("V28 full-training population changed")
    scalar = scalar_features(frame)
    dataset = TensorDataset(
        torch.from_numpy(crops.astype(np.float32) / 255.0),
        torch.from_numpy(scalar),
        torch.from_numpy(frame.utility.to_numpy(np.float32)),
        torch.from_numpy(frame.positive_fraction.to_numpy(np.float32)),
    )
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2,
                        generator=generator)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    model = utility_net().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )
    history = []
    for epoch in range(epochs):
        model.train()
        losses = []
        for image, values, utility, fraction in loader:
            image, values = image.to(device), values.to(device)
            utility, fraction = utility.to(device), fraction.to(device)
            output = model(image, values)
            loss = (
                torch.nn.functional.smooth_l1_loss(output[:, 0].sigmoid(), utility)
                + torch.nn.functional.binary_cross_entropy(output[:, 1].sigmoid(), fraction)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch + 1, "mean_loss": float(np.mean(losses))})
        print(f"v28 full train epoch={epoch + 1}/{epochs} loss={history[-1]['mean_loss']:.6f}",
              flush=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = MODEL_PATH.with_suffix(".tmp")
    torch.save({
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "epochs": epochs, "seed": SEED, "scalar_columns": list(SCALAR_COLUMNS),
        "crop_size": cohort.CROP_SIZE, "product_threshold": PRODUCT_THRESHOLD,
        "refine_threshold": REFINE_THRESHOLD,
        "oof_report_sha256": authorization["report_sha256"],
    }, temporary)
    temporary.replace(MODEL_PATH)
    report = {
        "status": "complete", "files": 566, "samples": len(frame),
        "positive": int((frame.positive_fraction > 0).sum()), "epochs": epochs,
        "seed": SEED, "history": history, "model_sha256": sha256(MODEL_PATH),
        "authorization": authorization,
    }
    write_json(TRAIN_REPORT, report)
    print(json.dumps(report, indent=2))
    return report


def load_scorer(device):
    import torch

    authorization = require_oof_authorization()
    if not MODEL_PATH.is_file() or not TRAIN_REPORT.is_file():
        raise RuntimeError("missing trained V28 utility model")
    report = json.loads(TRAIN_REPORT.read_text())
    if report.get("model_sha256") != sha256(MODEL_PATH):
        raise RuntimeError("V28 model hash differs from training report")
    payload = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    expected = {
        "epochs": EPOCHS, "seed": SEED, "scalar_columns": list(SCALAR_COLUMNS),
        "crop_size": cohort.CROP_SIZE, "product_threshold": PRODUCT_THRESHOLD,
        "refine_threshold": REFINE_THRESHOLD,
        "oof_report_sha256": authorization["report_sha256"],
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"V28 model metadata mismatch for {key}")
    model = utility_net()
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device).eval()


def proposal_frame(proposals: dict, refined: list[dict], bounds: list[tuple]) -> pd.DataFrame:
    hybrid_rles, scores = proposals["hybrid"]
    control_keys = {
        json.dumps(cohort.serialized_rle(rle), sort_keys=True)
        for rle in proposals["control"][0]
    }
    tile_keys = {
        json.dumps(cohort.serialized_rle(rle), sort_keys=True)
        for rle in proposals["sliced"][0]
    }
    rows = []
    for rle, refined_rle, confidence, bound in zip(hybrid_rles, refined, scores, bounds):
        key = json.dumps(cohort.serialized_rle(rle), sort_keys=True)
        area = float(mask_utils.area(rle))
        refined_area = float(mask_utils.area(refined_rle))
        x0, y0, x1, y1 = bound
        rows.append({
            "confidence": float(confidence), "log_area": float(math.log1p(area)),
            "area_ratio": float(refined_area / max(area, 1.0)),
            "crop_width": x1 - x0, "crop_height": y1 - y0,
            "source_full": key in control_keys, "source_tile": key in tile_keys,
        })
    return pd.DataFrame(rows, columns=SCALAR_COLUMNS)


def normalized_crops(crops) -> np.ndarray:
    values = np.asarray(crops, dtype=np.uint8)
    if values.ndim != 4 or values.shape[1:] != (4, cohort.CROP_SIZE, cohort.CROP_SIZE):
        raise ValueError("V28 crops must be N x 4 x 64 x 64")
    return values


def predict_utility(model, crops, scalar: np.ndarray, device) -> np.ndarray:
    import torch

    crops = normalized_crops(crops)
    if not len(crops):
        return np.zeros(0, dtype=np.float32)
    blocks = []
    with torch.no_grad():
        for start in range(0, len(crops), 256):
            image = torch.from_numpy(crops[start:start + 256].astype(np.float32) / 255.0).to(device)
            values = torch.from_numpy(scalar[start:start + 256]).to(device)
            blocks.append(model(image, values)[:, 0].sigmoid().cpu().numpy())
    return np.concatenate(blocks)


def disjoint_refined(rles: list[dict], scores, threshold: float = PRODUCT_THRESHOLD):
    if len(rles) != len(scores):
        raise ValueError("refined masks and scores must align")
    if not rles:
        return [], np.zeros(0, dtype=float)
    occupied = np.zeros(tuple(rles[0]["size"]), dtype=bool)
    kept_rles, kept_scores = [], []
    for index in sorted(range(len(scores)), key=lambda item: (-float(scores[item]), item)):
        mask = mask_utils.decode(rles[index]).astype(bool)
        mask &= ~occupied
        if mask.sum() < MIN_AREA:
            continue
        occupied |= mask
        if float(scores[index]) >= threshold:
            kept_rles.append(mask_utils.encode(np.asfortranarray(mask.astype(np.uint8))))
            kept_scores.append(float(scores[index]))
    return kept_rles, np.asarray(kept_scores, dtype=float)


def confidence_candidate(runtime, rles, scores, image_path: Path, confidence: float):
    refined, refined_scores = runtime["v26"].refine(runtime, rles, scores, image_path)[REFINE_THRESHOLD]
    keep = np.asarray(refined_scores) >= confidence if len(refined_scores) else np.zeros(0, bool)
    return [rle for rle, flag in zip(refined, keep) if flag]


def utility_candidate(runtime, model, proposals: dict, image_path: Path, device):
    hybrid_rles, confidence = proposals["hybrid"]
    refined, _, bounds, crops = cohort.independent_refined_masks(runtime, hybrid_rles, image_path)
    frame = proposal_frame(proposals, refined, bounds)
    utility = predict_utility(model, crops, scalar_features(frame), device)
    product = np.asarray(confidence, dtype=float) * utility
    selected, _ = disjoint_refined(refined, product)
    return selected, product, utility


def counts_for_file(runtime, name: str, candidates: dict[str, list[dict]]) -> dict[str, np.ndarray]:
    by_file = runtime["split"][4]
    totals = {key: np.zeros(4, dtype=float) for key in candidates}
    for image_id in sorted(by_file[name]):
        info = runtime["images"][image_id]
        gt = [runtime["core"].annotation_rle(info, entry)
              for entry in runtime["annotations"].get(image_id, [])]
        gt = [rle for rle in gt if mask_utils.area(rle) > 0]
        for key, predicted in candidates.items():
            matrix = (mask_utils.iou(predicted, gt, [0] * len(gt)).T
                      if predicted and gt else np.zeros((len(gt), len(predicted))))
            totals[key] += runtime["core"].pq_counts(matrix, len(gt), len(predicted))
    return totals


def evaluate(group: str, shard: int, limit: int = 0) -> dict:
    import torch
    from ultralytics import YOLO

    if group == "audit":
        selection_path = OUTPUT / "tune_selection.json"
        if not selection_path.is_file() or not json.loads(selection_path.read_text()).get("audit_authorized"):
            raise RuntimeError("tune gates did not authorize audit")
    runtime = cohort.load_runtime()
    population = sorted(runtime["split"][1] if group == "tune" else runtime["split"][2])
    names = population[shard::SHARDS]
    if limit:
        names = names[:limit]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_scorer(device)
    detector = YOLO(str(cohort.DETECTOR))
    totals = {key: np.zeros(4, dtype=float) for key in ("control", "hybrid_confidence", "v28_product")}
    diagnostics = []
    for position, name in enumerate(names, 1):
        image_path = cohort.RAW_IMAGES / name
        proposals, proposal_diagnostics = runtime["v26"].proposals_for_image(
            runtime, detector, image_path, cohort.CELL,
        )
        control = confidence_candidate(runtime, *proposals["control"], image_path,
                                       CONTROL_CONFIDENCE)
        hybrid = confidence_candidate(runtime, *proposals["hybrid"], image_path,
                                      HYBRID_CONFIDENCE)
        candidate, products, utilities = utility_candidate(runtime, model, proposals, image_path, device)
        file_counts = counts_for_file(runtime, name, {
            "control": control, "hybrid_confidence": hybrid, "v28_product": candidate,
        })
        for key, values in file_counts.items():
            totals[key] += values
            diagnostics.append({
                "file_name": name, "group": group, "candidate": key,
                **dict(zip(COUNTS, values)),
            })
        diagnostics[-1].update({
            "raw_proposals": len(proposals["hybrid"][0]),
            "selected": len(candidate), "product_min": float(products.min()) if len(products) else None,
            "product_max": float(products.max()) if len(products) else None,
            "utility_mean": float(utilities.mean()) if len(utilities) else None,
            **proposal_diagnostics,
        })
        print(f"v28 {group} shard={shard} {position}/{len(names)}", flush=True)
    by_file = runtime["split"][4]
    result = {
        "status": "complete", "group": group, "shard": shard, "limit": limit,
        "files": names, "file_count": len(names),
        "records": sum(len(by_file[name]) for name in names),
        "gt": int(sum(len(runtime["annotations"].get(image_id, []))
                      for name in names for image_id in by_file[name])),
        "results": {key: metric_record(value) for key, value in totals.items()},
        "model_sha256": sha256(MODEL_PATH),
    }
    stage = "smoke" if limit else group
    destination = OUTPUT / stage / f"shard_{shard}"
    write_json(destination / "result.json", result)
    pd.DataFrame(diagnostics).to_csv(destination / "diagnostics.csv", index=False)
    print(json.dumps(result, indent=2))
    return result


def merge_group(group: str):
    reports, diagnostics = [], []
    for shard in range(SHARDS):
        root = OUTPUT / group / f"shard_{shard}"
        reports.append(json.loads((root / "result.json").read_text()))
        diagnostics.append(pd.read_csv(root / "diagnostics.csv"))
    names = [name for report in reports for name in report["files"]]
    expected = EXPECTED_TUNE if group == "tune" else EXPECTED_AUDIT
    actual = {
        "files": len(names), "records": sum(row["records"] for row in reports),
        "gt": sum(row["gt"] for row in reports),
    }
    if len(names) != len(set(names)) or actual != expected:
        raise RuntimeError(f"V28 merged population mismatch: {actual} != {expected}")
    hashes = {row["model_sha256"] for row in reports}
    if hashes != {sha256(MODEL_PATH)}:
        raise RuntimeError("V28 shards used different utility models")
    totals = {key: np.zeros(4, dtype=float) for key in reports[0]["results"]}
    for report in reports:
        if set(report["results"]) != set(totals):
            raise RuntimeError("V28 shard candidate mismatch")
        for key, metric in report["results"].items():
            totals[key] += [metric[field] for field in COUNTS]
    metrics = {key: metric_record(value) for key, value in totals.items()}
    merged = OUTPUT / f"merged_{group}"
    write_json(merged / "results.json", {"population": actual, "results": metrics})
    pd.concat(diagnostics, ignore_index=True).to_csv(merged / "diagnostics.csv", index=False)
    return metrics, actual


def assert_regression(observed, expected, label: str) -> None:
    if abs(float(observed[0]) - float(expected[0])) > 1e-6:
        raise RuntimeError(f"{label} IoU sum changed: {observed[0]} != {expected[0]}")
    if [int(round(value)) for value in observed[1:]] != [int(value) for value in expected[1:]]:
        raise RuntimeError(f"{label} counts changed: {observed[1:]} != {expected[1:]}")


def evaluate_gates(metrics: dict) -> dict:
    candidate = metrics["v28_product"]
    control = metrics["control"]
    hybrid = metrics["hybrid_confidence"]
    gates = {
        "absolute_v17_plus_0.010": candidate["pq"] >= ABSOLUTE_TUNE_REQUIRED,
        "matched_hybrid_plus_0.010": candidate["pq"] >= hybrid["pq"] + HYBRID_DELTA_REQUIRED,
        "tp_at_least_v17": candidate["tp"] >= control["tp"],
        "fp_no_more_than_20pct_v17": candidate["fp"] <= 1.20 * control["fp"],
    }
    return {"gates": gates, "audit_authorized": bool(all(gates.values()))}


def merge_tune() -> dict:
    metrics, population = merge_group("tune")
    assert_regression(
        [metrics["control"][field] for field in COUNTS], EXPECTED_CONTROL, "V17 control",
    )
    assert_regression(
        [metrics["hybrid_confidence"][field] for field in COUNTS], EXPECTED_HYBRID,
        "V26 hybrid confidence",
    )
    decision = evaluate_gates(metrics)
    report = {
        "status": "complete", "selected_on": "frozen OOF threshold only",
        "population": population,
        "configuration": {
            "cell": cohort.CELL, "refine_threshold": REFINE_THRESHOLD,
            "score": "detector_confidence * predicted_utility",
            "product_threshold": PRODUCT_THRESHOLD,
        },
        "metrics": metrics,
        "deltas": {
            "pq_vs_v17": metrics["v28_product"]["pq"] - metrics["control"]["pq"],
            "pq_vs_v26_hybrid": (
                metrics["v28_product"]["pq"] - metrics["hybrid_confidence"]["pq"]
            ),
        },
        "requirements": {
            "absolute_tune_pq": ABSOLUTE_TUNE_REQUIRED,
            "hybrid_delta": HYBRID_DELTA_REQUIRED,
        },
        **decision,
        "action": "evaluate frozen V28 candidate on audit once" if decision["audit_authorized"]
                  else "close V28 without audit/test/export",
    }
    write_json(OUTPUT / "tune_selection.json", report)
    print(json.dumps(report, indent=2))
    return report


def merge_audit() -> dict:
    selection = json.loads((OUTPUT / "tune_selection.json").read_text())
    if not selection.get("audit_authorized"):
        raise RuntimeError("tune gates did not authorize audit")
    metrics, population = merge_group("audit")
    candidate = metrics["v28_product"]
    export_authorized = candidate["pq"] >= V17_AUDIT_PQ
    report = {
        "status": "complete", "population": population, "metrics": metrics,
        "required_audit_pq": V17_AUDIT_PQ,
        "export_authorized": bool(export_authorized),
        "action": "freeze a separate test exporter for explicit approval" if export_authorized
                  else "close V28 without test/export",
    }
    write_json(OUTPUT / "v28_evaluation.json", report)
    print(json.dumps(report, indent=2))
    return report


def verify() -> dict:
    authorization = require_oof_authorization()
    report = {
        "status": "verified", "authorization": authorization,
        "epochs": EPOCHS, "seed": SEED, "crop_size": cohort.CROP_SIZE,
        "scalar_columns": list(SCALAR_COLUMNS), "cell": cohort.CELL,
        "refine_threshold": REFINE_THRESHOLD, "product_threshold": PRODUCT_THRESHOLD,
        "control_confidence": CONTROL_CONFIDENCE, "hybrid_confidence": HYBRID_CONFIDENCE,
        "tune_requirements": {
            "absolute_pq": ABSOLUTE_TUNE_REQUIRED, "hybrid_delta": HYBRID_DELTA_REQUIRED,
        },
        "audit_requirement": V17_AUDIT_PQ,
        "reads_test": False, "exports_csv": False, "submits": False,
    }
    write_json(BASE / "v28_verification.json", report)
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("verify", "train", "evaluate", "merge"))
    parser.add_argument("--group", choices=("tune", "audit"), default="tune")
    parser.add_argument("--shard", type=int, choices=range(SHARDS))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    if args.limit < 0:
        parser.error("limit must be nonnegative")
    if args.mode == "verify":
        verify()
    elif args.mode == "train":
        train_full(args.epochs)
    elif args.mode == "evaluate":
        if args.shard is None:
            parser.error("evaluate requires --shard")
        evaluate(args.group, args.shard, args.limit)
    elif args.group == "tune":
        merge_tune()
    else:
        merge_audit()


if __name__ == "__main__":
    main()
