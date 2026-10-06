"""V24 canonical evaluation for matched multi-rater target policies.

Uses the already-tested V23/V17 evaluation implementation with raw detector and
refiner pixels for every arm. Tune evaluates both policies at two seeds. Audit is
opened exactly once only if the primary consensus arm clears V17 by +0.01,
clears its matched control by +0.01, improves raw recall by +0.01 in its selected
cell, and the replica agrees in PQ direction.

No mode reads test images, exports a CSV, or submits to Kaggle.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from scripts import evaluate_v23_radial_clahe as engine

BASE = Path(os.environ.get("V24_BASE", "/mnt/solar-v24-consensus"))
DATA_ROOT = Path(os.environ.get(
    "V24_DATA_ROOT",
    "/home/azureuser/solar-filament-2026/data/raw/filament-segmentation-2026/"
    "MAGFiLO_1.0_Kaggle_2026",
))
INPUTS = BASE / "inputs"
OUTPUT = BASE / "evaluation"
RUNTIME = BASE / "runtime"
RAW_IMAGES = DATA_ROOT / "train/train_images"
ANNOTATIONS = DATA_ROOT / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"

ARM_PATHS = {
    "control": BASE / "outputs/train/control/seed1/final.pt",
    "consensus": BASE / "outputs/train/consensus/seed1/final.pt",
    "control-replica": BASE / "outputs/train/control/seed2/final.pt",
    "consensus-replica": BASE / "outputs/train/consensus/seed2/final.pt",
}
V17_TUNE_PQ = 0.448702717279432
V17_AUDIT_PQ = 0.4707634285309831
TUNE_REQUIRED = V17_TUNE_PQ + 0.010
MATCHED_CONTROL_DELTA = 0.010
RAW_RECALL_DELTA = 0.010
ANNOTATION_SHA256 = "5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1"
PROVENANCE_SHA256 = "ceb09f980d5be195db35f2954c26fd19ea7ffc0f318b2bae5b7c803504b17122"
CANONICAL_REFINER_SHA256 = "8d00805318233a4196c706c1ec4eba9136ca1cb80d62298f082e82f7a2d01fb1"
V11_REFINER_SHA256 = "fae496b9d00819f49c5278e81e41a7917b31d18ea80f827e162746d840d14cd2"


def write_json(path: Path, value) -> None:
    engine.write_json(path, value)


def configure_engine() -> None:
    engine.BASE = BASE
    engine.DATA_ROOT = DATA_ROOT
    engine.RAW_IMAGES = RAW_IMAGES
    engine.TRANSFORMED_IMAGES = RAW_IMAGES
    engine.ANNOTATIONS = ANNOTATIONS
    engine.INPUTS = INPUTS
    engine.RUNTIME = RUNTIME
    engine.OUTPUT = OUTPUT
    engine.V6_PROVENANCE = INPUTS / "v6_source/v6_provenance.json"
    engine.SOLAR_BASE = INPUTS / "v6_source/solar_base.py"
    engine.CANONICAL_REFINER = INPUTS / "best_refiner_coco.pt"
    engine.V11_REFINER = INPUTS / "proposal_centered_e4.pt"
    engine.ARM_PATHS = ARM_PATHS
    engine.CONTROL_DETECTOR = ARM_PATHS["control"]
    engine.ARM_DETECTOR = ARM_PATHS["consensus"]
    engine.CONTROL_REPLICA = ARM_PATHS["control-replica"]
    engine.ARM_REPLICA = ARM_PATHS["consensus-replica"]
    engine.verify_inputs = verify_inputs
    engine.cache_path = cache_path


def verify_inputs() -> dict:
    expected = {
        INPUTS / "v6_source/v6_provenance.json": PROVENANCE_SHA256,
        INPUTS / "best_refiner_coco.pt": CANONICAL_REFINER_SHA256,
        INPUTS / "proposal_centered_e4.pt": V11_REFINER_SHA256,
        ANNOTATIONS: ANNOTATION_SHA256,
    }
    for path, digest in expected.items():
        if not path.is_file() or engine.sha256(path) != digest:
            raise RuntimeError(f"missing or changed pinned input {path}")
    consensus = json.loads((BASE / "consensus_manifest.json").read_text())
    verification = json.loads((BASE / "verification.json").read_text())
    if consensus["summary"]["train_records"] != 937:
        raise RuntimeError("consensus training population changed")
    if verification["target_id"] != consensus["target_id"]:
        raise RuntimeError("consensus target identity changed")
    checkpoints = {}
    for arm, path in ARM_PATHS.items():
        if not path.is_file():
            raise RuntimeError(f"missing trained detector {arm}: {path}")
        checkpoints[arm] = engine.sha256(path)
    return {
        "preprocessing": {"transform_id": "raw"},
        "target_id": consensus["target_id"],
        "checkpoints": checkpoints,
    }


def cache_path(arm: str, detector_sha: str, name: str, cell: dict) -> Path:
    target_id = json.loads((BASE / "consensus_manifest.json").read_text())["target_id"]
    token = f"v24|{arm}|{target_id}|{detector_sha}|{cell['name']}|{name}|1024|0.10"
    digest = engine.hashlib.sha256(token.encode()).hexdigest()[:20]
    return OUTPUT / "proposal_cache" / arm / f"{digest}.json.gz"


def best_candidate(report: dict) -> str:
    return max(sorted(report["results"]), key=lambda key: report["results"][key]["pq"])


def select_tune() -> dict:
    reports = {
        arm: json.loads((OUTPUT / f"{arm}_tune.json").read_text())
        for arm in ARM_PATHS
    }
    if any(report["group"] != "tune" for report in reports.values()):
        raise RuntimeError("selection input is not tune-only")
    selected = {arm: best_candidate(report) for arm, report in reports.items()}
    metrics = {arm: reports[arm]["results"][selected[arm]] for arm in reports}
    primary_delta = metrics["consensus"]["pq"] - metrics["control"]["pq"]
    replica_delta = metrics["consensus-replica"]["pq"] - metrics["control-replica"]["pq"]
    selected_cell = selected["consensus"].split("/", 1)[0]
    raw_control = reports["control"]["raw_proposals"][selected_cell]
    raw_consensus = reports["consensus"]["raw_proposals"][selected_cell]
    raw_recall_delta = raw_consensus["recall_gt_0_5"] - raw_control["recall_gt_0_5"]
    gates = {
        "absolute_tune": metrics["consensus"]["pq"] >= TUNE_REQUIRED,
        "matched_control": primary_delta >= MATCHED_CONTROL_DELTA,
        "raw_recall": raw_recall_delta >= RAW_RECALL_DELTA,
        "replication_direction": replica_delta > 0,
    }
    audit_authorized = all(gates.values())
    report = {
        "status": "complete", "selected_on": "tune only",
        "selected": {
            arm: {"candidate": selected[arm], **metrics[arm]} for arm in reports
        },
        "v17_tune_pq": V17_TUNE_PQ, "required_tune_pq": TUNE_REQUIRED,
        "required_delta_vs_control": MATCHED_CONTROL_DELTA,
        "required_raw_recall_delta": RAW_RECALL_DELTA,
        "delta_vs_v17": metrics["consensus"]["pq"] - V17_TUNE_PQ,
        "delta_vs_matched_control": primary_delta,
        "replica_delta": replica_delta,
        "selected_cell": selected_cell,
        "raw_proposal_control": raw_control,
        "raw_proposal_consensus": raw_consensus,
        "raw_recall_delta": raw_recall_delta,
        "gates": gates, "audit_authorized": bool(audit_authorized),
        "action": ("evaluate selected consensus candidate on audit once"
                   if audit_authorized else "close V24 without audit/export"),
    }
    write_json(OUTPUT / "tune_selection.json", report)
    print(json.dumps(report, indent=2))
    return report


def evaluate_audit() -> dict:
    selection = json.loads((OUTPUT / "tune_selection.json").read_text())
    if not selection["audit_authorized"]:
        raise RuntimeError("tune gates did not authorize audit")
    final_path = OUTPUT / "v24_evaluation.json"
    if final_path.is_file():
        print(final_path.read_text()); return json.loads(final_path.read_text())
    marker = OUTPUT / "audit_opened.json"
    if marker.exists():
        raise RuntimeError("audit was already opened without a completed result")
    candidate = selection["selected"]["consensus"]["candidate"]
    write_json(marker, {
        "candidate": candidate,
        "checkpoint_sha256": engine.sha256(ARM_PATHS["consensus"]),
        "opened_unix": time.time(),
    })
    audit = engine.evaluate_arm("consensus", "audit", candidate)
    metrics = audit["results"][candidate]
    export_authorized = metrics["pq"] >= V17_AUDIT_PQ
    result = {
        "status": "complete", "tune_selection": selection,
        "audit": {"candidate": candidate, **metrics},
        "required_audit_pq": V17_AUDIT_PQ,
        "export_authorized": bool(export_authorized),
        "action": ("implement gated test export for explicit submission authorization"
                   if export_authorized else "close V24 without export"),
    }
    write_json(final_path, result)
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    configure_engine()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("evaluate-tune", "select-tune", "evaluate-audit"))
    parser.add_argument("--arm", choices=tuple(ARM_PATHS))
    args = parser.parse_args()
    engine.cv2.setNumThreads(2)
    if args.mode == "evaluate-tune":
        if not args.arm:
            parser.error("evaluate-tune requires --arm")
        engine.evaluate_arm(args.arm, "tune")
    elif args.mode == "select-tune":
        select_tune()
    else:
        evaluate_audit()


configure_engine()

if __name__ == "__main__":
    main()
