"""V25 tune-first canonical evaluation. No test/export/submission path."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from scripts import evaluate_v23_radial_clahe as engine

BASE = Path(os.environ.get("V25_BASE", "/mnt/solar-v25-missing-label-loss"))
DATA_ROOT = Path(os.environ.get("V25_DATA_ROOT", "/home/azureuser/solar-filament-2026/data/raw/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"))
INPUTS, OUTPUT, RUNTIME = BASE / "inputs", BASE / "evaluation", BASE / "runtime"
RAW_IMAGES = DATA_ROOT / "train/train_images"
ANNOTATIONS = DATA_ROOT / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
ARM_PATHS = {
    "control": BASE / "outputs/train/control/seed1/final.pt",
    "uncertain": BASE / "outputs/train/uncertain/seed1/final.pt",
    "control-replica": BASE / "outputs/train/control/seed2/final.pt",
    "uncertain-replica": BASE / "outputs/train/uncertain/seed2/final.pt",
}
V17_TUNE_PQ, V17_AUDIT_PQ = 0.448702717279432, 0.4707634285309831
TUNE_REQUIRED, MATCHED_CONTROL_DELTA, RAW_RECALL_DELTA = V17_TUNE_PQ + .01, .01, .01
PINS = {
    "v6_source/v6_provenance.json": "ceb09f980d5be195db35f2954c26fd19ea7ffc0f318b2bae5b7c803504b17122",
    "best_refiner_coco.pt": "8d00805318233a4196c706c1ec4eba9136ca1cb80d62298f082e82f7a2d01fb1",
    "proposal_centered_e4.pt": "fae496b9d00819f49c5278e81e41a7917b31d18ea80f827e162746d840d14cd2",
}


def verify_inputs():
    for relative, digest in PINS.items():
        path = INPUTS / relative
        if not path.is_file() or engine.sha256(path) != digest:
            raise RuntimeError(f"changed pinned input {path}")
    manifest = json.loads((BASE / "experiment_manifest.json").read_text())
    checkpoints = {}
    for arm, path in ARM_PATHS.items():
        if not path.is_file(): raise RuntimeError(f"missing {arm}")
        checkpoints[arm] = engine.sha256(path)
    return {"preprocessing": {"transform_id": "raw"}, "uncertainty_id": manifest["uncertainty_id"], "checkpoints": checkpoints}


def cache_path(arm, detector_sha, name, cell):
    uid = json.loads((BASE / "experiment_manifest.json").read_text())["uncertainty_id"]
    token = f"v25|{arm}|{uid}|{detector_sha}|{cell['name']}|{name}|1024|0.10"
    return OUTPUT / "proposal_cache" / arm / (engine.hashlib.sha256(token.encode()).hexdigest()[:20] + ".json.gz")


def configure():
    engine.BASE, engine.DATA_ROOT = BASE, DATA_ROOT
    engine.RAW_IMAGES = engine.TRANSFORMED_IMAGES = RAW_IMAGES
    engine.ANNOTATIONS, engine.INPUTS, engine.RUNTIME, engine.OUTPUT = ANNOTATIONS, INPUTS, RUNTIME, OUTPUT
    engine.V6_PROVENANCE, engine.SOLAR_BASE = INPUTS / "v6_source/v6_provenance.json", INPUTS / "v6_source/solar_base.py"
    engine.CANONICAL_REFINER, engine.V11_REFINER = INPUTS / "best_refiner_coco.pt", INPUTS / "proposal_centered_e4.pt"
    engine.ARM_PATHS, engine.verify_inputs, engine.cache_path = ARM_PATHS, verify_inputs, cache_path


def select_tune():
    reports = {arm: json.loads((OUTPUT / f"{arm}_tune.json").read_text()) for arm in ARM_PATHS}
    selected = {arm: max(sorted(r["results"]), key=lambda k: r["results"][k]["pq"]) for arm, r in reports.items()}
    metrics = {arm: reports[arm]["results"][selected[arm]] for arm in reports}
    cell = selected["uncertain"].split("/", 1)[0]
    raw_c, raw_u = reports["control"]["raw_proposals"][cell], reports["uncertain"]["raw_proposals"][cell]
    delta, replica = metrics["uncertain"]["pq"] - metrics["control"]["pq"], metrics["uncertain-replica"]["pq"] - metrics["control-replica"]["pq"]
    raw_delta = raw_u["recall_gt_0_5"] - raw_c["recall_gt_0_5"]
    gates = {"absolute_tune": metrics["uncertain"]["pq"] >= TUNE_REQUIRED, "matched_control": delta >= .01, "raw_recall": raw_delta >= .01, "replication_direction": replica > 0}
    authorized = all(gates.values())
    result = {"status":"complete","selected_on":"tune only","selected":{a:{"candidate":selected[a],**metrics[a]} for a in reports},"v17_tune_pq":V17_TUNE_PQ,"required_tune_pq":TUNE_REQUIRED,"delta_vs_v17":metrics["uncertain"]["pq"]-V17_TUNE_PQ,"delta_vs_matched_control":delta,"replica_delta":replica,"selected_cell":cell,"raw_proposal_control":raw_c,"raw_proposal_uncertain":raw_u,"raw_recall_delta":raw_delta,"gates":gates,"audit_authorized":authorized,"action":"evaluate audit once" if authorized else "close V25 without audit/export"}
    engine.write_json(OUTPUT / "tune_selection.json", result); print(json.dumps(result, indent=2)); return result


def evaluate_audit():
    selection = json.loads((OUTPUT / "tune_selection.json").read_text())
    if not selection["audit_authorized"]: raise RuntimeError("tune gates did not authorize audit")
    marker = OUTPUT / "audit_opened.json"
    if marker.exists(): raise RuntimeError("audit already opened")
    candidate = selection["selected"]["uncertain"]["candidate"]
    engine.write_json(marker, {"candidate":candidate,"checkpoint_sha256":engine.sha256(ARM_PATHS["uncertain"]),"opened_unix":time.time()})
    audit = engine.evaluate_arm("uncertain", "audit", candidate); metrics = audit["results"][candidate]
    result={"status":"complete","tune_selection":selection,"audit":{"candidate":candidate,**metrics},"required_audit_pq":V17_AUDIT_PQ,"export_authorized":metrics["pq"]>=V17_AUDIT_PQ}
    engine.write_json(OUTPUT / "v25_evaluation.json", result); print(json.dumps(result,indent=2)); return result


def main():
    configure(); parser=argparse.ArgumentParser(); parser.add_argument("mode",choices=("evaluate-tune","select-tune","evaluate-audit")); parser.add_argument("--arm",choices=tuple(ARM_PATHS)); args=parser.parse_args(); engine.cv2.setNumThreads(2)
    if args.mode=="evaluate-tune":
        if not args.arm: parser.error("--arm required")
        engine.evaluate_arm(args.arm,"tune")
    elif args.mode=="select-tune": select_tune()
    else: evaluate_audit()

configure()
if __name__=="__main__": main()
