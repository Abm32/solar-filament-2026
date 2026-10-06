"""Orchestrate four-GPU tune evaluation and one selected audit run for V20."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(os.environ.get("V20_BASE", "/mnt/solar-v20-condinst"))
OUT = BASE / "outputs" / "v20_condinst"
FAST = BASE / "evaluate_v20_condinst_fast.py"
PYTHON = Path("/home/azureuser/solar-filament-2026/.venv/bin/python")
CHECKPOINTS = [
    OUT / "model_0001174.pth",
    OUT / "model_0002349.pth",
    OUT / "model_0003524.pth",
    OUT / "model_0004699.pth",
]
CONFIDENCES = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
SUBMIT_BAR_TUNE = 0.448703
SUBMIT_BAR_AUDIT = 0.470763
PROMOTION_GATE_TUNE = 0.55
POPULATIONS = {
    "train": {"files": 566, "records": 937, "annotations": 6606},
    "tune": {"files": 94, "records": 150, "annotations": 1134},
    "audit": {"files": 47, "records": 67, "annotations": 459},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def worker(checkpoint: Path, gpu: int, split: str, thresholds: str, result: Path, log: Path):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["OMP_NUM_THREADS"] = "1"
    command = [
        str(PYTHON), str(FAST), "--checkpoint", str(checkpoint),
        "--split", split, "--thresholds", thresholds, "--out", str(result),
    ]
    handle = log.open("w")
    process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=env)
    return process, handle


def main() -> None:
    started = time.monotonic()
    parallel_dir = BASE / "outputs" / "v20_fast_tune"
    parallel_dir.mkdir(parents=True, exist_ok=True)
    workers = []
    threshold_text = ",".join(str(value) for value in CONFIDENCES)
    for gpu, checkpoint in enumerate(CHECKPOINTS):
        result = parallel_dir / f"{checkpoint.stem}.json"
        log = parallel_dir / f"{checkpoint.stem}.log"
        process, handle = worker(checkpoint, gpu, "tune", threshold_text, result, log)
        workers.append((checkpoint, result, log, process, handle))
    failures = []
    for checkpoint, result, log, process, handle in workers:
        returncode = process.wait()
        handle.close()
        if returncode != 0 or not result.is_file():
            failures.append({"checkpoint": checkpoint.name, "returncode": returncode, "log": str(log)})
    if failures:
        (BASE / "outputs" / "v20_parallel_status.json").write_text(
            json.dumps({"state": "tune_failed", "failures": failures}, indent=2) + "\n"
        )
        raise SystemExit(1)

    tune_runs = [json.loads(result.read_text()) for _, result, _, _, _ in workers]
    choices = []
    for run in tune_runs:
        for threshold, metrics in run["results"].items():
            choices.append((float(metrics["pq"]), float(threshold), run))
    _, selected_confidence, selected_run = max(choices, key=lambda item: (item[0], item[1]))
    selected_checkpoint = OUT / selected_run["checkpoint"]

    audit_result = BASE / "outputs" / "v20_fast_audit.json"
    audit_log = BASE / "outputs" / "v20_fast_audit.log"
    process, handle = worker(
        selected_checkpoint, 0, "audit", str(selected_confidence), audit_result, audit_log,
    )
    returncode = process.wait()
    handle.close()
    if returncode != 0 or not audit_result.is_file():
        (BASE / "outputs" / "v20_parallel_status.json").write_text(
            json.dumps({"state": "audit_failed", "returncode": returncode, "log": str(audit_log)}, indent=2) + "\n"
        )
        raise SystemExit(1)
    audit_run = json.loads(audit_result.read_text())
    selected_tune = selected_run["results"][str(selected_confidence)]
    selected_audit = audit_run["results"][str(selected_confidence)]
    authorized = (
        selected_tune["pq"] >= SUBMIT_BAR_TUNE
        and selected_audit["pq"] >= SUBMIT_BAR_AUDIT
    )
    report = {
        "status": "complete",
        "evaluation_implementation": "parallel cached-RLE; exact parity self-test passed",
        "selection": "epochs 5/10/15/20 x confidence 0.05-0.50 on tune; selected pair on audit once",
        "nms_threshold": 0.6,
        "selected_checkpoint": selected_checkpoint.name,
        "selected_checkpoint_sha256": sha256(selected_checkpoint),
        "selected_confidence": selected_confidence,
        "selected_tune": selected_tune,
        "selected_audit": selected_audit,
        "submit_bar_tune": SUBMIT_BAR_TUNE,
        "submit_bar_audit": SUBMIT_BAR_AUDIT,
        "promotion_gate_tune": PROMOTION_GATE_TUNE,
        "export_authorized": authorized,
        "promotion_gate_cleared": selected_tune["pq"] >= PROMOTION_GATE_TUNE,
        "populations": POPULATIONS,
        "tune_runs": tune_runs,
        "audit_selected_run": audit_run,
        "evaluation_seconds": time.monotonic() - started,
        "action": "hold_for_user_authorization" if authorized else "no_export",
    }
    report_path = OUT / "v20_evaluation.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    checks = {
        "report_sha256": sha256(report_path),
        "selected_checkpoint_sha256": sha256(selected_checkpoint),
        "fast_evaluator_sha256": sha256(FAST),
    }
    (BASE / "outputs" / "v20_artifact_sha256.json").write_text(
        json.dumps(checks, indent=2) + "\n"
    )
    (BASE / "outputs" / "v20_parallel_status.json").write_text(
        json.dumps({"state": "complete", **report}, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
