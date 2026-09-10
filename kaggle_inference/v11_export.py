"""Export-only V11 test inference from the already-trained gate-failed checkpoint.

This does NOT train and does NOT re-select. The V11 experiment
(`abhimanyu122/solar-filament-v11-refiner` version 1) already selected
`v11_proposal_centered_e4_refined_constrained` on tuning PQ, froze that
selection before auditing, and then FAILED its predeclared +0.01 tuning-PQ
promotion gate. That gate governs promotion: whether V11 replaces the accepted
baseline. It deliberately blocked the export branch, so no test predictions
exist.

This script measures the same frozen candidate on the public leaderboard without
promoting it. It reuses the identical pinned runtime, the same 321 frozen raw
detector caches, the same geometry/threshold/ownership rules, and the exact
trained checkpoint recovered from the completed run's output. The candidate
selection, threshold, confidence, and mode are read from that run's frozen
`selected_config.json`; nothing here can choose a different candidate.

Guarantees enforced at runtime:
  * the recovered checkpoint hash matches the pin embedded by the stager;
  * the checkpoint's own run identity equals the completed V11 run identity;
  * the frozen control export reproduces the accepted 0.38 CSV byte hash;
  * no optimizer, scheduler, or training loop is constructed at all.

`promotable` stays false in every artifact written here. Promotion still
requires the original gate to pass in a future experiment.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils


CONTROL = "v11_frozen_control"
ARM = "proposal_centered"
EXPECTED_RUN_IDENTITY = "7c45acf26f210acce00f1bff581288bc2eb6121485237c7c028588d1de69164c"
SOURCE_RUN = "abhimanyu122/solar-filament-v11-refiner"
# Every arm's epoch-4 tuning PQ from the completed run, so an ablation export
# records the local number it is testing against the public result.
ARM_TUNE_PQ = {"proposal_centered": 0.438001, "gt_centered": 0.437313}
GATE = dict(promotable=False, reason="predeclared improvement gates failed",
            deltas=dict(tune_v6=0.005177338559703704, tune_v10=0.003936376179929257,
                        audit_v6=0.004754753668608613, audit_v10=0.0030472597663005474),
            required_min_tune_delta_v6=0.01)
CHECKPOINT_SHA256 = None
MODULE_SHA256 = None
RAW_CACHE_SHA256 = None


def expected_candidate(arm=None):
    return f"v11_{arm or ARM}_e4_refined_constrained"


def locate_checkpoint(input_root, digest, arm=None):
    """Find the single recovered candidate checkpoint and verify its identity."""
    name = f"{arm or ARM}_e4.pt"
    matches = [path for path in Path(input_root).rglob(name)
               if v11.sha256(path) == digest]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one pinned {name} checkpoint")
    return matches[0]


def verify_checkpoint_identity(path, expected_identity=EXPECTED_RUN_IDENTITY):
    """The checkpoint must come from the completed V11 run, not another experiment."""
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("model", "run_identity", "epoch", "cohort_sha256"):
        if key not in payload:
            raise ValueError(f"Recovered checkpoint is missing {key}")
    if payload["run_identity"] != expected_identity:
        raise ValueError("Checkpoint run identity does not match the completed V11 run")
    if payload["epoch"] != 4:
        raise ValueError("Expected the epoch-4 checkpoint")
    return dict(run_identity=payload["run_identity"], epoch=payload["epoch"],
                cohort_sha256=payload["cohort_sha256"], arm_file=Path(path).name)


def verify_frozen_selection(selection, arm=None):
    """Reuse the run's frozen per-arm result; never re-select here.

    For the selected winner this is the run's own frozen selection. For an
    ablation arm it is the same document with the arm substituted, which is why
    the arm must be declared by the stager rather than inferred at runtime.
    """
    arm = arm or ARM
    if selection.get("candidate") != expected_candidate("proposal_centered"):
        raise ValueError("Frozen selection is not the expected V11 winner")
    if not selection.get("frozen_before_audit") or selection.get("smoke"):
        raise ValueError("Selection must be the non-smoke, pre-audit frozen choice")
    if selection.get("run_identity") != EXPECTED_RUN_IDENTITY:
        raise ValueError("Frozen selection belongs to another run")
    if arm not in ARM_TUNE_PQ:
        raise ValueError(f"Unknown export arm {arm}")
    # The exported candidate must appear in the run's own frozen tuning table,
    # so an ablation cannot invent an unmeasured candidate.
    key = expected_candidate(arm)
    if key not in selection.get("tune_counts", {}):
        raise ValueError(f"{key} was not evaluated by the completed run")
    config = dict(arm=arm, epoch=4, conditioned=False, mode="refined_constrained",
                  priority=2 if arm == "proposal_centered" else 1)
    if arm == "proposal_centered" and selection["config"] != config:
        raise ValueError("Frozen candidate configuration changed")
    return config


def load_frozen_selection(input_root):
    candidates = [path for path in Path(input_root).rglob("selected_config.json")
                  if json.loads(path.read_text()).get("run_identity") == EXPECTED_RUN_IDENTITY]
    if len(candidates) != 1:
        raise ValueError("Expected exactly one frozen V11 selection document")
    return json.loads(candidates[0].read_text())


def export(runtime, model, config, output, filename):
    """Identical geometry, threshold, ownership, and 180-image coverage as V11."""
    model.eval()
    rows, image_counts = [], {}
    for index, path in enumerate(sorted(runtime["base"].CFG.test_dir.glob("*.jpeg")), 1):
        predictions = v11.predict_modes(runtime, model, path, "test",
                                        config["conditioned"], [config["mode"]])[config["mode"]]
        image_counts[path.name] = len(predictions)
        occupied = np.zeros((2048, 2048), dtype=bool)
        for number, rle in enumerate(predictions, 1):
            mask = mask_utils.decode(rle).astype(bool)
            if mask.sum() < 20 or (mask & occupied).any():
                raise ValueError("Invalid test prediction area or pixel ownership")
            occupied |= mask
            rows.append(dict(filament_id=f"{path.stem}_{number}",
                             segmentation_rle=rle["counts"].decode()))
        if index % 20 == 0:
            print(f"export={filename} images={index}/180 rows={len(rows)}", flush=True)
    if len(image_counts) != 180:
        raise ValueError("Incomplete test coverage during export")
    destination = output / filename
    pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]).to_csv(destination, index=False)
    return dict(file=filename, rows=len(rows), sha256=v11.sha256(destination),
                evaluated_images=180,
                empty_images=[name for name, count in image_counts.items() if not count],
                structural_validation="all masks decoded, area>=20, per-image ownership disjoint")


def run_export():
    global v11, v10
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working"))
    args = parser.parse_args()
    if not MODULE_SHA256 or not CHECKPOINT_SHA256:
        raise ValueError("Run scripts/prepare_v11_export.py to stage a pinned export")
    arm = ARM
    candidate = expected_candidate(arm)
    import sys
    v11 = sys.modules["v11_refiner"]
    v10 = sys.modules["v10_quality"]
    v11.RAW_CACHE_SHA256 = RAW_CACHE_SHA256
    v11.MODULE_SHA256 = MODULE_SHA256
    import torch
    args.output.mkdir(parents=True, exist_ok=True)
    identity = v11.digest_json(dict(script_sha256=v11.sha256(Path(__file__)), modules=MODULE_SHA256,
                                    checkpoint_sha256=CHECKPOINT_SHA256,
                                    raw_cache_sha256=RAW_CACHE_SHA256, source_run=SOURCE_RUN,
                                    candidate=candidate))
    manifest_path = args.output / "export_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous.get("export_identity") != identity:
            raise ValueError("Output directory belongs to a different export")
        if previous.get("complete"):
            print(json.dumps(dict(already_complete=True, candidate=candidate)), flush=True)
            return
    elif any((args.output / name).exists() for name in ("submission.csv", "submission_control.csv")):
        raise ValueError("Unsigned existing submission artifacts require a fresh output directory")
    # Reuses every V11 input check: source hashes, split, crop geometry, all 321
    # raw caches, frozen control diagnostics, and exact library versions.
    runtime = v11.load_runtime(args)
    selection = load_frozen_selection(args.input_root)
    config = verify_frozen_selection(selection, arm)
    checkpoint = locate_checkpoint(args.input_root, CHECKPOINT_SHA256, arm)
    checkpoint_identity = verify_checkpoint_identity(checkpoint)
    if checkpoint_identity["arm_file"] != f"{arm}_e4.pt":
        raise ValueError("Recovered checkpoint does not match the declared arm")
    manifest = dict(experiment="v11_export_only", source_run=SOURCE_RUN,
                    export_identity=identity, candidate=candidate, arm=arm, config=config,
                    frozen_tune_pq=ARM_TUNE_PQ[arm],
                    checkpoint_sha256=CHECKPOINT_SHA256, checkpoint_identity=checkpoint_identity,
                    gate=GATE, promotable=False, trained_here=False,
                    threshold=v11.THRESHOLD, confidence=v11.CONFIDENCE,
                    min_prediction_area=20, detector_resolution=1024,
                    detector_views=4, crop_views=4, crop_size=v11.CROP_SIZE,
                    versions=runtime["versions"], complete=False,
                    public_score="unmeasured", automatic_submission=False,
                    note="Gate-failed candidate exported for measurement only; not promoted")
    v11.write_json(manifest_path, manifest)
    # Control first: its byte hash must equal the accepted 0.38 submission, which
    # proves this runtime reproduces the established pipeline exactly.
    control_model = v11.build_model(runtime["base"], runtime["refiner"])
    control_export = export(runtime, control_model, dict(conditioned=False, mode="refined_constrained"),
                            args.output, "submission_control.csv")
    v10.verify_control_export(args.output / "submission_control.csv")
    print(json.dumps(dict(control_verified=True, **control_export)), flush=True)
    del control_model
    torch.cuda.empty_cache()
    candidate_model = v11.build_model(runtime["base"], runtime["refiner"],
                                      config["conditioned"], checkpoint)
    candidate_export = export(runtime, candidate_model, config, args.output, "submission.csv")
    del candidate_model
    torch.cuda.empty_cache()
    if candidate_export["sha256"] == control_export["sha256"]:
        raise ValueError("Candidate export is byte-identical to the control; nothing was measured")
    v11.write_json(args.output / "submission_candidates.json",
                   [dict(control_export, candidate=CONTROL, control_verified=True),
                    dict(candidate_export, candidate=candidate, promotable=False,
                         gate=GATE, full_control_validated=True,
                         requires_independent_csv_validation=True)])
    manifest.update(complete=True, full_control_validated=True,
                    control_export=control_export, candidate_export=candidate_export)
    v11.write_json(manifest_path, manifest)
    print(json.dumps(dict(candidate=candidate, promotable=False,
                          rows=candidate_export["rows"], sha256=candidate_export["sha256"],
                          complete=True)), flush=True)


if __name__ == "__main__":
    run_export()
