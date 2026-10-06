"""V22: four-GPU PointRend Mask R-CNN for solar-filament instance segmentation.

Modes:
  verify    Validate environment, data populations, split isolation, and frozen config.
  smoke     One worst-case forward/backward step on one GPU.
  probe     First 100 global DDP steps on all four GPUs.
  train     Resume the probe checkpoint and finish the frozen 4,700-step run.
  evaluate  Evaluate four fixed checkpoints in parallel on tune; open audit once.

This script never exports a test CSV and never submits to Kaggle.
Design contract: docs/v22_design_2026-09-12.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE = Path(os.environ.get("V22_BASE", "/mnt/solar-v22-pointrend"))
V20_BASE = Path(os.environ.get("V20_BASE", "/mnt/solar-v20-condinst"))
PREP = V20_BASE / "prep"
if not PREP.is_dir():
    PREP = PROJECT_ROOT / "outputs" / "v20_condinst_prep"
IMAGE_ROOT = V20_BASE / "prep" / "normalized_images"
TRAIN_JSON = PREP / "condinst_train.json"
TUNE_JSON = PREP / "condinst_tune.json"
AUDIT_JSON = PREP / "condinst_audit.json"
POINTREND_ROOT = V20_BASE / "src" / "detectron2" / "projects" / "PointRend"
CONFIG = POINTREND_ROOT / "configs" / "InstanceSegmentation" / "pointrend_rcnn_R_50_FPN_1x_coco.yaml"
WEIGHTS = BASE / "src" / "pointrend_rcnn_R_50_FPN_1x_coco.pkl"
OUTPUT = BASE / "outputs" / "v22_pointrend"
STATUS = BASE / "operation_status.json"

WEIGHTS_SHA256 = "2a658c57f565b1badc0e324790b86d50d60f2c30139a85e33b93d0e4746897af"
SEED = 42
TRAIN_RECORDS = 937
GLOBAL_BATCH = 4
EPOCHS = 20
STEPS_PER_EPOCH = math.ceil(TRAIN_RECORDS / GLOBAL_BATCH)
MAX_ITER = STEPS_PER_EPOCH * EPOCHS
CHECKPOINT_EPOCHS = (5, 10, 15, 20)
CHECKPOINT_PERIOD = STEPS_PER_EPOCH * 5
PROBE_STEPS = 100
CONFIDENCES = tuple(round(0.05 * value, 2) for value in range(1, 11))
SUBMIT_BAR_TUNE = 0.448703
SUBMIT_BAR_AUDIT = 0.470763
PROMOTION_GATE_TUNE = 0.55
EXPECTED = {
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


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def population(path: Path) -> dict:
    data = read_json(path)
    return {
        "files": len({image["file_name"] for image in data["images"]}),
        "records": len(data["images"]),
        "annotations": len(data["annotations"]),
    }


def split_filenames(path: Path) -> set[str]:
    return {image["file_name"] for image in read_json(path)["images"]}


def assert_inputs() -> None:
    paths = {"train": TRAIN_JSON, "tune": TUNE_JSON, "audit": AUDIT_JSON}
    for group, path in paths.items():
        if not path.is_file():
            raise RuntimeError(f"missing {group} COCO data: {path}")
        actual = population(path)
        if actual != EXPECTED[group]:
            raise RuntimeError(f"{group} population mismatch: {actual} != {EXPECTED[group]}")
    groups = {group: split_filenames(path) for group, path in paths.items()}
    if groups["train"] & groups["tune"]:
        raise RuntimeError("physical filename leakage between train and tune")
    if groups["train"] & groups["audit"]:
        raise RuntimeError("physical filename leakage between train and audit")
    if groups["tune"] & groups["audit"]:
        raise RuntimeError("physical filename leakage between tune and audit")
    if len(list(IMAGE_ROOT.glob("*.jpeg"))) != 707:
        raise RuntimeError("normalized image population is not exactly 707")
    if sha256(WEIGHTS) != WEIGHTS_SHA256:
        raise RuntimeError("official PointRend checkpoint identity mismatch")
    if not CONFIG.is_file():
        raise RuntimeError(f"missing official PointRend config: {CONFIG}")


def register_datasets(smoke: bool = False) -> None:
    from detectron2.data import DatasetCatalog
    from detectron2.data.datasets import register_coco_instances

    for name, path in (
        ("v22_train", TRAIN_JSON),
        ("v22_tune", TUNE_JSON),
        ("v22_audit", AUDIT_JSON),
    ):
        if name not in DatasetCatalog.list():
            register_coco_instances(name, {}, str(path), str(IMAGE_ROOT))
    if smoke and "v22_smoke" not in DatasetCatalog.list():
        records = DatasetCatalog.get("v22_train")
        densest = max(records, key=lambda record: len(record.get("annotations", [])))
        DatasetCatalog.register("v22_smoke", lambda record=densest: [record])


def build_cfg(train_dataset: str = "v22_train"):
    from detectron2.config import get_cfg
    from point_rend import add_pointrend_config

    cfg = get_cfg()
    add_pointrend_config(cfg)
    cfg.merge_from_file(str(CONFIG))
    cfg.SEED = SEED
    cfg.OUTPUT_DIR = str(OUTPUT)
    cfg.DATASETS.TRAIN = (train_dataset,)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True
    cfg.MODEL.DEVICE = "cuda"
    cfg.MODEL.WEIGHTS = str(WEIGHTS)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.POINT_HEAD.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = min(CONFIDENCES)
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.25
    cfg.MODEL.POINT_HEAD.SUBDIVISION_STEPS = 5
    cfg.MODEL.POINT_HEAD.SUBDIVISION_NUM_POINTS = 784
    cfg.MODEL.POINT_HEAD.TRAIN_NUM_POINTS = 196
    cfg.MODEL.POINT_HEAD.OVERSAMPLE_RATIO = 3
    cfg.MODEL.POINT_HEAD.IMPORTANCE_SAMPLE_RATIO = 0.75
    cfg.INPUT.MIN_SIZE_TRAIN = (1024,)
    cfg.INPUT.MAX_SIZE_TRAIN = 1024
    cfg.INPUT.MIN_SIZE_TEST = 1024
    cfg.INPUT.MAX_SIZE_TEST = 1024
    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.INPUT.FORMAT = "BGR"
    cfg.SOLVER.IMS_PER_BATCH = GLOBAL_BATCH
    cfg.SOLVER.BASE_LR = 0.0025
    cfg.SOLVER.MAX_ITER = MAX_ITER
    cfg.SOLVER.STEPS = (STEPS_PER_EPOCH * 15, STEPS_PER_EPOCH * 18)
    cfg.SOLVER.WARMUP_ITERS = 200
    cfg.SOLVER.WARMUP_FACTOR = 0.001
    cfg.SOLVER.CHECKPOINT_PERIOD = CHECKPOINT_PERIOD
    cfg.SOLVER.AMP.ENABLED = True
    cfg.SOLVER.WEIGHT_DECAY = 1e-4
    cfg.SOLVER.MOMENTUM = 0.9
    cfg.TEST.EVAL_PERIOD = 0
    cfg.TEST.DETECTIONS_PER_IMAGE = 100
    cfg.freeze()
    return cfg


def frozen_config() -> dict:
    cfg = build_cfg()
    return {
        "architecture": "PointRend Mask R-CNN R50-FPN",
        "weights_sha256": WEIGHTS_SHA256,
        "train_records": TRAIN_RECORDS,
        "global_batch": GLOBAL_BATCH,
        "epochs": EPOCHS,
        "steps_per_epoch": STEPS_PER_EPOCH,
        "max_iter": MAX_ITER,
        "checkpoint_epochs": list(CHECKPOINT_EPOCHS),
        "input_size": 1024,
        "mask_output_size": 224,
        "point_subdivision_steps": cfg.MODEL.POINT_HEAD.SUBDIVISION_STEPS,
        "point_subdivision_points": cfg.MODEL.POINT_HEAD.SUBDIVISION_NUM_POINTS,
        "point_train_points": cfg.MODEL.POINT_HEAD.TRAIN_NUM_POINTS,
        "point_oversample_ratio": cfg.MODEL.POINT_HEAD.OVERSAMPLE_RATIO,
        "point_importance_ratio": cfg.MODEL.POINT_HEAD.IMPORTANCE_SAMPLE_RATIO,
        "base_lr": cfg.SOLVER.BASE_LR,
        "lr_steps": list(cfg.SOLVER.STEPS),
        "warmup_iters": cfg.SOLVER.WARMUP_ITERS,
        "confidence_grid": list(CONFIDENCES),
        "nms_threshold": cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
        "detections_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
        "submit_bar_tune": SUBMIT_BAR_TUNE,
        "submit_bar_audit": SUBMIT_BAR_AUDIT,
        "promotion_gate_tune": PROMOTION_GATE_TUNE,
    }


def verify() -> None:
    import detectron2
    import point_rend
    import torch

    assert_inputs()
    config = frozen_config()
    if config["max_iter"] != 4700 or config["steps_per_epoch"] != 235:
        raise RuntimeError("frozen schedule arithmetic drift")
    report = {
        "status": "verified",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpus": torch.cuda.device_count(),
        "detectron2": str(Path(detectron2.__file__).resolve()),
        "point_rend": str(Path(point_rend.__file__).resolve()),
        "populations": {
            group: population(path)
            for group, path in (("train", TRAIN_JSON), ("tune", TUNE_JSON), ("audit", AUDIT_JSON))
        },
        "source_sha256": {
            path.name: sha256(path) for path in (TRAIN_JSON, TUNE_JSON, AUDIT_JSON, CONFIG, WEIGHTS)
        },
        "config": config,
        "submits": False,
        "exports_test_csv": False,
    }
    write_json(BASE / "verification.json", report)
    print(json.dumps(report, indent=2))


def latest_total_loss(trainer) -> float:
    value = trainer.storage.history("total_loss").latest()
    return float(value)


def run_trainer(mode: str) -> None:
    import torch
    from detectron2.engine import DefaultTrainer
    from detectron2.utils import comm

    smoke = mode == "smoke"
    register_datasets(smoke=smoke)
    cfg = build_cfg("v22_smoke" if smoke else "v22_train")
    if smoke:
        cfg.defrost()
        cfg.SOLVER.IMS_PER_BATCH = 1
        cfg.SOLVER.MAX_ITER = 1
        cfg.SOLVER.STEPS = ()
        cfg.SOLVER.CHECKPOINT_PERIOD = 1
        cfg.OUTPUT_DIR = str(BASE / "outputs" / "v22_smoke")
        cfg.freeze()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    trainer = DefaultTrainer(cfg)
    resume = mode in {"probe", "train"} and (OUTPUT / "last_checkpoint").is_file()
    trainer.resume_or_load(resume=resume)
    start = trainer.start_iter
    target = 1 if smoke else (PROBE_STEPS if mode == "probe" else MAX_ITER)
    if start > target:
        raise RuntimeError(f"checkpoint iteration {start} is beyond requested target {target}")
    trainer.max_iter = target
    started = time.monotonic()
    if start < target:
        trainer.train()
    elapsed = time.monotonic() - started
    is_main = comm.is_main_process()
    total_loss = (
        latest_total_loss(trainer) if is_main and start < target else float("nan")
    )
    peak = torch.cuda.max_memory_allocated() / (1024**3)
    if is_main and start < target and not math.isfinite(total_loss):
        raise RuntimeError(f"non-finite total loss: {total_loss}")
    if smoke and peak > 14.0:
        raise RuntimeError(f"memory gate failed: peak {peak:.2f} GiB")
    comm.synchronize()
    if mode == "probe" and start < target and is_main:
        framework_checkpoint = OUTPUT / "model_final.pth"
        if not framework_checkpoint.is_file():
            raise RuntimeError("Detectron2 did not write its end-of-probe checkpoint")
        named_probe = OUTPUT / "model_probe_0100.pth"
        shutil.copy2(framework_checkpoint, named_probe)
        (OUTPUT / "last_checkpoint").write_text(named_probe.name)
    comm.synchronize()
    result = {
        "mode": mode,
        "start_iter": start,
        "target_iter": target,
        "steps_run": max(target - start, 0),
        "seconds": elapsed,
        "seconds_per_global_step": elapsed / max(target - start, 1),
        "projected_total_hours": (elapsed / max(target - start, 1)) * MAX_ITER / 3600,
        "peak_memory_gib_rank0": peak,
        "last_total_loss": total_loss,
    }
    comm.synchronize()
    if is_main:
        write_json(BASE / "outputs" / f"v22_{mode}_result.json", result)
        write_json(STATUS, {"state": f"{mode}_complete", **result})
        print("V22_RESULT", json.dumps(result), flush=True)
        if mode == "probe" and result["projected_total_hours"] > 6.0:
            raise RuntimeError("runtime gate failed: projected training exceeds six hours")


def annotation_records(path: Path) -> tuple[dict[str, list[list[dict]]], dict[str, dict]]:
    data = read_json(path)
    annotations: dict[int, list[dict]] = defaultdict(list)
    for annotation in data["annotations"]:
        annotations[annotation["image_id"]].append(annotation)
    targets: dict[str, list[list[dict]]] = defaultdict(list)
    info: dict[str, dict] = {}
    for image in data["images"]:
        targets[image["file_name"]].append(annotations[image["id"]])
        info[image["file_name"]] = image
    return dict(targets), info


def decode_annotations(annotations: list[dict], height: int, width: int) -> list[np.ndarray]:
    masks = []
    for annotation in annotations:
        segmentation = annotation["segmentation"]
        if isinstance(segmentation, dict):
            rle = segmentation.copy()
            if isinstance(rle["counts"], str):
                rle["counts"] = rle["counts"].encode("utf-8")
        else:
            rle = mask_utils.merge(mask_utils.frPyObjects(segmentation, height, width))
        masks.append(mask_utils.decode(rle).astype(bool))
    return masks


def ownership_masks(masks: np.ndarray, scores: np.ndarray, threshold: float) -> list[np.ndarray]:
    if len(masks) != len(scores):
        raise ValueError("mask and score count mismatch")
    if not len(masks):
        return []
    order = np.argsort(-scores, kind="mergesort")
    occupied = np.zeros(masks.shape[1:], dtype=bool)
    accepted = []
    for index in order:
        if float(scores[index]) < threshold:
            continue
        residual = masks[index].astype(bool) & ~occupied
        if int(residual.sum()) < 20:
            continue
        accepted.append(residual)
        occupied |= residual
    return accepted


def pq_counts(predictions: list[np.ndarray], targets: list[np.ndarray]) -> list[float]:
    if not predictions or not targets:
        return [0.0, 0, len(predictions), len(targets)]
    pred_rles = mask_utils.encode(np.asfortranarray(np.stack(predictions, axis=-1).astype(np.uint8)))
    target_rles = mask_utils.encode(np.asfortranarray(np.stack(targets, axis=-1).astype(np.uint8)))
    iou = np.asarray(mask_utils.iou(pred_rles, target_rles, [0] * len(targets))).T
    hits = iou > 0.5
    return [
        float(iou[hits].sum()),
        int(hits.sum()),
        int((~hits.any(axis=0)).sum()),
        int((~hits.any(axis=1)).sum()),
    ]


def metrics(total: list[float]) -> dict:
    iou_sum, tp, fp, fn = total
    denominator = tp + 0.5 * (fp + fn)
    sq = iou_sum / tp if tp else 0.0
    rq = tp / denominator if denominator else 0.0
    return {
        "pq": iou_sum / denominator if denominator else 0.0,
        "sq": sq,
        "rq": rq,
        "tp_iou": iou_sum,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


def checkpoint_candidates() -> list[tuple[int, Path]]:
    expected = [(epoch, STEPS_PER_EPOCH * epoch - 1) for epoch in CHECKPOINT_EPOCHS]
    candidates = []
    for epoch, iteration in expected:
        path = OUTPUT / f"model_{iteration:07d}.pth"
        if not path.is_file() and epoch == EPOCHS:
            path = OUTPUT / "model_final.pth"
        if not path.is_file():
            raise RuntimeError(f"missing epoch-{epoch} checkpoint: {path}")
        candidates.append((epoch, path))
    return candidates


def infer_group(checkpoint: Path, split_path: Path, thresholds: tuple[float, ...]) -> dict:
    from detectron2.engine import DefaultPredictor

    register_datasets()
    cfg = build_cfg()
    cfg.defrost()
    cfg.MODEL.WEIGHTS = str(checkpoint)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = min(thresholds)
    cfg.freeze()
    predictor = DefaultPredictor(cfg)
    targets_by_file, info = annotation_records(split_path)
    totals = {threshold: [0.0, 0, 0, 0] for threshold in thresholds}
    for index, name in enumerate(sorted(targets_by_file), 1):
        image = cv2.imread(str(IMAGE_ROOT / name))
        if image is None:
            raise RuntimeError(f"could not read normalized image {name}")
        instances = predictor(image)["instances"].to("cpu")
        masks = (
            instances.pred_masks.numpy()
            if len(instances)
            else np.zeros((0, image.shape[0], image.shape[1]), dtype=bool)
        )
        scores = instances.scores.numpy() if len(instances) else np.zeros(0, dtype=np.float32)
        for threshold in thresholds:
            predictions = ownership_masks(masks, scores, threshold)
            for annotations in targets_by_file[name]:
                targets = decode_annotations(annotations, info[name]["height"], info[name]["width"])
                counts = pq_counts(predictions, targets)
                for position, value in enumerate(counts):
                    totals[threshold][position] += value
        if index % 20 == 0 or index == len(targets_by_file):
            print(f"{checkpoint.name}: inferred {index}/{len(targets_by_file)}", flush=True)
    return {
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": sha256(checkpoint),
        "split": split_path.name,
        "thresholds": list(thresholds),
        "results": {str(threshold): metrics(total) for threshold, total in totals.items()},
    }


def evaluation_worker(checkpoint: Path, split: str, thresholds: tuple[float, ...], out: Path) -> None:
    split_path = {"tune": TUNE_JSON, "audit": AUDIT_JSON}[split]
    write_json(out, infer_group(checkpoint, split_path, thresholds))


def spawn_worker(checkpoint: Path, gpu: int, split: str, thresholds: tuple[float, ...], out: Path):
    log = out.with_suffix(".log")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["OMP_NUM_THREADS"] = "1"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "evaluate-worker",
        "--checkpoint",
        str(checkpoint),
        "--split",
        split,
        "--thresholds",
        ",".join(map(str, thresholds)),
        "--out",
        str(out),
    ]
    handle = log.open("w")
    process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=env)
    return process, handle, log


def select_tune_candidate(tune_runs: list[tuple[int, dict]]) -> tuple[int, float, dict]:
    choices = []
    for epoch, run in tune_runs:
        for threshold, result in run["results"].items():
            choices.append((result["pq"], float(threshold), -epoch, epoch, run))
    if not choices:
        raise ValueError("no tune candidates")
    _, confidence, _, epoch, run = max(choices, key=lambda item: item[:3])
    return epoch, confidence, run


def evaluate() -> None:
    assert_inputs()
    report_path = OUTPUT / "v22_evaluation.json"
    if report_path.is_file():
        print(report_path.read_text())
        return
    parallel = OUTPUT / "parallel_tune"
    parallel.mkdir(parents=True, exist_ok=True)
    workers = []
    for gpu, (epoch, checkpoint) in enumerate(checkpoint_candidates()):
        out = parallel / f"epoch_{epoch}.json"
        process, handle, log = spawn_worker(checkpoint, gpu, "tune", CONFIDENCES, out)
        workers.append((epoch, checkpoint, out, process, handle, log))
    failures = []
    for epoch, checkpoint, out, process, handle, log in workers:
        returncode = process.wait()
        handle.close()
        if returncode != 0 or not out.is_file():
            failures.append({"epoch": epoch, "checkpoint": checkpoint.name, "returncode": returncode, "log": str(log)})
    if failures:
        write_json(STATUS, {"state": "evaluation_tune_failed", "failures": failures})
        raise RuntimeError(f"parallel tune evaluation failed: {failures}")
    tune_runs = [(epoch, json.loads(out.read_text())) for epoch, _, out, _, _, _ in workers]
    selected_epoch, selected_confidence, selected_run = select_tune_candidate(tune_runs)
    selected_checkpoint = next(path for epoch, path in checkpoint_candidates() if epoch == selected_epoch)
    audit_marker = OUTPUT / "audit_opened.json"
    if audit_marker.exists():
        raise RuntimeError("audit has already been opened without a completed evaluation report")
    write_json(audit_marker, {
        "checkpoint": selected_checkpoint.name,
        "checkpoint_sha256": sha256(selected_checkpoint),
        "epoch": selected_epoch,
        "confidence": selected_confidence,
        "opened_unix": time.time(),
    })
    audit_out = OUTPUT / "audit_selected.json"
    process, handle, log = spawn_worker(
        selected_checkpoint, 0, "audit", (selected_confidence,), audit_out
    )
    returncode = process.wait()
    handle.close()
    if returncode != 0 or not audit_out.is_file():
        write_json(STATUS, {"state": "evaluation_audit_failed", "returncode": returncode, "log": str(log)})
        raise RuntimeError(f"selected audit evaluation failed; inspect {log}")
    audit_run = json.loads(audit_out.read_text())
    selected_tune = selected_run["results"][str(selected_confidence)]
    selected_audit = audit_run["results"][str(selected_confidence)]
    authorized = (
        selected_tune["pq"] >= SUBMIT_BAR_TUNE
        and selected_audit["pq"] >= SUBMIT_BAR_AUDIT
    )
    report = {
        "status": "complete",
        "architecture": "PointRend Mask R-CNN R50-FPN",
        "selection": "epochs 5/10/15/20 x confidence 0.05-0.50 on tune; audit selected pair once",
        "selected_epoch": selected_epoch,
        "selected_checkpoint": selected_checkpoint.name,
        "selected_checkpoint_sha256": sha256(selected_checkpoint),
        "selected_confidence": selected_confidence,
        "selected_tune": selected_tune,
        "selected_audit": selected_audit,
        "submit_bar_tune": SUBMIT_BAR_TUNE,
        "submit_bar_audit": SUBMIT_BAR_AUDIT,
        "promotion_gate_tune": PROMOTION_GATE_TUNE,
        "export_authorized": authorized,
        "promotion_gate_cleared": authorized and selected_tune["pq"] >= PROMOTION_GATE_TUNE,
        "populations": {
            group: population(path)
            for group, path in (("train", TRAIN_JSON), ("tune", TUNE_JSON), ("audit", AUDIT_JSON))
        },
        "tune_runs": [{"epoch": epoch, **run} for epoch, run in tune_runs],
        "audit_selected_run": audit_run,
        "action": "prepare_validated_export_for_authorization" if authorized else "close_v22_no_export",
    }
    write_json(report_path, report)
    write_json(STATUS, {"state": "evaluation_complete", **report})
    print(json.dumps(report, indent=2))


def parse_thresholds(value: str) -> tuple[float, ...]:
    result = tuple(float(item) for item in value.split(",") if item)
    if not result or any(item not in CONFIDENCES for item in result):
        raise ValueError(f"thresholds outside frozen grid: {result}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("verify", "smoke", "probe", "train", "evaluate", "evaluate-worker")
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--split", choices=("tune", "audit"))
    parser.add_argument("--thresholds")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    random.seed(SEED)
    np.random.seed(SEED)
    if args.mode == "verify":
        verify()
    elif args.mode == "evaluate":
        evaluate()
    elif args.mode == "evaluate-worker":
        if not all((args.checkpoint, args.split, args.thresholds, args.out)):
            parser.error("evaluate-worker requires checkpoint, split, thresholds, and out")
        evaluation_worker(args.checkpoint, args.split, parse_thresholds(args.thresholds), args.out)
    else:
        assert_inputs()
        from detectron2.engine import launch

        write_json(STATUS, {"state": f"{args.mode}_running", "started_unix": time.time()})
        launch(
            run_trainer,
            num_gpus_per_machine=1 if args.mode == "smoke" else 4,
            num_machines=1,
            machine_rank=0,
            dist_url="auto",
            args=(args.mode,),
        )


if __name__ == "__main__":
    main()
