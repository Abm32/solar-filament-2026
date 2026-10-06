"""V20 CondInst training and canonical PQ evaluation on the isolated Azure runtime.

Modes:
  prepare-images  Robustly normalize approved competition train images on /mnt.
  smoke           One worst-case forward/backward step on one GPU.
  probe           100 global DDP steps using the full 20-epoch schedule.
  train           Resume the probe checkpoint and finish 20 epochs on four GPUs.
  evaluate        Tune-select checkpoint/confidence; audit selected candidate once.

No mode exports or submits a Kaggle CSV.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils

BASE = Path(os.environ.get("V20_BASE", "/mnt/solar-v20-condinst"))
PREP = BASE / "prep"
IMAGE_ROOT = PREP / "normalized_images"
OUTPUT = BASE / "outputs" / "v20_condinst"
ADET = BASE / "src" / "AdelaiDet"
CONFIG = ADET / "configs" / "CondInst" / "MS_R_50_1x.yaml"
WEIGHTS = BASE / "src" / "CondInst_MS_R_50_1x.pth"
TRAIN_JSON = PREP / "condinst_train.json"
TUNE_JSON = PREP / "condinst_tune.json"
AUDIT_JSON = PREP / "condinst_audit.json"
SOURCE_IMAGES = Path(os.environ.get(
    "V20_SOURCE_IMAGES",
    "/home/azureuser/solar-filament-2026/data/raw/filament-segmentation-2026/"
    "MAGFiLO_1.0_Kaggle_2026/train/train_images",
))

SEED = 42
TRAIN_RECORDS = 937
GLOBAL_BATCH = 4
EPOCHS = 20
STEPS_PER_EPOCH = (TRAIN_RECORDS + GLOBAL_BATCH - 1) // GLOBAL_BATCH
MAX_ITER = STEPS_PER_EPOCH * EPOCHS  # 4,700 including the final partial conceptual batch
CHECKPOINT_PERIOD = STEPS_PER_EPOCH * 5
PROBE_STEPS = 100
CONFIDENCES = tuple(round(0.05 * i, 2) for i in range(1, 11))
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


def population(path: Path) -> dict:
    data = read_json(path)
    return {
        "files": len({image["file_name"] for image in data["images"]}),
        "records": len(data["images"]),
        "annotations": len(data["annotations"]),
    }


def assert_inputs() -> None:
    for group, path in (("train", TRAIN_JSON), ("tune", TUNE_JSON), ("audit", AUDIT_JSON)):
        actual = population(path)
        if actual != EXPECTED[group]:
            raise RuntimeError(f"{group} population mismatch: {actual} != {EXPECTED[group]}")
    groups = {
        group: {image["file_name"] for image in read_json(path)["images"]}
        for group, path in (("train", TRAIN_JSON), ("tune", TUNE_JSON), ("audit", AUDIT_JSON))
    }
    if groups["train"] & groups["tune"] or groups["train"] & groups["audit"] or groups["tune"] & groups["audit"]:
        raise RuntimeError("physical filename leakage across splits")


def normalize_one(source: Path, target: Path) -> None:
    gray = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError(f"could not read {source}")
    values = gray[gray > 0]
    if values.size < gray.size // 4:
        raise RuntimeError(f"invalid solar disk in {source}")
    lo, hi = np.percentile(values, (2, 98))
    normalized = np.clip((gray.astype(np.float32) - lo) * 255.0 / max(hi - lo, 1.0), 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise RuntimeError(f"could not write {target}")


def prepare_images() -> None:
    assert_inputs()
    names = sorted({
        image["file_name"]
        for path in (TRAIN_JSON, TUNE_JSON, AUDIT_JSON)
        for image in read_json(path)["images"]
    })
    started = time.monotonic()
    for index, name in enumerate(names, 1):
        target = IMAGE_ROOT / name
        if not target.exists():
            normalize_one(SOURCE_IMAGES / name, target)
        if index % 100 == 0:
            print(f"normalized {index}/{len(names)}", flush=True)
    if len(list(IMAGE_ROOT.glob("*.jpeg"))) != 707:
        raise RuntimeError("normalized image count is not 707")
    manifest = {
        "images": len(names),
        "seconds": time.monotonic() - started,
        "method": "grayscale disk nonzero p2-p98 -> uint8, replicated BGR, JPEG quality 95",
        "source_json_sha256": {p.name: sha256(p) for p in (TRAIN_JSON, TUNE_JSON, AUDIT_JSON)},
    }
    (PREP / "normalized_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def register_datasets(smoke: bool = False) -> None:
    from detectron2.data.datasets import register_coco_instances
    from detectron2.data import DatasetCatalog

    datasets = (("v20_train", TRAIN_JSON), ("v20_tune", TUNE_JSON), ("v20_audit", AUDIT_JSON))
    for name, path in datasets:
        if name not in DatasetCatalog.list():
            register_coco_instances(name, {}, str(path), str(IMAGE_ROOT))
    if smoke:
        records = DatasetCatalog.get("v20_train")
        densest = max(records, key=lambda record: len(record.get("annotations", [])))
        DatasetCatalog.register("v20_smoke", lambda d=[densest]: d)


def build_cfg(train_dataset: str = "v20_train"):
    from adet.config import get_cfg

    cfg = get_cfg()
    cfg.merge_from_file(str(CONFIG))
    cfg.SEED = SEED
    cfg.OUTPUT_DIR = str(OUTPUT)
    cfg.DATASETS.TRAIN = (train_dataset,)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True
    cfg.MODEL.DEVICE = "cuda"
    cfg.MODEL.WEIGHTS = str(WEIGHTS)
    cfg.MODEL.FCOS.NUM_CLASSES = 1
    cfg.MODEL.FCOS.NMS_TH = 0.6
    cfg.MODEL.FCOS.INFERENCE_TH_TRAIN = 0.05
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = 0.05
    cfg.MODEL.FCOS.POST_NMS_TOPK_TEST = 100
    cfg.MODEL.FCOS.LOSS_ALPHA = 0.25
    cfg.MODEL.FCOS.LOSS_GAMMA = 2.0
    cfg.MODEL.RESNETS.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.FPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.CONDINST.MASK_BRANCH.IN_FEATURES = ["p2", "p3", "p4", "p5"]
    cfg.MODEL.CONDINST.MASK_BRANCH.SEMANTIC_LOSS_ON = True
    cfg.MODEL.CONDINST.MASK_BRANCH.NORM = "GN"
    cfg.MODEL.CONDINST.MASK_OUT_STRIDE = 4
    cfg.MODEL.CONDINST.TOPK_PROPOSALS_PER_IM = 64
    cfg.MODEL.CONDINST.MAX_PROPOSALS = -1
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
    cfg.SOLVER.CHECKPOINT_PERIOD = CHECKPOINT_PERIOD
    cfg.SOLVER.AMP.ENABLED = True
    cfg.SOLVER.WEIGHT_DECAY = 1e-4
    cfg.TEST.EVAL_PERIOD = 0
    cfg.freeze()
    return cfg


def run_trainer(mode: str) -> None:
    import torch
    from detectron2.engine import DefaultTrainer

    smoke = mode == "smoke"
    register_datasets(smoke=smoke)
    cfg = build_cfg("v20_smoke" if smoke else "v20_train")
    if smoke:
        cfg.defrost()
        cfg.SOLVER.IMS_PER_BATCH = 1
        cfg.SOLVER.MAX_ITER = 1
        cfg.SOLVER.STEPS = ()
        cfg.OUTPUT_DIR = str(BASE / "outputs" / "v20_smoke")
        cfg.freeze()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=(mode in {"probe", "train"} and (OUTPUT / "last_checkpoint").exists()))
    start = trainer.start_iter
    target = 1 if smoke else (PROBE_STEPS if mode == "probe" else MAX_ITER)
    trainer.max_iter = target
    started = time.monotonic()
    trainer.train()
    elapsed = time.monotonic() - started
    peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    if mode == "probe":
        trainer.checkpointer.save("model_probe_0100", iteration=PROBE_STEPS - 1)
    result = {
        "mode": mode, "start_iter": start, "target_iter": target,
        "steps_run": target - start, "seconds": elapsed,
        "seconds_per_global_step": elapsed / max(target - start, 1),
        "peak_memory_gib_rank0": peak,
        "max_iter": MAX_ITER, "steps_per_epoch": STEPS_PER_EPOCH,
        "projected_total_hours": (elapsed / max(target - start, 1)) * MAX_ITER / 3600,
    }
    from detectron2.utils import comm
    comm.synchronize()
    if comm.is_main_process():
        path = (BASE / "outputs" / f"v20_{mode}_result.json")
        path.write_text(json.dumps(result, indent=2) + "\n")
        print("V20_RESULT", json.dumps(result), flush=True)
    if smoke and peak > 14.0:
        raise RuntimeError(f"memory gate failed: peak {peak:.2f} GiB leaves <2 GiB physical headroom")


def annotation_records(path: Path) -> tuple[dict[str, list[list[dict]]], dict[str, dict]]:
    data = read_json(path)
    annotations = defaultdict(list)
    for ann in data["annotations"]:
        annotations[ann["image_id"]].append(ann)
    targets, info = defaultdict(list), {}
    for image in data["images"]:
        targets[image["file_name"]].append(annotations[image["id"]])
        info[image["file_name"]] = image
    return dict(targets), info


def decode_annotations(annotations: list[dict], height: int, width: int) -> list[np.ndarray]:
    masks = []
    for ann in annotations:
        rle = mask_utils.merge(mask_utils.frPyObjects(ann["segmentation"], height, width))
        masks.append(mask_utils.decode(rle).astype(bool))
    return masks


def ownership_masks(masks: np.ndarray, scores: np.ndarray, threshold: float) -> list[np.ndarray]:
    order = np.argsort(-scores)
    occupied = np.zeros(masks.shape[1:], dtype=bool)
    accepted = []
    for index in order:
        if float(scores[index]) < threshold:
            continue
        residual = masks[index].astype(bool) & ~occupied
        if residual.sum() < 20:
            continue
        accepted.append(residual)
        occupied |= residual
    return accepted


def pq_counts(predictions: list[np.ndarray], targets: list[np.ndarray]) -> list[float]:
    if not predictions or not targets:
        return [0.0, 0, len(predictions), len(targets)]
    pred_rles = mask_utils.encode(np.asfortranarray(np.stack(predictions, -1).astype(np.uint8)))
    gt_rles = mask_utils.encode(np.asfortranarray(np.stack(targets, -1).astype(np.uint8)))
    iou = np.asarray(mask_utils.iou(pred_rles, gt_rles, [0] * len(targets))).T
    hits = iou > 0.5
    return [float(iou[hits].sum()), int(hits.sum()), int((~hits.any(0)).sum()), int((~hits.any(1)).sum())]


def pq(total: list[float]) -> float:
    denom = total[1] + 0.5 * (total[2] + total[3])
    return total[0] / denom if denom else 0.0


def checkpoint_candidates() -> list[Path]:
    import re

    numbered = []
    for path in OUTPUT.glob("model_*.pth"):
        match = re.fullmatch(r"model_(\d{7})\.pth", path.name)
        if match:
            numbered.append((int(match.group(1)), path))
    numbered.sort()
    expected_iterations = [STEPS_PER_EPOCH * epoch - 1 for epoch in (5, 10, 15, 20)]
    by_iteration = {iteration: path for iteration, path in numbered}
    missing = [iteration for iteration in expected_iterations if iteration not in by_iteration]
    if missing:
        raise RuntimeError(
            f"missing fixed epoch checkpoints {missing}; found {sorted(by_iteration)}"
        )
    return [by_iteration[iteration] for iteration in expected_iterations]


def infer_group(checkpoint: Path, path: Path,
                thresholds: tuple[float, ...] = CONFIDENCES) -> dict:
    from detectron2.engine import DefaultPredictor

    register_datasets()
    cfg = build_cfg()
    cfg.defrost()
    cfg.MODEL.WEIGHTS = str(checkpoint)
    cfg.freeze()
    predictor = DefaultPredictor(cfg)
    targets_by_file, info = annotation_records(path)
    totals = {threshold: [0.0, 0, 0, 0] for threshold in thresholds}
    per_file = []
    for index, name in enumerate(sorted(targets_by_file), 1):
        image = cv2.imread(str(IMAGE_ROOT / name))
        outputs = predictor(image)["instances"].to("cpu")
        masks = outputs.pred_masks.numpy() if len(outputs) else np.zeros((0, image.shape[0], image.shape[1]), bool)
        scores = outputs.scores.numpy() if len(outputs) else np.zeros(0, np.float32)
        row = {"file_name": name, "raw_predictions": len(outputs)}
        for threshold in thresholds:
            predictions = ownership_masks(masks, scores, threshold)
            subtotal = [0.0, 0, 0, 0]
            for annotations in targets_by_file[name]:
                targets = decode_annotations(annotations, info[name]["height"], info[name]["width"])
                counts = pq_counts(predictions, targets)
                for j, value in enumerate(counts):
                    totals[threshold][j] += value
                    subtotal[j] += value
            row[str(threshold)] = subtotal
        per_file.append(row)
        if index % 20 == 0:
            print(f"{checkpoint.name}: inferred {index}/{len(targets_by_file)}", flush=True)
    return {
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": sha256(checkpoint),
        "results": {
            str(threshold): {"pq": pq(total), "tp_iou": total[0], "tp": int(total[1]), "fp": int(total[2]), "fn": int(total[3])}
            for threshold, total in totals.items()
        },
        "per_file": per_file,
    }


def evaluate() -> None:
    assert_inputs()
    checkpoints = checkpoint_candidates()
    tune_runs = [infer_group(checkpoint, TUNE_JSON) for checkpoint in checkpoints]
    choices = []
    for run in tune_runs:
        for threshold, result in run["results"].items():
            choices.append((result["pq"], float(threshold), run))
    best_pq, best_threshold, best_run = max(choices, key=lambda item: (item[0], item[1]))
    selected_checkpoint = OUTPUT / best_run["checkpoint"]
    # Audit is opened exactly once, only for the tune-selected checkpoint and confidence.
    audit_run = infer_group(selected_checkpoint, AUDIT_JSON, (best_threshold,))
    selected_tune = best_run["results"][str(best_threshold)]
    selected_audit = audit_run["results"][str(best_threshold)]
    authorized = selected_tune["pq"] >= SUBMIT_BAR_TUNE and selected_audit["pq"] >= SUBMIT_BAR_AUDIT
    report = {
        "status": "complete",
        "selection": "four fixed checkpoints x confidence 0.05-0.50 on tune; audit selected pair once",
        "nms_threshold": 0.6,
        "selected_checkpoint": selected_checkpoint.name,
        "selected_checkpoint_sha256": sha256(selected_checkpoint),
        "selected_confidence": best_threshold,
        "selected_tune": selected_tune,
        "selected_audit": selected_audit,
        "submit_bar_tune": SUBMIT_BAR_TUNE,
        "submit_bar_audit": SUBMIT_BAR_AUDIT,
        "promotion_gate_tune": PROMOTION_GATE_TUNE,
        "export_authorized": authorized,
        "promotion_gate_cleared": selected_tune["pq"] >= PROMOTION_GATE_TUNE,
        "populations": {g: population(p) for g, p in (("train", TRAIN_JSON), ("tune", TUNE_JSON), ("audit", AUDIT_JSON))},
        "tune_runs": [{k: v for k, v in run.items() if k != "per_file"} for run in tune_runs],
        "audit_selected_run": {k: v for k, v in audit_run.items() if k != "per_file"},
    }
    (OUTPUT / "v20_evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print("V20_EVALUATION", json.dumps(report, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare-images", "smoke", "probe", "train", "evaluate"))
    args = parser.parse_args()
    random.seed(SEED); np.random.seed(SEED)
    assert_inputs()
    if args.mode == "prepare-images":
        prepare_images()
    elif args.mode == "evaluate":
        evaluate()
    else:
        from detectron2.engine import launch
        num_gpus = 1 if args.mode == "smoke" else 4
        launch(
            run_trainer,
            num_gpus_per_machine=num_gpus,
            num_machines=1,
            machine_rank=0,
            dist_url="auto",
            args=(args.mode,),
        )


if __name__ == "__main__":
    main()
