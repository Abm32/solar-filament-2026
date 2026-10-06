"""V19: Mask2Former training for solar filament instance segmentation.

A genuinely different detector family from the YOLO-seg line (V6-V18), per
docs/v19_design_2026-09-09.md. Runs either as a Kaggle kernel (competition
data pre-mounted read-only at /kaggle/input/competitions, staged via
scripts/prepare_v19_mask2former.py) or on a standalone GPU VM (Azure
NCASv3_T4, downloading data via the Kaggle CLI) if that quota clears later --
this script detects which environment it's in via KAGGLE_KERNEL_RUN_TYPE and
adjusts dependency installation and data staging accordingly. Either way it
converts the project's COCO annotations into the split Mask2Former/
transformers expects, and (once --full is implemented) will train and
evaluate with this project's own PQ metric (src/pq_eval.py) on the SAME
tune/audit split convention used by every V6-V18 run, so results are directly
comparable to the accepted V17 baseline (tune 0.448703, public 0.39).

Usage:

    python scripts/train_v19_mask2former.py --smoke        # memory/shape check first
    python scripts/train_v19_mask2former.py --full          # full training run

The frozen-control discipline used throughout this project (V11-V18) doesn't
apply verbatim here since there is no prior Mask2Former artifact to reproduce
byte-for-byte. What DOES carry over: the same train/tune/audit file split, the
same PQ implementation, and the same SUBMIT_BAR / PROMOTION_GATE thresholds
declared in docs/v19_design_2026-09-09.md.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Predeclared protocol constants (docs/v19_design_2026-09-09.md)
# ---------------------------------------------------------------------------
SUBMIT_BAR_TUNE = 0.448703          # V17's accepted tune PQ
SUBMIT_BAR_AUDIT = 0.470763         # V17's corresponding audit PQ
PROMOTION_GATE_TUNE = 0.55          # large, unambiguous local jump
IMG_SIZE_TRAIN = 1024
IMG_SIZE_INFER = 2048
BACKBONE = "facebook/mask2former-swin-tiny-coco-instance"  # HF checkpoint; swapped to
# resnet50 config below if the swin memory footprint fails the smoke test.
BATCH_SIZE = 2
GRAD_ACCUM = 4
NUM_QUERIES = 100
SEED = 42


def _sh(cmd: list[str], **kwargs) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kwargs)


def ensure_environment() -> None:
    """Install the dependency set this script needs. Idempotent.

    Skips torch/torchvision on Kaggle kernels (KAGGLE_KERNEL_RUN_TYPE is set):
    the Kaggle GPU docker image ships a CUDA-matched torch build already, and
    reinstalling a pinned wheel risks breaking that match. On a standalone VM
    (no Kaggle env markers) install everything explicitly.
    """
    import os

    packages = ["transformers==4.44.2", "pycocotools==2.0.8", "scipy==1.15.3"]
    # scipy is required by Mask2FormerLoss's Hungarian-matching criterion
    # (transformers.models.mask2former.modeling_mask2former imports
    # scipy.optimize.linear_sum_assignment) but was missing from both this
    # list and the Kaggle kernel image's pre-installed set; Kaggle's base
    # image happens to ship scipy already, which is why this was invisible
    # until running on a standalone VM.
    if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        packages = [
            "torch==2.4.1",
            "torchvision==0.19.1",
            *packages,
            "opencv-python-headless==4.10.0.84",
            "pandas==2.2.2",
            "kaggle==1.6.17",
        ]
    _sh([sys.executable, "-m", "pip", "install", "--quiet", *packages])


def ensure_data(data_root: Path) -> Path:
    """Return the directory containing `MAGFiLO_1.0_Kaggle_2026/`.

    On a Kaggle kernel the competition is already mounted read-only (this
    script's staging adds `competition_sources`); `data_root` there is
    `/kaggle/input/competitions`, which already contains
    `filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/...` -- nothing to
    download, and the mount is read-only so a download would fail anyway.
    On a standalone VM, pull it via the Kaggle CLI if not already staged.
    """
    marker = data_root / "filament-segmentation-2026" / "MAGFiLO_1.0_Kaggle_2026"
    if marker.exists():
        print(f"data already staged at {marker}")
        return data_root
    data_root.mkdir(parents=True, exist_ok=True)
    _sh([
        "kaggle", "competitions", "download",
        "-c", "filament-segmentation-2026",
        "-p", str(data_root),
    ])
    zips = list(data_root.glob("*.zip"))
    for z in zips:
        _sh(["unzip", "-q", "-o", str(z), "-d", str(data_root)])
    if not marker.exists():
        raise RuntimeError(f"Download completed but expected data not found at {marker}")
    return data_root


@dataclass
class Split:
    train_files: list[str]
    tune_files: list[str]
    audit_files: list[str]


def load_split(coco_path: Path) -> Split:
    """Reproduce this project's train/tune/audit file split from the COCO json.

    This is the SAME algorithm as kaggle_inference/optimize_inference.py's
    validation_split(), used unchanged by every run V6-V18: sort files by
    name, Random(42).shuffle, take the first 20% as held-out, the first 94
    of held-out as tune, the rest as audit. It is deterministic given the
    same COCO file, so it does not need a manifest lookup -- reproducing it
    directly here (rather than depending on an outputs/ manifest key that
    does not actually exist in this project's manifests) is what keeps V19's
    tune/audit PQ comparable to V17's 0.448703 baseline.
    """
    import random

    from src.coco import best_annotator_image_ids, index_coco, load_coco

    coco = load_coco(coco_path)
    images, annotations = index_coco(coco)
    best_ids = best_annotator_image_ids(images, annotations)
    files = sorted(best_ids)
    random.Random(SEED).shuffle(files)
    held_out = files[:round(len(files) * 0.2)]
    tune_files = held_out[:94]
    audit_files = held_out[94:]
    train_files = [name for name in files if name not in set(held_out)]
    assert set(tune_files).isdisjoint(audit_files)
    assert set(train_files).isdisjoint(held_out)
    return Split(train_files=train_files, tune_files=tune_files, audit_files=audit_files)


def build_hf_dataset(coco_path: Path, image_dir: Path, split: Split):
    """Convert COCO annotations into Mask2Former records.

    Retain every annotator image_id for each file. V6 and all canonical V9-V18
    PQ evaluations score one file-level prediction against every supplied
    annotator record; collapsing file_name to one image_id silently changed the
    target population (728 vs 1,134 tune GT instances) and made V19's metric
    incomparable to its V17 submission bar.
    """
    from src.coco import index_coco, load_coco  # project's own loader

    coco = load_coco(coco_path)
    images, annotations = index_coco(coco)
    file_to_ids: dict[str, list[int]] = {}
    for image_id, info in images.items():
        file_to_ids.setdefault(info["file_name"], []).append(image_id)

    def subset(file_names: list[str]) -> list[dict]:
        records = []
        for file_name in file_names:
            for image_id in sorted(file_to_ids.get(file_name, [])):
                records.append({
                    "file_name": file_name,
                    "image_id": image_id,
                    "image_path": str(image_dir / file_name),
                    "annotations": annotations.get(image_id, []),
                })
        return records

    return {
        "train": subset(split.train_files),
        "tune": subset(split.tune_files),
        "audit": subset(split.audit_files),
    }


def run_smoke_test(dataset: dict, out_dir: Path) -> None:
    """Memory/shape probe on the WORST case (max-instance-density image), per
    the V18 post-mortem: never conclude a per-batch memory ceiling from a
    random subsample."""
    train = dataset["train"]
    if not train:
        raise RuntimeError("empty train split; check data staging")
    densest = max(train, key=lambda r: len(r["annotations"]))
    print(f"worst-case smoke image: {densest['file_name']} "
          f"({len(densest['annotations'])} instances)")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available -- smoke test must run on the GPU VM, not locally"
        )

    from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
    from pycocotools import mask as mask_utils
    import cv2
    import numpy as np

    processor = Mask2FormerImageProcessor(
        size={"height": IMG_SIZE_TRAIN, "width": IMG_SIZE_TRAIN},
        do_resize=True,
        do_normalize=True,
        num_labels=1,  # single class: filament
        ignore_index=0,  # background id in instance_map is 0; must be dropped
        # from all_labels before the instance_id_to_semantic_id lookup, or
        # convert_segmentation_map_to_binary_masks tries to look up id 0
        # (background) in a mapping that only has real instance ids.
    )
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        BACKBONE,
        num_labels=1,
        num_queries=NUM_QUERIES,
        ignore_mismatched_sizes=True,
    ).cuda()

    image = cv2.imread(densest["image_path"])
    if image is None:
        raise RuntimeError(f"could not read {densest['image_path']}")
    height, width = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Mask2FormerImageProcessor expects ONE single-channel instance-ID map per
    # image (background id = 0), plus instance_id_to_semantic_id -- NOT a list
    # of per-instance binary masks. Burn each decoded annotation mask into a
    # shared canvas under its own instance id (ids start at 1; 0 stays
    # background), decoding REAL annotation polygons/RLE (same path as
    # scripts/audit_annotation_rasterization.py's mask_utils.frPyObjects), not
    # placeholders -- realistic per-pixel content plus the real instance COUNT
    # is what makes this probe trustworthy.
    instance_map = np.zeros((height, width), dtype=np.int32)
    instance_id_to_semantic_id = {}
    for instance_id, ann in enumerate(densest["annotations"], start=1):
        rle = mask_utils.merge(mask_utils.frPyObjects(ann["segmentation"], height, width))
        decoded = mask_utils.decode(rle).astype(bool)
        instance_map[decoded] = instance_id
        instance_id_to_semantic_id[instance_id] = 0  # single class: filament

    raw_inputs = processor(
        images=[image] * BATCH_SIZE,
        segmentation_maps=[instance_map] * BATCH_SIZE,
        instance_id_to_semantic_id=instance_id_to_semantic_id,
        return_tensors="pt",
    )
    inputs = {}
    for key, value in raw_inputs.items():
        if hasattr(value, "cuda"):
            inputs[key] = value.cuda()
        elif isinstance(value, list) and value and hasattr(value[0], "cuda"):
            # mask_labels / class_labels are lists of per-image tensors (they
            # don't share a common size across images, so encode_inputs can't
            # batch them into one tensor) -- .cuda() must be applied per
            # element, not to the list itself, or they silently stay on CPU
            # and grid_sample raises a cuda/cpu device mismatch deep in the
            # Mask2Former loss.
            inputs[key] = [item.cuda() for item in value]
        else:
            inputs[key] = value

    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.float16):
        outputs = model(**inputs)
    peak_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"smoke test peak GPU memory: {peak_gib:.2f} GiB (T4 budget: 14.56 GiB)")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "v19_smoke_result.json").write_text(json.dumps({
        "densest_image": densest["file_name"],
        "instance_count": len(densest["annotations"]),
        "batch_size": BATCH_SIZE,
        "peak_gpu_memory_gib": peak_gib,
        "backbone": BACKBONE,
        "cleared_t4_budget": peak_gib < 13.0,  # leave headroom, per V18 lesson
    }, indent=2))

    if peak_gib >= 13.0:
        raise RuntimeError(
            f"Smoke test peak memory {peak_gib:.2f} GiB leaves insufficient "
            "headroom on a T4 (14.56 GiB). Reduce batch size, num_queries, or "
            "switch BACKBONE to a resnet50 config before attempting a full run. "
            "This mirrors V18's exact failure mode -- do not override this check."
        )
    print("Smoke test cleared T4 memory budget. Safe to launch full run.")


# ---------------------------------------------------------------------------
# Full training (only reachable after --smoke has cleared the memory budget)
# ---------------------------------------------------------------------------

EPOCHS = 20
LR = 1e-4
WEIGHT_DECAY = 1e-4
MASK_THRESHOLD = 0.5
PQ_IOU_THRESHOLD = 0.5
WARMUP_FRACTION = 0.05  # linear warmup over the first 5% of optimizer steps,
# then cosine decay to 0 -- see docs/v19_regression_2026-09-10.md. Flat
# lr=1e-4 for 20 epochs on a partially-reinitialized pretrained transformer
# (2-class head reinit, backbone pretrained) produced a loss plateau with an
# oscillating, non-converging tune PQ (E029); warmup avoids destabilizing the
# pretrained backbone weights in the first few hundred steps.


def _load_instance_map(record: dict) -> tuple:
    """Decode one record's REAL annotation polygons/RLE into a single
    instance-ID canvas + instance_id_to_semantic_id, same construction as
    run_smoke_test but for an arbitrary record instead of only the densest
    training image."""
    from pycocotools import mask as mask_utils
    import cv2
    import numpy as np

    image = cv2.imread(record["image_path"])
    if image is None:
        raise RuntimeError(f"could not read {record['image_path']}")
    height, width = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    instance_map = np.zeros((height, width), dtype=np.int32)
    instance_id_to_semantic_id = {}
    for instance_id, ann in enumerate(record["annotations"], start=1):
        rle = mask_utils.merge(mask_utils.frPyObjects(ann["segmentation"], height, width))
        decoded = mask_utils.decode(rle).astype(bool)
        instance_map[decoded] = instance_id
        instance_id_to_semantic_id[instance_id] = 0
    return image, instance_map, instance_id_to_semantic_id


def _batches(records: list, batch_size: int):
    for start in range(0, len(records), batch_size):
        yield records[start:start + batch_size]


def train_one_epoch(model, processor, train_records, optimizer, batch_size, grad_accum, device,
                     scheduler=None):
    import torch

    model.train()
    total_loss, n_batches = 0.0, 0
    optimizer.zero_grad()
    for step, batch in enumerate(_batches(train_records, batch_size)):
        images, instance_maps, id_maps = [], [], []
        for record in batch:
            image, instance_map, id_map = _load_instance_map(record)
            images.append(image)
            instance_maps.append(instance_map)
            id_maps.append(id_map)
        raw_inputs = processor(
            images=images, segmentation_maps=instance_maps,
            instance_id_to_semantic_id=id_maps, ignore_index=0,
            size={"height": IMG_SIZE_TRAIN, "width": IMG_SIZE_TRAIN},
            return_tensors="pt",
        )
        inputs = {}
        for key, value in raw_inputs.items():
            if hasattr(value, "to"):
                inputs[key] = value.to(device)
            elif isinstance(value, list) and value and hasattr(value[0], "to"):
                inputs[key] = [item.to(device) for item in value]
            else:
                inputs[key] = value

        with torch.autocast("cuda", dtype=torch.float16):
            outputs = model(**inputs)
            loss = outputs.loss / grad_accum
        loss.backward()
        if (step + 1) % grad_accum == 0:
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()
        total_loss += outputs.loss.item()
        n_batches += 1
    optimizer.step()  # flush any partial accumulation window at epoch end
    optimizer.zero_grad()
    if scheduler is not None:
        scheduler.step()
    return total_loss / max(n_batches, 1)


def predict_instances(model, processor, image, device, infer_size, mask_threshold=MASK_THRESHOLD):
    """Run inference on one image, resized to infer_size, and return a list
    of (binary_mask_at_infer_size, score) pairs above mask_threshold."""
    import torch

    with torch.no_grad():
        inputs = processor(
            images=[image], size={"height": infer_size, "width": infer_size}, return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.autocast("cuda", dtype=torch.float16):
            outputs = model(**inputs)
        result = processor.post_process_instance_segmentation(
            outputs, threshold=mask_threshold, target_sizes=[(infer_size, infer_size)],
        )[0]
    masks, scores = [], []
    segmentation = result["segmentation"].cpu().numpy()
    for segment in result["segments_info"]:
        masks.append(segmentation == segment["id"])
        scores.append(float(segment["score"]))
    return masks, scores


def evaluate_pq(model, processor, records, device, infer_size):
    """Direct IoU-matrix PQ, matching kaggle_inference/optimize_inference.py's
    pq_counts/pq_value exactly (not the RLE/DataFrame path in src/pq_eval.py,
    which is for scoring already-exported submission CSVs) -- avoids an RLE
    round-trip during in-training evaluation while staying numerically
    identical: same IoU>0.5 hit rule, same tp/fp/fn accounting."""
    from pycocotools import mask as mask_utils
    import numpy as np

    model.eval()
    total_tp_iou, total_tp, total_fp, total_fn = 0.0, 0, 0, 0
    for record in records:
        image, gt_instance_map, _ = _load_instance_map(record)
        height, width = gt_instance_map.shape
        gt_ids = [i for i in np.unique(gt_instance_map) if i != 0]
        gt_masks = [gt_instance_map == gid for gid in gt_ids]

        pred_masks, _ = predict_instances(model, processor, image, device, infer_size)
        # Predictions come back at infer_size; GT is at native resolution --
        # resize predictions up to match before computing IoU, same
        # convention every prior version uses (predict at a working
        # resolution, score against native-resolution GT).
        if infer_size != height or infer_size != width:
            import cv2
            pred_masks = [
                cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
                for mask in pred_masks
            ]

        n_gt, n_pred = len(gt_masks), len(pred_masks)
        if n_gt == 0:
            total_fp += n_pred
            continue
        if n_pred == 0:
            total_fn += n_gt
            continue

        gt_rles = mask_utils.encode(np.asfortranarray(np.stack(gt_masks, axis=-1).astype(np.uint8)))
        pred_rles = mask_utils.encode(np.asfortranarray(np.stack(pred_masks, axis=-1).astype(np.uint8)))
        iou = mask_utils.iou(pred_rles, gt_rles, [0] * n_gt)  # [n_pred, n_gt]
        iou = np.asarray(iou).T  # -> [n_gt, n_pred], matching pq_counts' convention
        hits = iou > PQ_IOU_THRESHOLD
        total_tp_iou += float(iou[hits].sum())
        total_tp += int(hits.sum())
        total_fp += int((~hits.any(axis=0)).sum())
        total_fn += int((~hits.any(axis=1)).sum())

    denominator = total_tp + 0.5 * (total_fp + total_fn)
    pq = float(total_tp_iou / denominator) if denominator else 0.0
    return dict(pq=pq, tp_iou=total_tp_iou, tp=total_tp, fp=total_fp, fn=total_fn)


def run_full_training(dataset: dict, out_dir: Path) -> dict:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available -- full training must run on the GPU VM")

    from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

    torch.manual_seed(SEED)
    device = torch.device("cuda")

    processor = Mask2FormerImageProcessor(
        do_resize=True, do_normalize=True, num_labels=1, ignore_index=0,
    )
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        BACKBONE, num_labels=1, num_queries=NUM_QUERIES, ignore_mismatched_sizes=True,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Linear warmup -> cosine decay across the full run's optimizer steps
    # (not epochs): flat lr=1e-4 for 20 epochs on a mostly-pretrained
    # backbone + freshly-reinitialized 2-class head produced a loss plateau
    # with an oscillating, non-converging tune PQ (E029, epoch range
    # 0.075-0.154 with no visible trend). One optimizer step happens every
    # GRAD_ACCUM micro-batches, matching train_one_epoch's step condition.
    steps_per_epoch = max(1, -(-len(dataset["train"]) // BATCH_SIZE) // GRAD_ACCUM)
    total_steps = max(1, steps_per_epoch * EPOCHS)
    warmup_steps = max(1, round(total_steps * WARMUP_FRACTION))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        import math
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_tune_pq, best_state_path = -1.0, out_dir / "best_model"
    import time

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.monotonic()
        train_loss = train_one_epoch(
            model, processor, dataset["train"], optimizer, BATCH_SIZE, GRAD_ACCUM, device,
            scheduler=scheduler,
        )
        # Model selection MUST evaluate at the resolution the model was
        # trained at (IMG_SIZE_TRAIN), not IMG_SIZE_INFER. E029's run
        # evaluated every epoch's tune PQ at 2048 -- 2x IMG_SIZE_TRAIN=1024 --
        # via plain single-shot upsampling, not the sliding-window inference
        # the design doc specified. That scale mismatch, on every single
        # epoch, is the most likely driver of both the outsized FP volume
        # (5-10x the YOLO control) and the non-converging PQ trace: the
        # metric used to pick the "best" checkpoint was never measuring the
        # model at a scale it had seen during training. 2048 sliding-window
        # inference belongs at final-submission time only (see
        # docs/v19_regression_2026-09-10.md), not in the per-epoch loop.
        tune_result = evaluate_pq(model, processor, dataset["tune"], device, IMG_SIZE_TRAIN)
        epoch_seconds = time.monotonic() - epoch_start
        entry = dict(epoch=epoch, train_loss=train_loss, tune_pq=tune_result["pq"],
                     tune_tp=tune_result["tp"], tune_fp=tune_result["fp"], tune_fn=tune_result["fn"],
                     epoch_seconds=epoch_seconds)
        history.append(entry)
        print(f"epoch={epoch}/{EPOCHS} train_loss={train_loss:.4f} "
              f"tune_pq={tune_result['pq']:.6f} (tp={tune_result['tp']} "
              f"fp={tune_result['fp']} fn={tune_result['fn']}) "
              f"epoch_seconds={epoch_seconds:.1f}", flush=True)
        (out_dir / "training_history.json").write_text(json.dumps(history, indent=2))
        if tune_result["pq"] > best_tune_pq:
            best_tune_pq = tune_result["pq"]
            model.save_pretrained(best_state_path)

    # Reload the best-tune checkpoint before the final audit-set evaluation,
    # so the reported audit PQ corresponds to the same weights SUBMIT_BAR/
    # PROMOTION_GATE are checked against, not whatever the last epoch left in
    # memory (mirrors the project's existing best-checkpoint-by-tune-PQ
    # selection convention from the V6-V18 refiner runs). Audit is evaluated
    # at IMG_SIZE_TRAIN too, for the same reason tune is -- comparing
    # SUBMIT_BAR/PROMOTION_GATE against a metric computed at training
    # resolution keeps model-selection and gate-checking internally
    # consistent. A separate IMG_SIZE_INFER (2048, sliding-window) pass is
    # deferred to submission-export time, once/if SUBMIT_BAR clears.
    from transformers import Mask2FormerForUniversalSegmentation as _M2F
    model = _M2F.from_pretrained(best_state_path).to(device)
    audit_result = evaluate_pq(model, processor, dataset["audit"], device, IMG_SIZE_TRAIN)

    result = dict(
        best_tune_pq=best_tune_pq, audit_pq=audit_result["pq"],
        audit_tp=audit_result["tp"], audit_fp=audit_result["fp"], audit_fn=audit_result["fn"],
        submit_bar_tune=SUBMIT_BAR_TUNE, submit_bar_audit=SUBMIT_BAR_AUDIT,
        promotion_gate_tune=PROMOTION_GATE_TUNE,
        cleared_submit_bar=(best_tune_pq >= SUBMIT_BAR_TUNE
                            and audit_result["pq"] >= SUBMIT_BAR_AUDIT),
        cleared_promotion_gate=best_tune_pq >= PROMOTION_GATE_TUNE,
        eval_resolution=IMG_SIZE_TRAIN,
        history=history,
    )
    (out_dir / "v19_full_result.json").write_text(json.dumps(result, indent=2))
    print(f"FINAL: best_tune_pq={best_tune_pq:.6f} audit_pq={audit_result['pq']:.6f} "
          f"cleared_submit_bar={result['cleared_submit_bar']} "
          f"cleared_promotion_gate={result['cleared_promotion_gate']}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run memory/shape probe only")
    parser.add_argument("--full", action="store_true", help="Run full training")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override EPOCHS (e.g. for a short timing probe before "
                             "committing GPU quota to a full run)")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "raw")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs" / "v19_mask2former")
    args = parser.parse_args()

    if not args.smoke and not args.full:
        parser.error("pass --smoke or --full")

    if args.epochs is not None:
        global EPOCHS
        EPOCHS = args.epochs

    ensure_environment()
    data_root = ensure_data(args.data_root)

    coco_path = (
        data_root / "filament-segmentation-2026" / "MAGFiLO_1.0_Kaggle_2026"
        / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    )
    image_dir = (
        data_root / "filament-segmentation-2026" / "MAGFiLO_1.0_Kaggle_2026"
        / "train" / "train_images"
    )
    split = load_split(coco_path)
    dataset = build_hf_dataset(coco_path, image_dir, split)
    print(f"split sizes: train={len(dataset['train'])} "
          f"tune={len(dataset['tune'])} audit={len(dataset['audit'])}")

    if args.smoke:
        run_smoke_test(dataset, args.out_dir)
        return

    run_full_training(dataset, args.out_dir)


if __name__ == "__main__":
    main()
