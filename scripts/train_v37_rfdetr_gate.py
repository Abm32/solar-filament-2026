"""V37 G0/G1: Azure-T4 fit-only RF-DETR-Seg-Nano resource gate.

This runner reads exactly 16 canonical fit annotator records and nine fit JPEGs,
performs one fixed-resolution epoch, and cannot read tune/audit/test, export CSV,
or submit. It must execute on a Tesla T4 and writes a JSON report on all exits.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import random
import shutil
import subprocess
import threading
import time
import traceback
import warnings
from collections import defaultdict
from pathlib import Path

ANNOTATION_SHA256 = "5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1"
CHECKPOINT_SHA256 = "a44613a4ecd6b5ba61a62002c600b0b6cb7a9da2936a45317ec4b62c635fb99b"
CHECKPOINT_MD5 = "9995497791d0ff1664a1d9ddee9cfd20"
WHEEL_SHA256 = "96fa7a8ecbcaeea5f5150ff0b503699d2d4d75a32a87eeabfc0ab90485076a52"
SELECTION_ID = "e561c523b60993e93f82ce03116737ce512f54f4103b4557c29e409c88c92653"
RFDETR_VERSION = "1.10.1"
RECORDS = 16
ANNOTATIONS = 57
PHYSICAL_IMAGES = 9
RESOLUTION = 768
MASK_DOWNSAMPLE_RATIO = 2
MAX_EPOCHS = 1
BATCH_SIZE = 1
MEMORY_GATE_MIB = 14336
TIME_GATE_SECONDS = 1200
SEED = 20260914


def digest(path: Path, algorithm: str = "sha256") -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def selection_records(document: dict) -> list[dict]:
    images = {str(row["id"]): row for row in document["images"]}
    annotations: dict[str, list[dict]] = defaultdict(list)
    by_file: dict[str, list[str]] = defaultdict(list)
    for row in document["annotations"]:
        annotations[str(row["image_id"])].append(row)
    for image_id, info in images.items():
        by_file[info["file_name"]].append(image_id)
    files = sorted(by_file); random.Random(42).shuffle(files)
    fit = sorted(files[141:])
    pairs = [(name, image_id) for name in fit for image_id in sorted(by_file[name])][:RECORDS]
    records = [{"file_name": name, "image_id": image_id,
                "annotations": len(annotations[image_id])} for name, image_id in pairs]
    identity = hashlib.sha256(json.dumps(
        records, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    if identity != SELECTION_ID:
        raise RuntimeError(f"V37 selection changed: {identity}")
    if len(records) != RECORDS or sum(row["annotations"] for row in records) != ANNOTATIONS:
        raise RuntimeError("V37 record population changed")
    if len({row["file_name"] for row in records}) != PHYSICAL_IMAGES:
        raise RuntimeError("V37 physical-image population changed")
    return records


def decode_mask(mask_utils, info: dict, annotation: dict):
    return mask_utils.decode(mask_utils.merge(mask_utils.frPyObjects(
        annotation["segmentation"], int(info["height"]), int(info["width"]),
    ))).astype(bool)


def build_dataset(annotation_path: Path, image_dir: Path, output: Path) -> dict:
    if digest(annotation_path) != ANNOTATION_SHA256:
        raise RuntimeError("canonical annotation identity changed")
    lowered = f"{annotation_path} {image_dir}".lower()
    forbidden = ("public_magfilo", "magfilo-v1-0-segmentation-of-solar-filaments",
                 "/test/", "test_images", "audit")
    if any(token in lowered for token in forbidden):
        raise RuntimeError("V37 data firewall rejected an input path")
    from pycocotools import mask as mask_utils

    document = json.loads(annotation_path.read_text())
    images = {str(row["id"]): row for row in document["images"]}
    annotations: dict[str, list[dict]] = defaultdict(list)
    for row in document["annotations"]:
        annotations[str(row["image_id"])].append(row)
    records = selection_records(document)
    if output.exists():
        raise RuntimeError(f"refusing to overwrite V37 dataset: {output}")
    split_documents = {}
    source_names = set()
    for split in ("train", "valid", "test"):
        split_dir = output / split; split_dir.mkdir(parents=True)
        converted_images, converted_annotations = [], []
        annotation_index = 1
        for index, record in enumerate(records, 1):
            source_names.add(record["file_name"])
            source = image_dir / record["file_name"]
            if not source.is_file():
                raise FileNotFoundError(source)
            alias = f"{index:02d}_{record['image_id']}.jpeg"
            shutil.copy2(source, split_dir / alias)
            info = images[record["image_id"]]
            converted_images.append({
                "id": index, "file_name": alias,
                "width": int(info["width"]), "height": int(info["height"]),
                "source_image_id": record["image_id"],
                "source_file_name": record["file_name"],
            })
            for source_annotation in annotations[record["image_id"]]:
                mask = decode_mask(mask_utils, info, source_annotation)
                ys, xs = mask.nonzero()
                if not len(xs):
                    raise RuntimeError("empty source annotation")
                x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
                segmentation = source_annotation["segmentation"]
                converted_annotations.append({
                    "id": annotation_index, "image_id": index, "category_id": 1,
                    "segmentation": segmentation, "area": int(mask.sum()),
                    "bbox": [x0, y0, x1 - x0, y1 - y0], "iscrowd": 0,
                    "source_annotation_id": str(source_annotation["id"]),
                })
                annotation_index += 1
        converted = {
            "info": {"description": "V37 fit-only numerical gate"},
            "licenses": [], "categories": [{"id": 1, "name": "filament", "supercategory": "none"}],
            "images": converted_images, "annotations": converted_annotations,
        }
        if len(converted_images) != RECORDS or len(converted_annotations) != ANNOTATIONS:
            raise RuntimeError(f"converted {split} population changed")
        annotation_file = split_dir / "_annotations.coco.json"
        annotation_file.write_text(json.dumps(converted, separators=(",", ":")) + "\n")
        split_documents[split] = {
            "records": len(converted_images), "annotations": len(converted_annotations),
            "json_sha256": digest(annotation_file),
        }
    manifest = {
        "selection_id": SELECTION_ID, "records": records,
        "physical_source_names": sorted(source_names), "splits": split_documents,
        "reads_tune": False, "reads_audit": False, "reads_test": False,
    }
    write_json(output / "manifest.json", manifest)
    return manifest


class MemoryMonitor:
    def __init__(self) -> None:
        self.peak_mib = 0
        self.samples = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                result = subprocess.run([
                    "nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                ], check=True, capture_output=True, text=True, timeout=5)
                values = [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]
                if values:
                    self.peak_mib = max(self.peak_mib, max(values)); self.samples += 1
            except Exception:
                pass
            self._stop.wait(0.25)

    def __enter__(self):
        self._thread.start(); return self

    def __exit__(self, *_):
        self._stop.set(); self._thread.join(timeout=10)


def package_versions() -> dict:
    names = ("rfdetr", "torch", "torchvision", "transformers", "pytorch-lightning",
             "torchmetrics", "supervision", "pydantic", "numpy", "scipy", "pycocotools")
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def finite_losses(output_dir: Path) -> dict:
    candidates = sorted(output_dir.rglob("metrics.csv"))
    values = []
    rows = 0
    for path in candidates:
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                rows += 1
                for key, raw in row.items():
                    if "loss" not in key.lower() or raw in (None, ""):
                        continue
                    try:
                        value = float(raw)
                    except ValueError:
                        continue
                    values.append(value)
    return {
        "metric_files": [str(path) for path in candidates], "metric_rows": rows,
        "loss_values": len(values), "all_finite": bool(values) and all(math.isfinite(v) for v in values),
        "min_loss": min(values) if values else None, "max_loss": max(values) if values else None,
    }


def run(args) -> dict:
    started = time.monotonic()
    result = {
        "status": "started", "experiment": "v37_rfdetr_seg_nano_azure_g1",
        "selection_id": SELECTION_ID, "resolution": RESOLUTION,
        "mask_downsample_ratio": MASK_DOWNSAMPLE_RATIO, "epochs": MAX_EPOCHS,
        "batch_size": BATCH_SIZE, "memory_gate_mib": MEMORY_GATE_MIB,
        "time_gate_seconds": TIME_GATE_SECONDS, "package_versions": package_versions(),
        "reads_tune": False, "reads_audit": False, "reads_test": False,
        "exports": False, "submits": False,
    }
    monitor = MemoryMonitor()
    try:
        if result["package_versions"]["rfdetr"] != RFDETR_VERSION:
            raise RuntimeError(f"expected rfdetr {RFDETR_VERSION}")
        if digest(args.checkpoint) != CHECKPOINT_SHA256 or digest(args.checkpoint, "md5") != CHECKPOINT_MD5:
            raise RuntimeError("official RF-DETR checkpoint identity changed")
        manifest = build_dataset(args.annotations, args.images, args.work / "dataset")
        result["dataset"] = manifest
        import torch
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        gpu_capability = list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None
        result["gpu"] = {"name": gpu_name, "capability": gpu_capability}
        if not torch.cuda.is_available() or "Tesla T4" not in str(gpu_name):
            raise RuntimeError(f"V37 requires an Azure Tesla T4, got {gpu_name!r}")
        if tuple(gpu_capability or ()) != (7, 5):
            raise RuntimeError(f"V37 requires compute capability 7.5, got {gpu_capability!r}")
        from rfdetr import RFDETRSegNano

        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED); torch.cuda.reset_peak_memory_stats()
        config_warning_messages = []
        load_log_messages = []
        class CaptureHandler(__import__("logging").Handler):
            def emit(self, record):
                load_log_messages.append(self.format(record))
        capture_handler = CaptureHandler()
        rfdetr_logger = __import__("logging").getLogger("rfdetr")
        rfdetr_logger.addHandler(capture_handler)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = RFDETRSegNano(
                    num_classes=1, resolution=RESOLUTION,
                    pretrain_weights=str(args.checkpoint),
                    mask_downsample_ratio=MASK_DOWNSAMPLE_RATIO,
                    gradient_checkpointing=True, device="cuda",
                )
                config_warning_messages.extend(str(item.message) for item in caught)
        finally:
            rfdetr_logger.removeHandler(capture_handler)
        allowed_config_warnings = [message for message in config_warning_messages if (
            "mask_downsample_ratio" in message and "significant portions" in message
        )]
        unexpected_config_warnings = [message for message in config_warning_messages if message not in allowed_config_warnings]
        partial_load_logs = [message for message in load_log_messages if "loaded only partially" in message.lower()]
        result["model_warnings"] = config_warning_messages
        result["allowed_config_warnings"] = allowed_config_warnings
        result["unexpected_config_warnings"] = unexpected_config_warnings
        result["load_log_messages"] = load_log_messages
        result["significant_load_warnings"] = partial_load_logs
        if unexpected_config_warnings or partial_load_logs:
            raise RuntimeError("pretrained checkpoint did not load under the frozen compatibility contract")
        train_output = args.work / "training"
        with monitor:
            model.train(
                dataset_dir=str(args.work / "dataset"), output_dir=str(train_output),
                epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, eval_batch_size=1,
                grad_accum_steps=1, lr=1e-4, num_workers=0, seed=SEED,
                early_stopping=False, eval_interval=1, use_ema=False,
                run_test=False, tensorboard=False, wandb=False,
                multi_scale=False, scale_jitter=False, aug_config={},
                augmentation_backend="torchvision", checkpoint_interval=1,
                device="cuda", save_dataset_grids=False,
                progress_bar="tqdm", compute_val_loss=True,
                notes={"experiment": "v37_azure_g1", "selection_id": SELECTION_ID},
            )
        result["losses"] = finite_losses(train_output)
        checkpoint_candidates = [
            train_output / "checkpoint_best_total.pth", train_output / "checkpoint_best_regular.pth",
            train_output / "last.ckpt", train_output / "checkpoint_0.ckpt",
        ]
        checkpoint = next((path for path in checkpoint_candidates if path.is_file()), None)
        if checkpoint is None:
            recursive = sorted(train_output.rglob("*.pth")) + sorted(train_output.rglob("*.ckpt"))
            checkpoint = recursive[0] if recursive else None
        if checkpoint is None:
            raise RuntimeError("RF-DETR produced no reloadable checkpoint")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            reloaded = RFDETRSegNano(
                num_classes=1, resolution=RESOLUTION, pretrain_weights=str(checkpoint),
                mask_downsample_ratio=MASK_DOWNSAMPLE_RATIO,
                gradient_checkpointing=True, device="cuda",
            )
            reload_warnings = [str(item.message) for item in caught]
        first_alias = args.work / "dataset/train" / (
            f"01_{manifest['records'][0]['image_id']}.jpeg"
        )
        if not first_alias.is_file():
            raise FileNotFoundError(f"converted reload alias missing: {first_alias}")
        prediction = reloaded.predict(str(first_alias), threshold=0.05, include_source_image=False)
        result["reload"] = {
            "checkpoint": str(checkpoint), "checkpoint_sha256": digest(checkpoint),
            "warnings": reload_warnings, "prediction_count": len(prediction),
        }
        result["status"] = "complete"
    except BaseException as error:
        result["status"] = "error"
        result["error_type"] = type(error).__name__
        result["error"] = str(error)
        result["traceback"] = traceback.format_exc()
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        result["peak_external_mib"] = monitor.peak_mib
        result["memory_samples"] = monitor.samples
        try:
            import torch
            result["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0
        except Exception:
            result["peak_allocated_mib"] = None
        losses_ok = result.get("losses", {}).get("all_finite", False)
        accelerator_ok = result.get("gpu", {}).get("name") is not None and (
            "Tesla T4" in result["gpu"]["name"]
            and result["gpu"].get("capability") == [7, 5]
        )
        result["gates"] = {
            "accelerator_t4": accelerator_ok,
            "status_complete": result["status"] == "complete",
            "finite_losses": losses_ok,
            "memory": 0 < result["peak_external_mib"] < MEMORY_GATE_MIB,
            "time": result["elapsed_seconds"] <= TIME_GATE_SECONDS,
            "checkpoint_reload": "reload" in result,
            "clean_pretrain_load": not result.get("significant_load_warnings", ["not checked"]),
            "data_firewall": not any((result["reads_tune"], result["reads_audit"], result["reads_test"],
                                      result["exports"], result["submits"])),
        }
        result["infrastructure_blocked"] = not accelerator_ok
        result["g1_evaluated"] = accelerator_ok
        result["g1_passed"] = accelerator_ok and all(result["gates"].values())
        result["g2_authorized"] = result["g1_passed"]
        if result["g1_passed"]:
            result["action"] = "implement frozen V37 G2"
        elif result["infrastructure_blocked"]:
            result["action"] = "repair Azure environment without changing V37 model/data/gates"
        else:
            result["action"] = "close V37 before G2"
        write_json(args.output, result)
        print(json.dumps({key: result[key] for key in (
            "status", "elapsed_seconds", "peak_external_mib", "peak_allocated_mib",
            "gates", "g1_passed", "g2_authorized", "action",
        )}, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    if result["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
