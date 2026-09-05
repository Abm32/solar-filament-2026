# Solar Filament Segmentation Challenge 2026

Repository for the Kaggle Solar Filament Segmentation Challenge 2026.

**Resume here (saved 2026-09-05):** [V10 submission and continuation handoff](docs/handoff_2026-09-06.md).
The [ready-to-upload V10 package](submission_ready/v10_2026-09-06/README.md) is validated
and unsubmitted; expected next daily reset is **6 September, 05:30 AM IST**.
Check live submissions first in case the user already uploaded it manually.

Current status and the V7/V8 model review are tracked in `docs/status_2026-09-05.md`.

The latest completed optimization is [V10 quality ordering and conservative V8 rescue](docs/v10_execution_2026-09-05.md).
Earlier COCO target rasterization correction, inference sweeps, and crop postprocessing
are recorded in [docs/optimization_2026-09-05.md](docs/optimization_2026-09-05.md).

The working strategy is a staged, reproducible pipeline:

1. Build a strong classical/CV baseline to validate RLE encoding and submission format.
2. Train a lightweight instance segmentation model using MAGFiLO masks.
3. Improve panoptic quality with post-processing that reduces fragmentation and over-merging.
4. Keep every experiment documented in `docs/experiments.md`.

Current conclusion: prioritize a detector-guided segmentation pipeline, not a plain semantic U-Net. The challenge is scored with Panoptic Quality, so instance separation and merge/split behavior matter as much as mask overlap.

## Setup

```bash
conda env create -f environment.yml
conda activate solar-filament-2026
```

## Data

Accept the Kaggle competition rules in the browser first, then run:

```bash
kaggle competitions download -c filament-segmentation-2026 -p data/raw
python scripts/unpack_data.py
```

Current local snapshot:

- `707` train images
- `180` test images
- `8,199` annotation entries
- `296` train files with duplicate annotator labels

## Baseline Submission

```bash
python -m src.infer_classical \
  --input-dir data/raw/test \
  --output outputs/submission_classical.csv
```

The submission CSV contains:

- `filament_id`
- `segmentation_rle`

RLE counts are encoded with `pycocotools` for 2048 x 2048 masks.

## Repository Status

GitHub: [Abm32/solar-filament-2026](https://github.com/Abm32/solar-filament-2026),
branch `main`. Code, experiment documentation, tests, and the prepared V10
submission package are versioned. Raw data, model weights, bulk run outputs,
credentials, and the downloaded third-party payload notebook remain excluded.

The prepared CSV can be uploaded without rerunning inference. Reproducing the
experiments additionally requires the data, checkpoints, and caches described
in the [handoff](docs/handoff_2026-09-06.md).

## Training Skeleton

```bash
python scripts/prepare_yolo_dataset.py --config configs/yolo_seg.yaml
python scripts/train_yolo_seg.py --config configs/yolo_seg.yaml
python -m src.train --config configs/baseline_unetplusplus.yaml
```

The YOLO segmentation path is the main track. The `src.train` semantic path is kept as a controlled baseline and expects prepared binary masks under `data/processed`.
