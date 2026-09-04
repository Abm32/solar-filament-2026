# Solar Filament Segmentation Challenge 2026

Repository for the Kaggle Solar Filament Segmentation Challenge 2026.

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

GitHub remote is pending because the local `gh` token is currently invalid. Re-authenticate with:

```bash
gh auth login -h github.com
```

Then create and push:

```bash
gh repo create solar-filament-2026 --public --source=. --remote=origin --push
```

## Training Skeleton

```bash
python scripts/prepare_yolo_dataset.py --config configs/yolo_seg.yaml
python scripts/train_yolo_seg.py --config configs/yolo_seg.yaml
python -m src.train --config configs/baseline_unetplusplus.yaml
```

The YOLO segmentation path is the main track. The `src.train` semantic path is kept as a controlled baseline and expects prepared binary masks under `data/processed`.
