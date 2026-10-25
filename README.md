# Solar Filament Segmentation Challenge 2026 — final solution

Instance segmentation of solar filaments in GONG H-alpha full-disk images
(Kaggle `filament-segmentation-2026`, MAGFiLO ground truth, per-annotator-record
Panoptic Quality). Author: Abhimanyu R B.

| item | value |
|---|---|
| final Kaggle submission | `submissions/submission_v84.csv` (sha256 `2925e2a2…`, 1112 rows, 180 images) |
| public leaderboard | **0.39** (selected as final) |
| 5-fold out-of-fold PQ, 707 training images | 0.459 |
| technical report (host ACM template) | `report/acm/main.pdf` (source `report/acm/main.tex`) |
| reproduction notebook | `notebooks/reproduce_v84.ipynb` |
| weights | `weights/` (SHA-256 pinned in `weights/SHA256SUMS`) |

## Pipeline

1. **Detector** — YOLO11s-seg with an extra stride-2 ("P2") prototype head
   (`weights/p2first_detector.pt`), trained on all annotator records, images at
   2048 px inferred at 1024 px. Four flip views; proposals are merged across
   views by mask IoU and scored by the fraction of views that agree
   (`cluster_views` in `scripts/v61_toolkit.py`).
2. **Refiner** — ResNet18-UNet on 256 px crops (`weights/soft_all_refiner_state.pt`),
   trained with soft targets (mean of annotator masks). Re-draws each proposal at
   full resolution, 4-flip TTA, threshold 0.50; falls back to the detector mask
   if the refined mask is empty (`scripts/train_v73_refiner.py`, `kaggle_inference/solar_base.py`).
3. **Selection** — keep proposals with view-agreement score ≥ 0.30 and area ≥ 200 px,
   then a LightGBM "demote" model (`weights/demote_all.txt`, 28 shape/score
   features in `scripts/v83_rule.py`) drops candidates whose predicted
   agreement is below half the Dinkelbach optimum λ = 0.4587.
4. **Export** — greedy non-overlap by score, COCO RLE (`scripts/export_v74.py`).

Everything the committee needs to regenerate the CSV is in this repository; no
training, test labels or external data are required. See the report for the
design rationale and the negative results that led here.

## Reproduce the submission

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt            # torch 2.4.1 / ultralytics 8.4.138 pins
# place the Kaggle competition data under data/raw/filament-segmentation-2026/
# (or set SOLAR_DATA_ROOT to the directory holding MAGFiLO_1.0_Kaggle_2026/)
jupyter nbconvert --to notebook --execute notebooks/reproduce_v84.ipynb --output executed.ipynb
```

The notebook (i) verifies the weight hashes, (ii) runs
`scripts/export_v74.py` over the 180 test images (≈ 7 min on one T4, CPU also
works with `SOLAR_DEVICE=cpu`), and (iii) asserts that the canonical row hash of
the regenerated CSV equals that of `submissions/submission_v84.csv`.
`SOLAR_SMOKE=1` runs a 5-image smoke test.

## Training (optional, not needed for reproduction)

```bash
python scripts/train_v61_fold.py --fold K --recipe p2first      # detector, 5 folds (+ --fold all)
python scripts/eval_v61_oof.py infer --fold K --recipe p2first   # OOF proposal cache
python scripts/train_v73_refiner.py --fold K                     # refiner on OOF proposals
python scripts/v84_bundle.py                                     # demote model + λ, OOF PQ
```

Folds are grouped by physical image (`fold_of_file` in `scripts/eval_v62_refiner.py`); every
out-of-fold number in the report comes from this split.

## Repository layout

| path | contents |
|---|---|
| `scripts/` | training, OOF evaluation, feature/rule code, export; one file per experiment version (`vNN_*.py`) |
| `kaggle_inference/` | inference-only modules used by the export (`solar_base.py` = refiner runtime) |
| `notebooks/` | reproduction notebook |
| `weights/` | detector, refiner, demote model + `SHA256SUMS` |
| `submissions/` | the exact CSV submitted to Kaggle + sha |
| `report/acm/` | 4-page technical report (host template) |
| `docs/` | pre-registrations, results and audits for every experiment V1–V100; `docs/experiments.md` is the ledger, `docs/README_experiment_log.md` the running status log |
| `tests/` | pytest suite for the scorer, folds, features, export (`pytest -q`) |

## Scoring note

Local PQ follows the host's rule exactly: per annotator record, IoU > 0.5
hits without one-to-one matching, counts pooled over images and records
(`scripts/v83_rule.py`, checked against the host's self-evaluation notebook
and against a public-leaderboard probe, `docs/v100_preregistration_2026-10-25.md`).

## Data rules followed

Only the competition training images and labels were used for fitting. The
public MAGFiLO 1.0 release (which overlaps the test set) and SWEFil were never
read. No external data enters the final model.
