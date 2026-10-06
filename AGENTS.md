# Repository Guidelines

Kaggle Solar Filament Segmentation Challenge 2026 (`filament-segmentation-2026`). Scored by
Panoptic Quality (PQ), so instance merge/split behavior matters as much as mask overlap.

## Project Structure & Module Organization

- `src/`: shared library. `rle.py` (pycocotools RLE, 2048x2048), `pq_eval.py` (local PQ metric),
  `coco.py` (MAGFiLO annotation loading, best-annotator selection), `infer_classical.py` (baseline).
- `scripts/`: one file per experiment step, versioned `vNN`. Prefixes: `prepare_*` stages a frozen
  Kaggle kernel (hash-checks annotations and source, never launches); `train_*` is the runner
  that executes inside that kernel or on an Azure VM; `analyze_*`/`evaluate_*`/`diagnose_*`
  produce JSON under `outputs/`; `export_*` builds submission CSVs.
- `kaggle_inference/`, `kaggle_kernel/`: modules bundled into Kaggle kernels (`kernel-metadata.json`).
- `submission_ready/<version>_<date>/`: frozen submission packages with READMEs.
- `docs/experiments.md`: experiment ledger with `E0NN` entries; per-version design/result notes in
  `docs/vNN_*_<date>.md`. Repo-level state lives in `README.md` and `../RESUME_TOMORROW.md`.
- `data/`, `outputs/`, weights (`*.pt`, `*.pth`) are gitignored.

## Build, Test, and Development Commands

```bash
conda env create -f environment.yml && conda activate solar-filament-2026
kaggle competitions download -c filament-segmentation-2026 -p data/raw
python scripts/unpack_data.py
pytest                                   # all tests (testpaths = tests, pythonpath = .)
pytest tests/test_pq_eval.py -k <name>   # single test
ruff check .
python -m src.infer_classical --input-dir data/raw/test --output outputs/submission_classical.csv
```

## Coding Style & Naming Conventions

Ruff: line length 100, target `py311`, rules `E,F,I,UP,B`. Most modules start with
`from __future__ import annotations`. New experiment code follows the `vNN` naming and gets a
matching `tests/test_vNN_*.py`.

## Experiment Protocol

- Data splits: canonical `fit` (training), `tune` (94 files), `audit` (47 files), plus Kaggle `test`.
  Each design doc declares which splits it may read; do not open audit/test, export, or submit
  unless that doc's gates authorize it.
- Write gates (PQ, recall, FP, memory, runtime thresholds) in the design doc before running;
  a failed gate closes the path. Record outcome, hashes, and evidence paths in a new ledger entry.
- Do not relaunch or deallocate shared Azure VMs without explicit instruction.
- Preserve the current public-best submission package; compare new candidates against it.

## Commit & Pull Request Guidelines

Imperative, sentence-case subject without a type prefix (e.g. `Fix V19 eval/train resolution
mismatch and add LR warmup+decay`). Prefix with the ledger ID when a commit records an experiment
(`E029: V19 Mask2Former full run — SUBMIT_BAR not cleared`). Commit at milestones per
`docs/repo_setup.md`. Remote: `origin` → `Abm32/solar-filament-2026`, branch `main`.
