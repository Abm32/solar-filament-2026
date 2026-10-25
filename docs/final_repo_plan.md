# Final reproducibility repo: plan (2026-10-10)

This is a plan only. Nothing has been created, pushed or published. The current repo
(`Abm32/solar-filament-2026`) stays private until the user decides otherwise.

## 1. What the host requires (verbatim sources on rogii: `~/v69_kernels/lb/pages.txt`)

- **Final Submission (Overview, added Aug 7).** Submit through the Google form:
  1. A **4-page technical report**, written in the host's Overleaf template and submitted as a single PDF.
  2. A **publicly accessible Git repo** (GitHub, GitLab or Bitbucket), with no access request needed. It
     must contain:
     - `requirements.txt` listing **all packages with versions**;
     - a **Jupyter notebook that illustrates the entire pipeline**.
     - "The committee should not require any additional file to reproduce the submitted predictions."
- **Open-Access Policy.** The repo must be **public immediately when the competition closes** and stay
  public at least until the winners are announced (Nov 30). Code modularity, reproducibility and
  documentation are scored (30% qualitative rubric: pipeline description, morphology, code quality).
- **Rules §2.8 / Prizes (winners).** Deliver training code, inference code, and a description of the
  compute environment. The code must regenerate the winning submission. It must be released under an
  OSI-approved licence that does not limit commercial use (third-party models with incompatible licences
  are exempt). Winners may also need to explain and reproduce every major component (the AI-assistance
  clause in §2.8.b).
- **Evaluation.** Inference may use only the supplied test H-alpha images. No other ground truth may be
  used for training.
- **Deadline.** 2026-11-15 (report, repo and form; the competition end time is 06:00 UTC).

## 2. Target layout, mapped to existing code

| repo path | content | existing source (copy, do not edit originals) |
|---|---|---|
| `README.md` | one-page overview, results table, how to run, data-use statement | new; numbers from `docs/final_report_draft.md` |
| `requirements.txt` | exact pins **of the env that produced the submission** | regenerate (see §3) |
| `LICENSE` | OSI licence (check the AGPL-3.0 Ultralytics interaction) | new |
| `notebooks/reproduce_submission.ipynb` | test images → CSV, end-to-end, no extra files | new; wraps `scripts/export_v62_shard.py` logic |
| `notebooks/train_pipeline.ipynb` (or a documented CLI) | folds → detector → OOF proposals → refiner | wraps the training scripts below |
| `solar/arch.py`, `solar/init.py`, `solar/patch.py` | P2/stride-2 graph, COCO head remap, proto-scale patch | `scripts/v62_arch.py`, `v62_init.py`, `v62_patch.py`, `configs/yolo11s-p2first-seg.yaml` |
| `solar/train_detector.py` | p2first recipe (80 ep mr2 + 20 ep mr1) | `scripts/train_v61_fold.py` + `train_v21_full_detector.py` (+ `v61_seg_loss.py` only if the chunked loss is needed) |
| `solar/folds.py` | 5-fold-by-file manifest | `scripts/make_v61_folds.py` (+ `outputs/v61_folds.json`) |
| `solar/oof.py` | 4-flip inference, consensus, ownership, PQ counts | `scripts/eval_v61_oof.py`, `scripts/v61_toolkit.py`, `kaggle_inference/detector_tta.py`, `optimize_inference.py` |
| `solar/refiner.py`, `solar/train_refiner.py` | ResNet18 U-Net crop refiner, r3 recipe | `kaggle_inference/v11_refiner.py`, `scripts/train_v63_refiner.py` (`--train-all`), `scripts/eval_v62_refiner.py`, `scripts/reproduce_v17_from_cache.py` |
| `solar/export.py` | refined_constrained, RLE, disjointness | `scripts/export_v62_shard.py`, `src/rle.py` |
| `solar/metric.py` | host-exact PQ, plus the Dice/IoU/1:n plots | `src/pq_eval.py`, `scripts/v69_metric_audit.py`, host self-evaluation notebook |
| `tools/validate_submission.py` | format and disjointness checker | `scripts/validate_submission.py` |
| `tests/` | the small set that guards the pipeline | `tests/test_rle.py`, `test_pq_eval.py`, `test_submission_validation.py`, `test_detector_tta.py`, `test_v11_refiner.py`, `test_v62_*`, `test_v61_*` |
| `weights/` | `p2first_all707.pt` (21.0 MB), `r3all_refiner_state.pt` (57.1 MB) + `SHA256SUMS` | rogii: `outputs/v61_persistent/p2first_fold99/mask_ratio1/final.pt`, `outputs/v65/export/r3all_refiner_state.pt` |
| `weights/` (V84 candidate) | `soft_all_refiner_state.pt` (57.1 MB, sha `ef9e3810…129bcfe1`), `demote_all.txt` (1.0 MB LightGBM, sha `4b8c2375…9f93a9b4`) + `demote_all.json` (λ, feature list) | rogii: `outputs/v84/export/` (see §8) |
| `docs/` | the report PDF/tex, the ablation table, pre-registration summaries | curated from `docs/` (not all ~70 docs) |

Leave out: experiment-only scripts V1-V73, SWEFil code and artifacts (V58/V59), neighbour-frame
fetchers, magnetogram fetchers (or keep them in a clearly marked `analysis/` folder with the
data-use note), the Dataverse test-GT file, and `kaggle.json`.

## 3. Environment pins (the trap)

The local `requirements.txt` / `environment.yml` pin **torch 2.8.0, ultralytics 8.3.190, Python 3.11**.
The submission was produced on rogii with **Python 3.10.12, torch 2.4.1+cu121, ultralytics 8.4.138,
numpy 2.2.6, opencv 5.0.0** (checked read-only on rogii, `.venv`). `v62_patch.py` monkeypatches
Ultralytics internals, so the version matters.

**Action:** generate `requirements.txt` from the rogii `.venv` (`pip freeze`, trimmed to the packages
actually used), pin ultralytics exactly, and record CUDA/driver/GPU (T4) in the README. Then test the
notebook in a fresh env (and on a Kaggle GPU) from these pins only.

## 4. Weights hosting ("no additional file")

Both deployed weight files are below GitHub's 100 MB per-file limit. Committing them directly in
`weights/` (not Git LFS, which has bandwidth quotas and may count as an "extra" fetch) is the
simplest reading of "no additional file". The notebook verifies their SHA256. The COCO
`yolo11s-seg.pt` and ImageNet ResNet18 are needed only for **training** and are downloaded by the
libraries. Document this, or vendor the files if the host objects. A mirror as a public Kaggle
Dataset/Model is optional. `/mnt/v61` is ephemeral, and `p2first_fold99` is not in `v61_persistent/SHA256SUMS`. I checked on
2026-10-10 that the persistent copy, the `/mnt` copy and the `detector_sha256` in
`outputs/v65/export/shard_0.json` are all `7ff39296…8174920`. Add this file to SHA256SUMS. The refiner
sha has not been checked yet.

## 5. Notebook contract (`reproduce_submission.ipynb`)

1. Input: a `test_images/` path, defaulting to the Kaggle competition path. Output: `submission.csv`.
2. Steps: load weights (hash-checked) → 4-flip detector inference at 1024 → flip_consensus → crop refiner
   (4-flip TTA) → refined_constrained at t0.45/c0.30 → pixel ownership → [min-area rule if adopted] →
   RLE → validator.
3. **Acceptance test:** the regenerated CSV must equal `submission_v65_p2first_r3all.csv`. Check
   row-for-row RLE equality, or per-mask IoU ≥ 0.999 if GPU nondeterminism bites. The tolerance must be
   stated.
4. A second section reproduces the OOF metric on one fold from cached or recomputed predictions and draws
   the host's three plots (Dice, IoU, 1:n/n:1).
5. Record the runtime per image (needed for the "end-to-end efficiency" criterion).

## 6. Checklist

- [ ] Decide the final two Kaggle selections (V65 p2first+r3 is one; 1280 variant pending V68 5-fold +
      refined test; V73 refiner pending).
- [ ] For **each** selected submission: weights, config and notebook path that regenerate it.
- [ ] Hash-verify and persist the deployed detector and refiner (see §4).
- [ ] `requirements.txt` regenerated from the producing env, then a clean-env install test.
- [ ] Notebook regenerates the selected CSV(s) bit-for-bit or within the stated tolerance; runtime logged.
- [ ] Training reproducibility: one documented path from raw data to both weight files. **Gap:** the r3
      refiner starts from `best_refiner_coco.pt` (V6 GT-crop training in `kaggle_kernel/train_and_submit.py`
      → COCO-mask fine-tune). The fine-tune script is not identified **[TBD]**. If V73's leak-free
      lineage (ImageNet → stage A → stage B, one script) matches r3, prefer it for the repo.
- [ ] Data-use statement (README + report): test GT set aside, no SWEFil in the pipeline, magnetograms and
      neighbour frames diagnostic-only on train files, single-image inference.
- [ ] No secrets (`kaggle.json`, VM hostnames) and no test-GT file; `data/` gitignored with a download script.
- [ ] Licence chosen; Ultralytics AGPL-3.0 implication checked.
- [ ] Tests pass (`pytest`), `ruff check .` clean.
- [ ] Report PDF in the Overleaf template, ≤ 4 pages, includes the Dice/IoU/1:n plots.
- [ ] Repo made public at competition close (not before, per user); Google form submitted before
      2026-11-15 06:00 UTC.

## 7. Open questions

1. **Thread 747960 (open, no host answer as of 2026-10-10):** must the repo reproduce **both** selected
   Kaggle submissions, or only the ranking one or the one in the report? Safe default: make both
   reproducible (`v69_kaggle_metric_audit` §1.1).
2. Does "no additional file" allow the notebook to download weights or pretrained checkpoints from a
   public URL, or must everything be in the repo? (The plan above avoids the question for the deployed
   weights.)
3. Overleaf template sections and figure/reference limits: not saved locally. Open the template.
4. Thread 748045 (ours, open): training-only label transfer to nearby GONG frames. It is moot for the
   current pipeline (the line failed), but the answer affects the data-use wording.
5. Does the host need the full training lineage (V6 → COCO → r3), or is the deployed lineage plus a
   documented retrain acceptable?
6. Licence: whether an AGPL-3.0 dependency (Ultralytics) conflicts with the OSI/commercial-use winner
   licence, and how to state it.
7. Is "computational efficiency" measured on the host's hardware or our reported numbers? What should
   we report (s/image, GPU type)?

## 8. V84 polish-bundle export: reproducibility notes (added 2026-10-11)

This is a candidate second selection. Doc: `docs/v84_export_2026-10-11.md`. CSV: `outputs/v84/export/submission_v84.csv`
(sha `2925e2a2…95c4871f`). It was validated and has NOT been submitted. It differs from V65 in four components. The detector,
views and consensus are unchanged.
1. **Refiner:** the all-data **soft-target** refiner, V73 two-stage leak-free recipe. ImageNet ResNet18 →
   stage A (8 epochs, 3e-4, GT crops) → stage B (12 epochs, 1e-4, p2first OOF proposals). The stage-B target is the pixel mean of
   the records' matched GTs. Seed 42, all 707 files:
   `train_v82_refiner_targets.py fold --arm soft --fold 0 --train-all` (uses `train_v73_refiner.py --train-all`).
   This lineage needs no `best_refiner_coco.pt`, which closes the §6 training-lineage gap for this selection.
2. **Mask threshold 0.50** (V65: 0.45).
3. **MIN200:** drop final masks under 200 px.
4. **Demote rule:** the V83 LightGBM (LightGBM 4.5.0 from `outputs/v70/pylib`, which is **not** in the rogii venv; it must be
   added to `requirements.txt`).
   - It uses 28 per-candidate features, computed by `scripts/v83_extract.py` `refine_feats` + `candidate_rows`.
   - It drops a kept mask when ŷ ≤ λ/2, with λ = 0.45872.
   - The model is fitted by `scripts/v84_demote_fit.py` on the 707-file soft/t0.50 OOF candidates (`scripts/v84_extract.py soft`).
     This means reproducing it requires all five V84 soft fold refiners plus the 707-file OOF extract.
- **Export:** `scripts/_v74_export.sh` with `MASK_T=0.50 DEMOTE=… REFINER=… IMGSZ=1024`. It calls
  `export_v74.py --demote-model`. Its defaults are unchanged and byte-identical to V74/V65 on the 5-image parity test.
- **Notebook implication:** `reproduce_submission.ipynb` needs a second path with the soft refiner, t0.50, MIN200 and the
  demote step, plus LightGBM in the environment. Acceptance: regenerate `submission_v84.csv` (1112 rows).
