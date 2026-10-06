# Research beyond the 0.39 plateau — 2026-09-11

## Completed V20 result

The approved V20 run is complete. The competition-only CondInst experiment used
all 937 training annotator records and completed 20 epochs / 4,700 global steps
on four Azure T4s in 34.2 minutes. Tune-only selection over fixed epochs
5/10/15/20 and confidence 0.05-0.50 selected epoch 10 / confidence 0.50:

- tune PQ **0.358191** (557 TP / 300 FP / 577 FN);
- audit PQ **0.397240** (244 TP / 92 FP / 215 FN);
- required V17 bars **0.448703 / 0.470763**;
- `export_authorized=false`.

CondInst improved over V19 E031 by +0.023064 tune / +0.038092 audit, confirming
that its anchor-free dynamic-mask family is a better fit than the tested
Mask2Former setup. It remains 0.090512 / 0.073523 below V17 and did not justify a
gated export. The user later explicitly authorized a one-off measurement override;
`submission_v20.csv` scored **0.30** public. Do not re-upload it. Full artifacts
and independent verification are under `outputs/v20_condinst_azure/`; see E032 in
the experiment ledger. The follow-up policy is unchanged: do not add epochs or
expand thresholds without a new mechanism-specific hypothesis.

## Result first

There is still no public, independently reproducible 0.55 recipe. The current
leaderboard has three teams at 0.56 and a large 0.55 cluster, but the newest
rigorous public entry (time-embargoed three-fold U-Net, local PQ 0.35236) scores
only 0.31 public. The opaque payload and private-checkpoint caveats from the
6 September audit therefore remain unresolved.

The strongest justified new experiment was **CondInst R50-FPN with a stride-4 P2
mask branch and auxiliary semantic loss**, trained only on approved competition
labels. E032 has now measured it: 0.358191 tune / 0.397240 audit, an improvement
over V19 but below both V17 bars.

## Fresh evidence

### Current competition state

Authenticated leaderboard check on 11 September shows three 0.56 teams and many
0.55 teams. Two 0.56 submissions are new since the 6 September snapshot, but no
new top method is public. Sixty-six competition kernels are now indexed. The
strongest newly reproducible source is `srivatsav-kannan/solar-seg`: its batch-Dice
continuation improves prospectively frozen, time-blocked CV from 0.33531 to
0.35236 (all three folds positive, paired interval [0.01181, 0.02230]), yet its
all-data model scores only **0.31 public**. This independently confirms our own
finding that reasonable local PQ gains do not explain the public 0.55 cluster.

### Solar-specific model research

MORDEN (2026) reports semantic IoU 0.812 / F1 0.895 on AHAS and ablations supporting
5x5 kernels, InstanceNorm, hierarchical multiscale pooling, focal loss, robust
per-image normalization and DenseCRF edge refinement. These mechanisms fit the
competition's multiscale, low-contrast, long-tail foreground. However, semantic
IoU is not instance PQ, AHAS differs from competition data, and external solar
labels/weights require rule verification. The full public MAGFiLO archive must not
be used because it contains competition test labels.

Guo et al. (Solar Physics, 2022) directly applied **CondInst** to 10,000 solar
filaments and reports precision 90.83%, recall 83.88%, AP 82.86%, F1 87.22%, with
specific success on fragmented filaments and uneven/low-contrast images. CondInst
uses anchor-free FCOS locations, dynamic per-instance full-map mask heads,
relative coordinates, focal detection loss and optional semantic supervision.
The primary paper shows that a P2 stride-4 mask branch improves masks and that
upsampling especially helps small objects. This directly targets our 18.7%
all-detector miss floor, small/category-3 FN concentration, and coarse-mask wall.

### Temporal/cross-site structure: measured and closed as direct propagation

MAGFiLO's official paper documents six synchronized GONG stations with adjacent
acquisitions offset by +/-30 seconds. In the competition split, one tune and two
audit files have frozen-training donors 20 seconds away; many more are within
hours. Registered image pairs visibly share solar structure.

A strict CPU benchmark used only frozen-training labels as donors, affine ECC
registration, tune-only time-window selection and audit-once evaluation. The
selected <=1 hour policy scored only **0.009549 tune PQ / 0.007470 audit PQ**.
Only 1 tune and 2 audit physical images were eligible, and annotator instance
identity does not transfer reliably. Direct temporal label propagation is closed;
it must not be used to patch test predictions.

Artifact: `outputs/temporal_transfer_2026-09-11.json`.

## Ranked hypotheses

1. **CondInst R50-FPN, P2 masks, semantic auxiliary branch — measured in E032.**
   It improved over V19 but failed both V17 bars, so this first architecture run
   is closed without export.
2. **MORDEN-style semantic teacher/branch — follow only if CondInst shows recall
   but weak mask quality.** Add 5x5/InstanceNorm/multiscale/focal ideas inside the
   auxiliary branch or use a separately trained competition-only foreground model.
   Do not import AHAS labels or weights without organizer permission.
3. **DenseCRF at native 2048 — conditional follow-up.** It can improve boundaries
   but cannot recover missed instances; run only if CondInst clears recall and its
   matched IoU remains limiting.
4. **Full-data refit / batch Dice semantic U-Net — do not copy as primary.** Clean
   evidence but 0.31 public; useful only as a possible diverse semantic teacher.
5. **Temporal transfer — rejected by measured audit result.**
6. **More YOLO/Mask2Former epochs, thresholds, or union ensembles — remain closed.**

## Prepared V20 contract

`scripts/prepare_v20_condinst.py` creates integer-ID, one-class COCO files while
retaining every annotator record and keeping physical filenames split-isolated.
Generated populations are exact:

| group | files | annotator records | annotations | max instances |
| --- | ---: | ---: | ---: | ---: |
| train | 566 | 937 | 6,606 | 26 |
| tune | 94 | 150 | 1,134 | 23 |
| audit | 47 | 67 | 459 | 17 |

Artifacts: `outputs/v20_condinst_prep/`. Source annotation SHA-256 is
`5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1`.
The contract explicitly excludes the full public MAGFiLO archive, AHAS/MHAS
labels or solar-specific weights without permission, and temporal test patching.

Executed first-run configuration:

- CondInst R50-FPN; one filament class; standard public COCO CondInst or ImageNet
  initialization only.
- 1024 input, robust per-image disk normalization, P2 stride-4 mask branch,
  relative coordinates, focal FCOS loss, Dice dynamic-mask loss, auxiliary
  semantic loss.
- All 937 training annotator records; frozen 94/47 tune/audit protocol.
- Smoke first on the 26-instance record, batch 1; require at least 2 GiB memory
  headroom. Then a timing probe before a bounded full run.
- Confidence/NMS grid frozen before audit. Tune selects; audit is read once.
- No CSV unless tune >=0.448703 **and** audit >=0.470763. A real architectural
  promotion requires tune >=0.55. Public submission still requires explicit
  authorization after local gates and full CSV validation.

## Compute and decision

Data preparation is complete and CPU-validated. Actual CondInst/Detectron2
integration, environment build, smoke and full training are GPU work. A T4/Azure
estimate should be measured by the smoke/timing probe; based on prior V19 runtime,
a conservative ceiling is 6 hours on one T4-equivalent GPU (about $26 on the
currently allocated four-T4 VM, which bills for the whole VM). Stop immediately
on OOM, failed canonical population checks, or an extrapolated runtime beyond the
approved cap.

This is the strongest remaining research bet, not a promise of 0.55. Its value is
that failure would close a solar-validated anchor-free dynamic-mask family rather
than repeat an already falsified lever.
