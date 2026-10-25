# V96 pre-registration: annotator-style fine-tune of the p2first detector (2026-10-22)

**Question.** V84 scores 0.399 against train annotators 010101/010401 and 0.435 against annotation group 01, versus
0.459 pooled and 0.56 against the easiest annotator (`docs/v95_local_world_result_2026-10-21.md`, addendum 2). The
public board (0.39) matches the hard style. Fold-0 raw candidates exist at IoU > 0.3 for 93 % of hard-annotator
filaments but match at IoU > 0.5 for only 70 % (easy annotators: 99 % / 93 %); hard annotators do not split
filaments (preds overlapping ≥ 2 GT: 1–2 %, same as easy). So the hard-style loss is extent / confidence, which a
detector trained on that style could learn. Does fine-tuning the fold-0 p2first detector on hard-style records raise
refined PQ on held-out hard-style records?

**Arms** (`scripts/train_v61_fold.py`, `STYLE_ARMS`), all fold 0, warm-started from
`outputs/v61_persistent/p2first_fold0/final.pt`, continuation stage only (20 epochs, V21 `continuation_args`,
`mask_ratio=1`), training records restricted to:
- `p2first_hard`: annotators 010101 + 010401 (119 of 935 fold-0 training records);
- `p2first_g01`: annotation group 01 (428 records);
- `p2first_more`: control, all 935 records (prices the extra 20 epochs themselves).

**Judge.** `scripts/v96_refined_fold0.py` (V81/V94 harness: views → flip consensus → V73 base fold-0 refiner →
cut 0.30 → per-record PQ) on the 141 fold-0 held-out files, per subset: `hard` (26 records), `g01` (88), `rest`
(131), `all` (219). Delta vs p2first with per-file bootstrap (2000 draws, seed 96). Primary subset: `hard` for
p2first_hard, `g01` for p2first_g01, `all` for p2first_more. Gate per arm: CONTINUE if Δ ≥ +0.020 and
P(Δ > 0) ≥ 0.90; KILL if Δ < +0.005; else INCONCLUSIVE. Secondary (reported, not gated): the `rest` and `all`
subsets, i.e. the price paid on other styles. The repro check p2first == V73 base fold 0 must pass.

**Pre-stated reading.** A CONTINUE on a style arm with a `rest` loss means the detector can be steered to a
style; whether the test style is that style is only readable on the public board (not submitted now). A KILL on
both style arms closes "train on hard-annotator style" as a lever. p2first_more CONTINUE alone means the gain is
extra training, not style.

**Not touched.** Test images, test labels, MAGFiLO release, leaked CSV. No Kaggle submission.

---
Result (2026-10-22 20:56Z): KILL on all three arms. See `docs/v96_style_finetune_result_2026-10-22.md`.
