# V64 pre-registration: copy-paste augmentation on p2first, fold 0 (2026-10-07, before any run)

## Why
Recall is the bottleneck (about 27% of GT never proposed, mostly small objects). Copy-paste
targets small/rare instances in the literature (Kisantal 2019; Ghiasi 2021) and was never tried
here: every detector so far used `copy_paste=0.0`. No new labelled data or reachable GONG
archive exists (see `v63_preregistration_2026-10-07.md`), so augmentation is the cheap way to
show the detector more small filaments.

## Arm
`p2first_cp` = `p2first` with Ultralytics `copy_paste=0.3` (default flip mode: a mirrored copy of
each filament, which stays on the centred solar disk) in both stages. Nothing else changes: same
init, seed 42, epochs, fold-0 dataset. Runner `scripts/run_v64_cp.sh` on rogii GPU 3.

## Gates (same rules and scorers as V62 fold 0; reference A = p2first, noise A' = p2first_seed2)
1. Structural (`analyze_v62_structural.py`): pooled small+thin oracle recall gain over p2first
   >= +0.030 AND >= 2x seed noise |A'-A|, AND proposals/image and FP/image at conf 0.35 each
   <= 1.30x p2first. Fail -> kill copy-paste.
2. PQ (`eval_v61_compare.py`, earned only by a structural pass): its built-in V61 verdict versus
   p2first with the p2first_seed2 noise scale.
A pass on both authorises folds 1-4 (all GPUs), then the V63-style refined test. It does not
authorise a submission. The copy-paste probability is not tuned on fold 0.
