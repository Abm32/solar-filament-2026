# V19 retrain on Azure — completed result and teardown

## Controlled all-annotator retrain — final result (2026-09-11)

> **COMPLETE; EXPORT NOT AUTHORIZED.** Launched 2026-09-11 03:20:25 UTC /
> 08:50:25 IST after explicit approval and completed training plus calibration at
> 09:11:50 UTC / 14:41:50 IST. Artifacts were pulled and independently verified.
> No CSV was generated and Kaggle confirms no V19 upload.

The corrected population was train 937 annotator records / 6,606 annotations /
566 files, tune 150 / 1,134 / 94, and audit 67 / 459 / 47. All model and training
constants were unchanged. The run completed all 20 epochs; default threshold
0.50 selected epoch 15 at tune PQ **0.259899**, with audit PQ **0.279828**.
Canonical tune-only calibration selected threshold **0.90**:

| Result | Value |
| --- | ---: |
| tune PQ | **0.335127** (504 TP / 238 FP / 630 FN) |
| audit PQ | **0.359148** (213 TP / 78 FP / 246 FN) |
| required tune / audit | 0.448703 / 0.470763 |
| margins | -0.113576 / -0.111615 |
| export authorized | **false** |

Local output:
`outputs/v19_mask2former_azure_all_annotators/`. Checkpoint SHA-256:
`9d59f379cf07430a424650527cacbbfce07494337b566ec20ba0959a84e34501`.
Calibration SHA-256:
`27a094ad36edb47f40fe2adada1f9fb2738e26e8a749fdf652d3b7be8a975022`.
Independent verification confirmed exact canonical populations, all 20 epochs,
checkpoint identity, both bars, false gate, and no CSV. The exporter now also
requires the canonical calibration artifact and exact checkpoint hash, threshold,
and evaluated resolution; no force bypass remains. Focused tests: 42 passed.

**Decision:** the last controlled V19 hypothesis is closed. Keep V17/public 0.39
and do not export, submit, add epochs, or run a blind sweep without a new specific
hypothesis. The Azure VM was still running but had no GPU or project processes;
it is shared with Rogii, so deallocation was not performed implicitly.

## Prior collapsed-record retrain and calibration

The fixed V19 Mask2Former retrain completed all 20 epochs. Its initial final
artifact reported best tune PQ **0.266068** (epoch 7) and audit PQ **0.282324** at
1024px. This confirmed that fixing the E029 train/eval resolution mismatch and
adding LR warmup/cosine decay materially improved the first run's 0.153578, but it
still missed the tune submission bar of 0.448703 by a wide margin.

A subsequent audit found that V19's dataset conversion retained one arbitrary
COCO image record per filename. Because duplicated filenames represent distinct
annotators, this affected **both optimization and metric accounting**: the run
trained on 566 records instead of the canonical 937 annotator records, and its
evaluation had only 728 tune GT instances rather than the 1,134 used by the
canonical V9-V18 evaluator. `build_hf_dataset` now retains every annotator record,
and future export authorization requires both V17 bars: tune **0.448703** and
audit **0.470763**.

Before any further training, the saved checkpoint was evaluated once on the
canonical population with `scripts/calibrate_v19_threshold.py`. It predicts each
file once, scores that prediction against every annotator record, chooses among a
pre-bounded 0.50-0.95 confidence grid on tune only, and evaluates only the chosen
threshold on audit.

| Result | Value |
| --- | ---: |
| checkpoint SHA-256 | `fd5c3abe7da714bae66489932bd6334b502978aefb4ff7616ef5fa8c6928ade6` |
| tune population | 94 files / 150 annotator records / 1,134 GT |
| audit population | 47 files / 67 annotator records / 459 GT |
| default threshold 0.50 tune PQ | 0.264025 |
| selected threshold | **0.75** |
| selected tune PQ | **0.302220** |
| selected audit PQ | **0.305922** |
| required tune / audit | 0.448703 / 0.470763 |
| export authorized | **false** |

No submission CSV was generated and no Kaggle submission was made. The local
calibration artifact is
`outputs/v19_mask2former_azure_full/v19_threshold_calibration.json`, SHA-256
`1fc32020639c6e164fa66d947610a117a1ce2676c98aae1863c9d0b96cb45522`.
The export gate was smoke-tested against this real artifact and refused before
creating a CSV.

## Current recommendation

Keep V17's accepted public score **0.39** and do not export the current V19
checkpoint. Confidence calibration helped by only +0.038194 tune PQ and the
checkpoint remains about 0.147/0.165 below the tune/audit bars. However, because
that checkpoint was trained on 566 collapsed file records rather than all 937
annotator records, one controlled all-annotator retrain is a specific, measurable
remaining hypothesis. Run it only with explicit approval for the paid GPU cost,
and keep every other variable plus the canonical calibration protocol fixed;
more epochs or a blind hyperparameter sweep are not justified.

## Azure resource and retained environment

- VM: `rogii-gpu`, resource group `ROGII-RG`, region `eastus`.
- Size: `Standard_NC64as_T4_v3` (4x Tesla T4, billed for the whole VM at roughly
  $4.35/hour while allocated). V19 used only GPU 0.
- SSH: `ssh -i ~/.ssh/rogii_vm_key azureuser@20.51.160.183`.
- Project: `~/solar-filament-2026/`; the existing `~/rogii/` project was not
  modified.
- Environment: project `.venv` with torch 2.4.1+cu121, transformers 4.44.2,
  pycocotools 2.0.8, scipy 1.15.3, OpenCV headless 4.10.0.84, pandas 2.2.2,
  and Kaggle 1.6.17.
- Remote outputs: `~/solar-filament-2026/outputs/v19_mask2former_azure_full/`.
- Training log: `/tmp/v19_full_train.log`; calibration log:
  `/tmp/v19_calibration.log`.

Check state and deallocate when safe:

```bash
az vm get-instance-view -g ROGII-RG -n rogii-gpu \
  --query "instanceView.statuses[?starts_with(code, 'PowerState/')].displayStatus" -o tsv
az vm deallocate -g ROGII-RG -n rogii-gpu
```

Deallocation stops compute billing while preserving the disk and VM configuration.
Because the VM belongs to another project, this action was not taken implicitly.

## Environment bugs fixed during setup

Running the raw script directly on the VM exposed two packaging defects that
Kaggle's staging path had masked:

1. `sys.path` inserted `REPO_ROOT / "src"` instead of `REPO_ROOT`, which cannot
   satisfy `from src.coco import ...`. This is fixed.
2. `scipy` was absent from the standalone-VM install list even though
   Mask2Former's Hungarian matcher imports it. `scipy==1.15.3` is now pinned.

These were environment defects, not causes of E029's model result. The training
logic fixes remain documented in `v19_regression_2026-09-10.md`.
