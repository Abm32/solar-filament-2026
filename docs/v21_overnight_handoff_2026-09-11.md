# V21 overnight handoff — full-data V17 recipe refit

Saved **2026-09-11 22:49 IST**. Repository:
`/home/abhimanyu/Desktop/digital_agency/kaggle/solar/solar-filament-2026`.
Azure VM: `rogii-gpu`, resource group `ROGII-RG`, IP `20.51.160.183`.

## Completion update — 2026-09-11 23:22 IST

This run completed successfully; the historical running/recovery sections below
are retained for provenance. Atomic status is `complete`, all detector/refiner and
persistent-copy hashes passed, and no GPU process remains.

Final proposal cohort: **707/707 files, 7,709 matched samples from 8,199
annotations**. Final detector/refiner SHA-256 values:

- `5cc803e4f3ca334c471fc1012ef1e27a142b59b085831fc09d5e35d621ca5629`
- `26013cec76b7c329f9891fdbf90def19abc60a696d134273837cf5dc3f5f0792`

Test inference was subsequently completed on all four T4s as four deterministic
45-image shards. Each shard finished in 72–77 seconds. The merged V17-recipe CSV
contains **1,091 masks across 174 populated images** and passed both the repository
validator and an independent full RLE/area/ownership decoder.

Submission package:
`../submission_ready/v21_full_refit_2026-09-11/`. CSV SHA-256:
`98f58d2e115ebb6ae9c4f270aa4f03e46d75c279ad6a5c5c09afefb9121e9016`.
V21 was submitted exactly once as Kaggle submission **56170647** at
2026-09-11 18:06:29 UTC. It completed with public score **0.38**, below V17's
0.39. V17 submission `56103401` remains best; do not re-upload V21.

## Current state

V21 is the previously planned final full-data refit of the frozen V17 recipe.
It uses only approved competition labels and fits all **707 physical files / 1,154
annotator records / 8,199 annotations**. Because all labels are now training data,
there is no local validation claim for this final model.

The detector chain is complete:

| stage | duration | checkpoint SHA-256 |
| --- | ---: | --- |
| V6 base recipe, 80 epochs | 4,439.94 s | `9bc4fcadc2fdf2f70d7f1d6411827bdc3f734b9012fadb9f4ea330a8739708c4` |
| `mask_ratio=1` continuation, 20 epochs | 2,131.14 s | `5cc803e4f3ca334c471fc1012ef1e27a142b59b085831fc09d5e35d621ca5629` |

The refiner's completed stages are also preserved:

- GT-crop base refiner: 8/8 epochs complete.
- Canonical low-LR continuation: 3/3 epochs complete (the actual E014-selected duration).
- Proposal cohort generation from the new detector is running/resumable.
- Proposal-centered continuation: 4 epochs will run automatically after cohort completion.

## Unattended execution guarantee

The earlier SSH-attached refiner was replaced with a real detached session:

- wrapper: `/mnt/solar-v21-full-refit/run_v21_unattended.sh`
- wrapper PID file: `/mnt/solar-v21-full-refit/unattended.pid`
- atomic status: `/mnt/solar-v21-full-refit/operation_status.json`
- detailed log: `/mnt/solar-v21-full-refit/refiner_unattended.log`
- launcher log: `/mnt/solar-v21-full-refit/unattended_launcher.log`
- ephemeral working directory: `/mnt/solar-v21-full-refit/`
- persistent copy destination:
  `~/solar-filament-2026/outputs/v21_full_refit_persistent/`

The detached wrapper was verified from a second SSH connection with **PPID 1** and
its own session ID. Proposal caches advanced from 126 to 136 after detachment,
proving forward progress. Closing Kiro, the terminal, or the local computer will
not terminate it, provided the Azure VM itself remains allocated and running.

The wrapper writes one of these final states atomically:

- `running`: training/proposal generation still active;
- `complete`: refiner finished, hashes verified, compact artifacts copied to
  persistent OS-disk storage;
- `failed`: inspect `refiner_unattended.log`;
- `verification_failed`: training returned but artifact/hash checks failed.

## Persistent safety copy already made

The expensive completed artifacts were copied off ephemeral `/mnt` immediately,
before the remaining job finished:

`~/solar-filament-2026/outputs/v21_full_refit_persistent/`

It currently contains the detector base/final checkpoints, refiner base/canonical
checkpoints, stage manifests, source scripts, and `checkpoints/SHA256SUMS`.
The detached wrapper will add the proposal-centered checkpoint, cohort manifest,
full result, logs, verification report, and final `operation_status.json` after
successful completion.

## Check tomorrow

```bash
ssh -i ~/.ssh/rogii_vm_key azureuser@20.51.160.183 '
  BASE=/mnt/solar-v21-full-refit
  cat "$BASE/operation_status.json"
  echo "--- progress ---"
  grep -E "proposal cohort files=|proposal train epoch=|refiner stage=" \
    "$BASE/refiner_unattended.log" | tail -20
  echo "--- processes ---"
  pgrep -af "run_v21_unattended|train_v21_full_refiner" || true
  echo "--- GPU ---"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
'
```

If status is `complete`, verify the persistent copy:

```bash
ssh -i ~/.ssh/rogii_vm_key azureuser@20.51.160.183 '
  P=~/solar-filament-2026/outputs/v21_full_refit_persistent
  cat "$P/operation_status.json"
  cat "$P/verification.json"
  (cd "$P/checkpoints" && sha256sum -c SHA256SUMS)
'
```

Then pull the persistent directory locally:

```bash
rsync -avz -e "ssh -i ~/.ssh/rogii_vm_key" \
  azureuser@20.51.160.183:~/solar-filament-2026/outputs/v21_full_refit_persistent/ \
  outputs/v21_full_refit_persistent/
```

## Remaining work tomorrow

Training completion is automatic. The wrapper deliberately does **not** perform
Kaggle submission or silently generate a leaderboard probe. After a verified
`complete` state:

1. Build test inference using the frozen V17 recipe:
   - full-data detector final checkpoint;
   - full-data proposal-centered refiner final checkpoint;
   - box `iou=0.3`, retina masks, agnostic NMS;
   - four detector views / flip consensus;
   - mask threshold 0.45, confidence 0.30;
   - constrained refinement, minimum area 20, disjoint pixel ownership.
2. Process all 180 test images and generate a CSV plus provenance manifest.
3. Independently validate IDs, all RLEs, minimum areas, per-image non-overlap,
   coverage and SHA-256.
4. Compare row/coverage/mask statistics with V17 and inspect representative images.
5. Submit only after explicit user authorization and a live duplicate/quota check.
6. Record submission ID and poll its public score.

Do not claim a local PQ improvement for V21: all 707 labeled files were used for
fitting. This is a final-fit public-directed experiment based on V17's previously
validated recipe.

## Failure recovery

If status is `failed`, inspect:

```bash
ssh -i ~/.ssh/rogii_vm_key azureuser@20.51.160.183 \
  'tail -100 /mnt/solar-v21-full-refit/refiner_unattended.log'
```

Do not delete `/mnt/solar-v21-full-refit/refiner/proposal_cache`; the run is
resumable and every existing cache is reused. Completed stage manifests make the
8-epoch base and 3-epoch canonical stages skip automatically. To relaunch only
if no real wrapper/refiner process exists:

```bash
ssh -i ~/.ssh/rogii_vm_key azureuser@20.51.160.183 '
  BASE=/mnt/solar-v21-full-refit
  pgrep -af "run_v21_unattended|train_v21_full_refiner" || true
  setsid nohup bash "$BASE/run_v21_unattended.sh" </dev/null \
    > "$BASE/unattended_launcher.log" 2>&1 &
  echo $! > "$BASE/unattended.pid"
'
```

The VM is shared with Rogii. Do not deallocate it while status is `running`.
After status is `complete` and the persistent/local copies are verified, confirm
Rogii is idle before deallocation.
