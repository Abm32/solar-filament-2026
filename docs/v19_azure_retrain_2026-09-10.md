# V19 retrain on Azure — where it's running, how to check it, how to stop it

> Status: **RUNNING** as of 2026-09-10 22:37 IST (launched 16:03 UTC / 21:33 IST).
> This doc is the live operational reference for the Azure retrain launched
> after fixing the two bugs in `docs/v19_regression_2026-09-10.md`.

## Where it's running

- **VM:** `rogii-gpu` (belongs to a different project — the Rogii wellbore
  competition — reused here because Azure's `Standard NCASv3_T4` family quota
  in this subscription is fully allocated to this VM's size already; creating
  a separate smaller T4 VM was rejected with `QuotaExceeded`, current
  limit=64, usage=64, and the VM's own allocation *is* that 64).
- **Resource group:** `ROGII-RG`. **Region:** `eastus`.
- **Size:** `Standard_NC64as_T4_v3` — 4x Tesla T4 (16 GiB each), 64 vCPUs, 432
  GiB RAM. Only **GPU 0** is used (`CUDA_VISIBLE_DEVICES=0`); the training code
  has no multi-GPU logic. **Billing is for the whole VM (~$4.35/hr on-demand)
  while it is running**, regardless of how many of its 4 GPUs are in use —
  this is materially more expensive per GPU-hour than a dedicated single-T4
  VM would have been, forced by the quota wall above.
- **Public IP:** `20.51.160.183`. **SSH:** `ssh -i ~/.ssh/rogii_vm_key azureuser@20.51.160.183`
- **Project directory on the VM:** `~/solar-filament-2026/` (separate from
  this VM's existing Rogii-project files under `~/rogii/`, per
  `docs/status_2026-09-04.md`; nothing there was touched).
- **Python env:** `~/solar-filament-2026/.venv` — `torch==2.4.1+cu121`,
  `transformers==4.44.2`, `pycocotools==2.0.8`, `scipy==1.15.3` (see
  "Bugs found on this VM" below), `opencv-python-headless==4.10.0.84`,
  `pandas==2.2.2`, `kaggle==1.6.17`.
- **Data:** `~/solar-filament-2026/data/raw/filament-segmentation-2026/` —
  downloaded via the Kaggle CLI directly on the VM (credentials copied to
  `~/.kaggle/kaggle.json`, `chmod 600`), unpacked with `scripts/unpack_data.py`.
- **Training process:** launched via
  ```
  cd ~/solar-filament-2026 && CUDA_VISIBLE_DEVICES=0 nohup .venv/bin/python \
    scripts/train_v19_mask2former.py --full --data-root data/raw \
    --out-dir outputs/v19_mask2former_azure_full > /tmp/v19_full_train.log 2>&1 &
  ```
  so it survives SSH disconnection. Log: `/tmp/v19_full_train.log` on the VM.
  Output directory: `~/solar-filament-2026/outputs/v19_mask2former_azure_full/`
  (`training_history.json` updated after every epoch, `best_model/` holds the
  best-tune-PQ checkpoint so far, `v19_full_result.json` written only at the
  very end).

## How to check status

From the local machine (no need to stay SSH'd in — the process is detached):

```bash
# Tail the live training log
ssh -i ~/.ssh/rogii_vm_key azureuser@20.51.160.183 "tail -30 /tmp/v19_full_train.log"

# Confirm the process is still alive and see GPU utilization
ssh -i ~/.ssh/rogii_vm_key azureuser@20.51.160.183 \
  "ps aux | grep train_v19 | grep -v grep; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv"

# Check per-epoch history written so far (valid mid-run, updated every epoch)
ssh -i ~/.ssh/rogii_vm_key azureuser@20.51.160.183 \
  "cat ~/solar-filament-2026/outputs/v19_mask2former_azure_full/training_history.json"

# Once finished, the final gate-check result:
ssh -i ~/.ssh/rogii_vm_key azureuser@20.51.160.183 \
  "cat ~/solar-filament-2026/outputs/v19_mask2former_azure_full/v19_full_result.json"
```

Each epoch has been taking ~600–620 s on this VM (vs ~800–900 s on Kaggle's
kernel T4 — this VM's T4 appears to run somewhat faster, or has less
contention), so 20 epochs is expected to take **roughly 3.3–3.5 hours** end to
end, plus a final audit-set evaluation pass.

## How to pull results back locally once done

```bash
rsync -avz -e "ssh -i ~/.ssh/rogii_vm_key" \
  azureuser@20.51.160.183:~/solar-filament-2026/outputs/v19_mask2former_azure_full/ \
  outputs/v19_mask2former_azure_full/
```

## How to stop everything and stop billing

```bash
# Kill the training process if it needs to be stopped early
ssh -i ~/.ssh/rogii_vm_key azureuser@20.51.160.183 "pkill -f train_v19_mask2former"

# Deallocate the VM (stops billing; the VM's disk/config persist, so it can
# be restarted later without redoing setup)
az vm deallocate -g ROGII-RG -n rogii-gpu
```

**This must be done when the run finishes and results are pulled — do not
leave `rogii-gpu` running.** At ~$4.35/hr, an idle day costs over $100.

## Bugs found and fixed while setting this VM up (in addition to E029's two)

Running the *raw* script directly on a VM (rather than through Kaggle's
kernel-staging bundler, `prepare_v19_mask2former.py`) surfaced two more
pre-existing defects that Kaggle's execution path had been silently masking:

1. **`sys.path` inserted `REPO_ROOT / "src"` instead of `REPO_ROOT`.** The
   script does `from src.coco import ...`, which requires `src`'s *parent*
   directory on the path, not `src` itself. This was invisible on Kaggle
   because the staged kernel bundle synthesizes `sys.modules['src.coco']`
   directly and never actually exercises this import statement as written.
   Fixed: `sys.path.insert(0, str(REPO_ROOT))`.
2. **`scipy` was missing from `ensure_environment`'s standalone-VM install
   list.** `Mask2FormerLoss`'s Hungarian-matching criterion imports
   `scipy.optimize.linear_sum_assignment`; Kaggle's base kernel image ships
   scipy already, so this gap was invisible there. Added `scipy==1.15.3`.

Both committed to git (see commit `0570003`). Neither affects the E029 result
or its diagnosis — they are packaging/environment gaps specific to running
outside Kaggle, not training-logic bugs.

## Early signal from this run (informal, not a final result)

As of epoch 6/20 (partial): tune PQ has been **0.20–0.23**, versus E029's
best-ever epoch value of 0.1536, and FP counts have dropped to 880–1440 versus
E029's 1386–3214. This is consistent with — though does not yet confirm — the
eval-resolution-mismatch fix being the dominant driver of E029's failure. The
run must finish and clear SUBMIT_BAR (0.448703) as measured by its own final
`v19_full_result.json`, not this informal mid-run read, before any conclusion
or submission decision is made.
