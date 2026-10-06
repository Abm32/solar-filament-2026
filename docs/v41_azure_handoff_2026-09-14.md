# V41 Azure execution handoff — 2026-09-14

## Active run

Frozen experiment: `docs/v41_design_2026-09-14.md`.

Azure resource:

- subscription: `e64e3da3-a8ef-452e-a709-7d7437e12be9`;
- resource group / VM: `ROGII-RG` / `rogii-gpu`;
- region / size: `eastus` / `Standard_NC64as_T4_v3`;
- SSH: `ssh rogii` (`azureuser@20.51.160.183`);
- selected accelerator: physical GPU 0, Tesla T4 16,384 MiB;
- shared VM: do **not** deallocate without separately confirming Rogii is idle.

V41 was launched detached under:

`/mnt/solar-v41-context-topology`

Launcher PID: read `/mnt/solar-v41-context-topology/v41.pid` (initially `244674`).
The launcher was verified with PPID 1. Its Python child is the frozen runner.

At the latest startup check:

- atomic status: `running`;
- exact runtime import passed: Torch 2.4.1+cu121, NumPy 2.2.6, OpenCV 5.0.0,
  SciPy 1.15.3, isolated scikit-image 0.25.2;
- only physical GPU 0 was visible to Torch;
- step 0 completed at PQ 0 / 0 TP / 42 FP / 308 FN, selected threshold 0.65,
  predicted foreground fraction 0.0001362, and 1,710.09 MiB allocated peak;
- finite optimizer logs were observed through at least step 100;
- GPU 0 was actively training at 7,019 MiB / 100% utilization;
- GPUs 1-3 remained at 0 MiB;
- no result JSON exists yet because the bounded run is still active.

The expensive step-0 decode is complete. Do not kill or restart the healthy
training process. The next predeclared evaluation is step 250.

## Identities

- runner SHA-256:
  `39a1a7273414b338ad9e630a677d79fea9840d2e9c0efd1c6bb759d4ab2e9aff`;
- annotation SHA-256:
  `5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1`;
- selection ID:
  `741d8124fd834c465743cb4ddbee265298354b4e59b5ab9650379d162673ca2c`;
- deployment manifest:
  `outputs/v41_azure_deployment/deployment_manifest.json`;
- isolated wheel hashes are recorded in
  `/mnt/solar-v41-context-topology/wheel_sha256.txt` and the deployment manifest.

The shared repository virtual environment was not modified. The missing
scikit-image runtime and three missing dependencies were installed only under:

`/mnt/solar-v41-context-topology/site-packages`

## Monitor

```bash
ssh rogii '
  B=/mnt/solar-v41-context-topology
  echo ===status===
  cat "$B/operation_status.json"
  echo ===process===
  P=$(cat "$B/v41.pid")
  ps -o user,pid,ppid,stat,etime,time,%cpu,%mem,rss,args -p "$P"
  pgrep -P "$P" -af . || true
  echo ===gpu===
  nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu,temperature.gpu \
    --format=csv,noheader
  echo ===log===
  tail -200 "$B/logs/v41_launcher.log"
  echo ===results===
  find "$B/results" -maxdepth 1 -type f -printf "%f %s bytes\n"
'
```

A healthy active run has launcher and child processes, status `running`, and
increasing elapsed/CPU time or GPU activity. During decoding GPU utilization may
be zero while CPU remains active.

## Completion and retrieval

Expected files:

- result JSON: `/mnt/solar-v41-context-topology/results/v41_g2_result.json`;
- checkpoint: `/mnt/solar-v41-context-topology/results/v41_g2_result.pt`;
- complete log: `/mnt/solar-v41-context-topology/logs/v41_launcher.log`;
- atomic terminal status: `/mnt/solar-v41-context-topology/operation_status.json`.

Retrieve without modifying remote artifacts:

```bash
mkdir -p outputs/v41_azure_result
rsync -avz -e 'ssh -i ~/.ssh/rogii_vm_key' \
  azureuser@20.51.160.183:/mnt/solar-v41-context-topology/results/ \
  outputs/v41_azure_result/results/
rsync -avz -e 'ssh -i ~/.ssh/rogii_vm_key' \
  azureuser@20.51.160.183:/mnt/solar-v41-context-topology/logs/ \
  outputs/v41_azure_result/logs/
scp rogii:/mnt/solar-v41-context-topology/operation_status.json \
  outputs/v41_azure_result/
```

Validate before interpreting:

1. terminal status must be `complete` with return code 0;
2. result must parse as strict JSON and report experiment
   `v41_context_topology_g2`;
3. selection ID and 16-record/308-GT cohort must match;
4. `reads_tune`, `reads_audit`, `reads_test`, `exports`, and `submits` must all be
   false;
5. independently inspect every gate, not only `g2_passed`;
6. checkpoint SHA must match the result;
7. no full-population design is authorized unless `g2_passed=true`.

## Failure handling

If status becomes `failed`, preserve the log, status, and any partial artifacts
before diagnosis. Do not relaunch with changed cohort, channels, architecture,
loss, thresholds, decoder, steps, or gates. A narrowly scoped runtime fix requires
an immutable new run and is allowed only if no quality result was produced.

Do not deallocate the VM as part of V41 cleanup; it is shared with Rogii.
