#!/usr/bin/env bash
# Detached Azure continuation wrapper for the V21 full-data refit.
set -uo pipefail

BASE=${V21_WORK:-/mnt/solar-v21-full-refit}
REPO=${V21_REPO:-$HOME/solar-filament-2026}
PERSIST=$REPO/outputs/v21_full_refit_persistent
STATUS=$BASE/operation_status.json
PY=$REPO/.venv/bin/python
STARTED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)

write_status() {
    local state=$1
    local detail=$2
    local finished=${3:-}
    local temporary="${STATUS}.tmp"
    "$PY" - "$temporary" "$state" "$detail" "$STARTED_AT" "$finished" <<'PY'
import json, sys
from pathlib import Path
path, state, detail, started, finished = sys.argv[1:]
value = {"state": state, "detail": detail, "started_at": started}
if finished:
    value["finished_at"] = finished
Path(path).write_text(json.dumps(value, indent=2) + "\n")
PY
    mv "$temporary" "$STATUS"
}

cached=$(find "$BASE/refiner/proposal_cache" -name '*.json.gz' 2>/dev/null | wc -l)
write_status running "detached refiner resume; existing proposal caches=${cached}"

export PYTHONPATH="$BASE/site-packages:$BASE/runtime"
export V21_WORK="$BASE"
export TMPDIR="$BASE/tmp"
export PIP_CACHE_DIR="$BASE/pip-cache"
export TORCH_HOME="$BASE/torch-cache"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
cd "$BASE" || exit 1

"$PY" train_v21_full_refiner.py > refiner_unattended.log 2>&1
return_code=$?
finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
if [[ $return_code -ne 0 || ! -s "$BASE/refiner_full_result.json" ]]; then
    write_status failed "refiner exited rc=${return_code}; inspect refiner_unattended.log" "$finished_at"
    exit "$return_code"
fi

mkdir -p "$PERSIST/checkpoints" "$PERSIST/stages"
rsync -a \
    "$BASE/detector_full_result.json" \
    "$BASE/refiner_full_result.json" \
    "$BASE/detector_smoke_result.json" \
    "$BASE/refiner_runtime_sha256.txt" \
    "$BASE/runtime_sha256.txt" \
    "$BASE/train_v21_full_detector.py" \
    "$BASE/train_v21_full_refiner.py" \
    "$BASE/detector_full.log" \
    "$BASE/refiner_unattended.log" \
    "$PERSIST/"

cp "$BASE/detector/base/final.pt" "$PERSIST/checkpoints/detector_base_final.pt"
cp "$BASE/detector/mask_ratio1/final.pt" "$PERSIST/checkpoints/detector_mask_ratio1_final.pt"
cp "$BASE/refiner/base/final.pt" "$PERSIST/checkpoints/refiner_base_final.pt"
cp "$BASE/refiner/canonical/final.pt" "$PERSIST/checkpoints/refiner_canonical_final.pt"
cp "$BASE/refiner/proposal_centered/final.pt" "$PERSIST/checkpoints/refiner_proposal_centered_final.pt"

cp "$BASE/detector/base/stage_result.json" "$PERSIST/stages/detector_base.json"
cp "$BASE/detector/mask_ratio1/stage_result.json" "$PERSIST/stages/detector_mask_ratio1.json"
cp "$BASE/refiner/base/stage_result.json" "$PERSIST/stages/refiner_base.json"
cp "$BASE/refiner/canonical/stage_result.json" "$PERSIST/stages/refiner_canonical.json"
cp "$BASE/refiner/proposal_centered/stage_result.json" "$PERSIST/stages/refiner_proposal_centered.json"
cp "$BASE/refiner/proposal_cohort.json" "$PERSIST/stages/proposal_cohort.json"

"$PY" - "$BASE" "$PERSIST" <<'PY'
import hashlib, json, sys
from pathlib import Path
base, persistent = map(Path, sys.argv[1:])
detector = json.loads((base / "detector_full_result.json").read_text())
refiner = json.loads((base / "refiner_full_result.json").read_text())

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

checks = {
    "detector_complete": detector.get("status") == "complete",
    "refiner_complete": refiner.get("status") == "complete",
    "population": refiner.get("population") == {
        "files": 707, "records": 1154, "annotations": 8199,
    },
    "detector_hash": sha256(base / "detector/mask_ratio1/final.pt")
        == detector["selected_checkpoint_sha256"],
    "refiner_hash": sha256(base / "refiner/proposal_centered/final.pt")
        == refiner["selected_checkpoint_sha256"],
    "persistent_detector_hash": sha256(persistent / "checkpoints/detector_mask_ratio1_final.pt")
        == detector["selected_checkpoint_sha256"],
    "persistent_refiner_hash": sha256(persistent / "checkpoints/refiner_proposal_centered_final.pt")
        == refiner["selected_checkpoint_sha256"],
}
result = {
    "checks": checks,
    "detector_sha256": detector["selected_checkpoint_sha256"],
    "refiner_sha256": refiner["selected_checkpoint_sha256"],
    "proposal_samples": refiner["proposal_cohort"]["matched_samples"],
}
(persistent / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
if not all(checks.values()):
    raise SystemExit(1)
PY
verification_code=$?
if [[ $verification_code -ne 0 ]]; then
    write_status verification_failed "training completed but artifact verification failed" "$finished_at"
    exit "$verification_code"
fi

write_status complete "refiner complete, verified, and copied to persistent storage" "$finished_at"
cp "$STATUS" "$PERSIST/operation_status.json"
exit 0
