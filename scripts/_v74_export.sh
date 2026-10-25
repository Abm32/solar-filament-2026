#!/usr/bin/env bash
# V74 export: final p2first_1280 checkpoint -> 4-GPU sharded export -> merge -> validate -> sha256.
# NEVER submits. Reads test IMAGES only. Doc: docs/v74_export_readiness_2026-10-10.md
#   bash scripts/_v74_export.sh /mnt/<work>/p2first_1280_fold99/mask_ratio1/final.pt
# Env overrides: IMGSZ (1280), MIN_AREA (200), REFINER (V65 r3all), GPUS ("0 1 2 3"), OUT (outputs/v74),
#   DEVICE (cuda:0; CPU dry run: GPUS="x x x x" DEVICE=cpu OUT=outputs/v74/dryrun_fold0),
#   MASK_T (0.45), DEMOTE (V84 demote json; empty = off), NAME (submission_v74)
set -euo pipefail
cd "$(dirname "$0")/.."
DET=${1:?usage: _v74_export.sh <detector final.pt>}
PY=.venv/bin/python
OUT=${OUT:-outputs/v74}
IMGSZ=${IMGSZ:-1280}
MIN_AREA=${MIN_AREA:-200}
REFINER=${REFINER:-outputs/v65/export/r3all_refiner_state.pt}
MASK_T=${MASK_T:-0.45}
CONF=${CONF:-0.30}
NAME=${NAME:-submission_v74}
DEMOTE_ARGS=(); [ -n "${DEMOTE:-}" ] && DEMOTE_ARGS=(--demote-model "$DEMOTE")
read -ra GPUS <<< "${GPUS:-0 1 2 3}"
TEST=data/raw/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test/test_images
N=${#GPUS[@]}
mkdir -p "$OUT/export"
DSHA=$(sha256sum "$DET" | cut -d' ' -f1); RSHA=$(sha256sum "$REFINER" | cut -d' ' -f1)
echo "$(date -Is) detector $DET $DSHA | refiner $REFINER $RSHA | imgsz $IMGSZ min_area $MIN_AREA mask_t $MASK_T conf $CONF demote ${DEMOTE:-off}" \
  | tee -a "$OUT/STATUS"

pids=()
for s in $(seq 0 $((N - 1))); do
  CUDA_VISIBLE_DEVICES=${GPUS[$s]} $PY scripts/export_v74.py --shard-index "$s" --num-shards "$N" \
    --detector "$DET" --detector-sha "$DSHA" --refiner "$REFINER" --refiner-sha "$RSHA" \
    --imgsz "$IMGSZ" --min-area "$MIN_AREA" --mask-threshold "$MASK_T" --confidence "$CONF" "${DEMOTE_ARGS[@]}" \
    --device "${DEVICE:-cuda:0}" --output-dir "$OUT/export" > "$OUT/export_shard$s.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || { echo "FAIL shard (see $OUT/export_shard*.log)" | tee -a "$OUT/STATUS"; exit 1; }; done

$PY - "$OUT" "$N" "$TEST" "$MIN_AREA" "$NAME" <<'EOF'
import json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, "scripts")
from validate_submission import validate_submission
out, n, test, min_area, name = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
for s in range(n):
    m = json.loads((out / f"export/shard_{s}.json").read_text())
    assert m["status"] == "complete" and m["num_shards"] == n, f"shard {s} incomplete"
df = pd.concat([pd.read_csv(out / f"export/shard_{s}.csv", dtype=str) for s in range(n)], ignore_index=True)
assert df.filament_id.is_unique
csv = out / f"{name}.csv"
df.to_csv(csv, index=False)
stems = {p.stem for p in test.glob("*.jpeg")}
assert len(stems) == 180, len(stems)
report = validate_submission(csv, stems, min_area=min_area)
(out / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
EOF
(cd "$OUT" && sha256sum "$NAME.csv" > "$NAME.csv.sha256")
echo "$(date -Is) done: $OUT/$NAME.csv $(cut -d' ' -f1 "$OUT/$NAME.csv.sha256") (validated, NOT submitted)" \
  | tee -a "$OUT/STATUS"
