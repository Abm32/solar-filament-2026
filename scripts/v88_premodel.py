"""V88 (CPU): ordering model for arm B_yhat. Fold-k LightGBM (V83 features/params) trained on the V84
extract rows (soft, t0.50) of folds != k. Writes outputs/v88/premodel_fold{k}.txt.
  python scripts/v88_premodel.py
"""
import pickle
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import v84_bundle as B  # noqa: E402

ROOT = SCRIPTS.parent
OUT = ROOT / "outputs/v88"
OUT.mkdir(parents=True, exist_ok=True)
A = B.arrays(pickle.load(open(ROOT / "outputs/v84/extract_soft.pkl", "rb"))[0.5])
for k in range(5):
    tr = A["fold"] != k
    B.V.fit(A["X"][tr], A["y"][tr], A["w"][tr]).save_model(str(OUT / f"premodel_fold{k}.txt"))
    print("fold", k, int(tr.sum()), flush=True)
