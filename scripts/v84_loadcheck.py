"""Check a bare refiner state_dict loads strictly into export_v74's CropUNet. python v84_loadcheck.py <state.pt>"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import export_v74 as X
core, tta, base, m = X.refiner_runtime(Path(sys.argv[1]), "cpu")
print("strict load ok", sum(p.numel() for p in m.parameters()))
