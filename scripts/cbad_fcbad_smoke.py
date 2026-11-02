import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hadstudio.detectors.matlab_bridge import find_matlab_engine, require_matlab_engine

found = find_matlab_engine()
print("engine", found)
if found is None:
    try:
        require_matlab_engine()
    except RuntimeError as exc:
        print("expected-without-backend", str(exc)[:220])
        print("ok")
        raise SystemExit(0)
from hadstudio.detectors.registry import run_detector
import numpy as np

cube = np.random.default_rng(0).normal(size=(8, 8, 12))
for det in ("cbad", "fcbad"):
    score, used = run_detector(det, cube)
    print(det, score.shape, used)
print("ok")
