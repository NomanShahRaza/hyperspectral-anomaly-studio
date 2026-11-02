import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from hadstudio.detectors.matlab_bridge import find_matlab_engine
from hadstudio.detectors.registry import run_detector

print("engine", find_matlab_engine())
cube = np.random.default_rng(0).normal(size=(8, 8, 12))
score, used = run_detector("lsad", cube, {"win_size": 5})
print("lsad", score.shape, float(np.nanmin(score)), float(np.nanmax(score)), used)
assert score.shape == cube.shape[:2]
assert np.isfinite(score).any()
print("ok")
