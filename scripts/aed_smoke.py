import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from hadstudio.detectors.matlab_bridge import find_matlab_engine
from hadstudio.detectors.registry import run_detector

print("engine", find_matlab_engine())
rng = np.random.default_rng(0)
cube = rng.normal(size=(32, 32, 20)) * 0.1
cube[8:14, 8:14, :] += 3
score, used = run_detector("aed", cube, {"para": 25, "n_dim": 3})
print("aed", score.shape, float(np.nanmin(score)), float(np.nanmax(score)), used)
assert score.shape == cube.shape[:2]
assert np.isfinite(score).all()
assert float(np.nanmax(score)) > float(np.nanmin(score))
print("ok")
