import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from hadstudio.detectors.registry import run_detector

cube = np.random.default_rng(0).normal(size=(8, 8, 12))
for det in ("lc", "lcs", "lq", "lqs"):
    score, used = run_detector(det, cube)
    print(det, score.shape, float(np.nanmax(score)), used)
print("ok")
