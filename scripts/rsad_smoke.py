import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from hadstudio.detectors.registry import run_detector

cube = np.random.default_rng(0).normal(size=(8, 8, 12))
score, used = run_detector("rsad", cube)
print("rsad", score.shape, float(np.nanmax(score)), float(np.nanmean(score)), used)
print("ok")
