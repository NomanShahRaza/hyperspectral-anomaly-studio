import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hadstudio.detectors.registry import run_detector
from hadstudio.detectors.statistics import center_standard, disfrft, frfe, matlab_rx
from hadstudio.library import find_scene, load_record_arrays
from hadstudio.metrics import evaluate_detection

rng = np.random.default_rng(0)
cube = rng.normal(size=(6, 5, 16))
score = frfe(cube)
assert score.shape == (6, 5)
assert np.isfinite(score).all()

pixels = cube.reshape(-1, 16)
out = disfrft(pixels, 0.5)
assert out.shape == pixels.shape
z = center_standard(np.abs(out).reshape(6, 5, 16))
assert np.isclose(z.mean(), 0.0, atol=1e-10)
print("synthetic", score.shape, float(score.mean()))

loaded = load_record_arrays(find_scene("hydice-urban"))
cube, gt = loaded["cube"], loaded["gt"]
score, _ = run_detector("frfe", cube)
metrics = evaluate_detection(score, gt, detector_id="frfe")
print("hydice", cube.shape, f"{metrics['auc']:.4f}", metrics["roc"])
assert metrics["roc"] == "matlab-demo"
print("ok")
