"""Quick sanity check of the unified HAD pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hadstudio.detectors.registry import list_detectors, run_detector
from hadstudio.dimred import apply_dimred
from hadstudio.metrics import evaluate_detection
from hadstudio.preprocess import apply_preprocess
from hadstudio.synthetic import make_demo_scene

scene = make_demo_scene()
cube = apply_preprocess(scene["cube"], "minmax")
reduced = apply_dimred(cube, "pca", n_components=8)["cube"]
print("detectors", len(list_detectors()))
print("reduced", reduced.shape)

for name in ["grx", "rsad", "drx", "dwtrx", "csd", "kifd", "aed", "lc", "cbad", "fcbad", "lrasr", "pab_dc"]:
    score, _ = run_detector(name, reduced)
    metrics = evaluate_detection(score, scene["gt"])
    print(f"{name:12s} auc={metrics['auc']:.4f}")
