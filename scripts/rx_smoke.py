import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hadstudio.detectors.registry import DETECTORS, run_detector
from hadstudio.dimred import apply_dimred
from hadstudio.metrics import evaluate_detection
from hadstudio.preprocess import apply_preprocess
from hadstudio.synthetic import make_demo_scene

assert "ridge" not in DETECTORS["grx"]["defaults"]
assert DETECTORS["lrx"]["defaults"] == {"inner": 5, "outer": 21}
assert "ridge" not in DETECTORS["lrx"]["defaults"]

scene = make_demo_scene()
cube = apply_dimred(apply_preprocess(scene["cube"], "minmax"), "pca", n_components=8)["cube"]
for name in ("grx", "lrx"):
    score, used = run_detector(name, cube)
    metrics = evaluate_detection(score, scene["gt"])
    print(name, "auc", metrics["auc"], "params", used, "shape", score.shape)
