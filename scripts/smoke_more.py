import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hadstudio.detectors.registry import run_detector
from hadstudio.dimred import apply_dimred
from hadstudio.metrics import evaluate_detection
from hadstudio.preprocess import apply_preprocess
from hadstudio.synthetic import make_demo_scene

scene = make_demo_scene()
cube = apply_dimred(apply_preprocess(scene["cube"], "minmax"), "pca", n_components=8)["cube"]
for name in ["csd", "lrx", "crd", "unrs"]:
    score, _ = run_detector(name, cube)
    print(name, evaluate_detection(score, scene["gt"])["auc"])

