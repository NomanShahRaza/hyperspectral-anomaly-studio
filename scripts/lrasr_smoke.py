"""Smoke-test original LRASR (main.m dictionary + lrra.m)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hadstudio.detectors.registry import run_detector
from hadstudio.synthetic import make_demo_scene

scene = make_demo_scene(bands=16)
score, params = run_detector(
    "lrasr",
    scene["cube"],
    {"n_clusters": 4, "atoms_per_cluster": 8, "lam": 0.1},
)
print("params", params)
print("score", score.shape, float(score.min()), float(score.max()), float(score.mean()))
