import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hadstudio.detectors.registry import run_detector
from hadstudio.synthetic import make_demo_scene
from hadstudio.visualize import threshold_views

cube = make_demo_scene()["cube"]
score, _ = run_detector("grx", cube)
hi = threshold_views(cube, score, 99)
lo = threshold_views(cube, score, 80)
print("p99", hi["flagged"], hi["fraction"])
print("p80", lo["flagged"], lo["fraction"])
assert lo["flagged"] > hi["flagged"]
print("ok")
