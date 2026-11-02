import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hadstudio.preprocess import apply_preprocess, parse_band_selection
from hadstudio.synthetic import make_demo_scene

assert parse_band_selection("0-4, 10", 80) == [0, 1, 2, 3, 4, 10]
cube = make_demo_scene()["cube"]
out = apply_preprocess(cube, "none", {"remove_bands": "0-4, 70-79"})
print("in", cube.shape, "out", out.shape)
assert out.shape[2] == cube.shape[2] - 15
print("ok")
