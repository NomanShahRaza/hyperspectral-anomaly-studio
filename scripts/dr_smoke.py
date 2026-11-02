import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hadstudio.dimred import apply_dimred
from hadstudio.synthetic import make_demo_scene

cube = make_demo_scene()["cube"]
for method in ["pca", "ica", "svd", "factor_analysis", "isomap"]:
    out = apply_dimred(cube, method, n_components=6)
    print(method, out["cube"].shape, out["method"])
