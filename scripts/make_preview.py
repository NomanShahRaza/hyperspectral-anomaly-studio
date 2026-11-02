"""Write a dark collage of the demo scene for the GitHub README."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hadstudio.detectors.registry import run_detector
from hadstudio.dimred import apply_dimred
from hadstudio.preprocess import apply_preprocess
from hadstudio.synthetic import make_demo_scene
from hadstudio.visualize import apply_colormap, false_color, overlay, to_uint8_rgb


def _panel(image: np.ndarray, size: int = 280) -> Image.Image:
    rgb = to_uint8_rgb(image)
    return Image.fromarray(rgb).resize((size, size), Image.NEAREST)


def main() -> None:
    out = ROOT / "docs" / "workbench.png"
    out.parent.mkdir(exist_ok=True)
    scene = make_demo_scene()
    cube = apply_dimred(apply_preprocess(scene["cube"], "minmax"), "pca", n_components=10)["cube"]
    score, _ = run_detector("grx", cube)
    rgb = false_color(scene["cube"])
    gt = np.repeat((scene["gt"] > 0).astype(np.float64)[..., None], 3, axis=2)
    heat = apply_colormap(score)
    over = overlay(rgb, score > np.quantile(score, 0.97))
    gap, pad, size = 10, 14, 280
    collage = Image.new("RGB", (pad * 2 + size * 4 + gap * 3, pad * 2 + size), (8, 16, 24))
    x = pad
    for image in (rgb, gt, heat, over):
        collage.paste(_panel(image, size), (x, pad))
        x += size + gap
    collage.save(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
