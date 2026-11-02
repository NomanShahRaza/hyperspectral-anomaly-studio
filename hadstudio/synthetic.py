"""Synthetic hyperspectral scene with planted anomalies for the in-app demo."""

from __future__ import annotations

from typing import Any

import numpy as np


def make_demo_scene(height: int = 64, width: int = 64, bands: int = 80, seed: int = 7) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    wavelengths = np.linspace(400.0, 2400.0, bands)
    materials = []
    centers = [520, 680, 900, 1450, 1700]
    widths = [40, 55, 80, 120, 90]
    for i in range(4):
        spectrum = 0.15 + 0.25 * rng.random()
        for c, w in zip(centers, widths):
            amp = 0.15 + 0.55 * rng.random()
            spectrum = spectrum + amp * np.exp(-0.5 * ((wavelengths - (c + 40 * i)) / w) ** 2)
        materials.append(spectrum)
    materials = np.asarray(materials)

    labels = np.zeros((height, width), dtype=int)
    labels[:, : width // 2] = 0
    labels[:, width // 2 :] = 1
    labels[height // 2 :, : width // 2] = 2
    labels[height // 2 :, width // 2 :] = 3

    cube = materials[labels]
    cube = cube + 0.015 * rng.normal(size=cube.shape)

    gt = np.zeros((height, width), dtype=np.uint8)
    anomaly = 0.08 + 0.9 * np.exp(-0.5 * ((wavelengths - 1100.0) / 35.0) ** 2)
    anomaly += 0.4 * np.exp(-0.5 * ((wavelengths - 2100.0) / 50.0) ** 2)
    plants = [(10, 12, 4, 3), (18, 40, 3, 3), (44, 16, 5, 2), (50, 48, 3, 4)]
    for r, c, h, w in plants:
        cube[r : r + h, c : c + w] = anomaly + 0.01 * rng.normal(size=(h, w, bands))
        gt[r : r + h, c : c + w] = 1

    return {
        "cube": cube.astype(np.float64),
        "gt": gt,
        "name": "synthetic_demo",
        "shape": {"height": height, "width": width, "bands": bands},
        "meta": {
            "source": "synthetic",
            "description": "Four-material background with four planted spectral anomalies.",
            "wavelengths": wavelengths.tolist(),
        },
    }
