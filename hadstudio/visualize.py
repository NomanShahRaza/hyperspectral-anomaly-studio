"""Render RGB previews, heatmaps, and overlays for the web client."""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image

from .utils import choose_false_color_bands, minmax_scale, percentile_stretch, to_uint8_rgb


def encode_png(image: np.ndarray) -> str:
    arr = to_uint8_rgb(image)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def false_color(cube: np.ndarray) -> np.ndarray:
    r, g, b = choose_false_color_bands(cube.shape[2])
    rgb = np.stack([cube[:, :, r], cube[:, :, g], cube[:, :, b]], axis=2)
    return percentile_stretch(rgb)


def apply_colormap(score: np.ndarray) -> np.ndarray:
    try:
        import matplotlib.cm as cm

        rgba = cm.inferno(minmax_scale(score))
        return rgba[..., :3]
    except Exception:
        s = minmax_scale(score)
        return np.stack([s, s * 0.45, 1.0 - s], axis=2)


def overlay(rgb: np.ndarray, mask: np.ndarray, color=(1.0, 0.08, 0.86), alpha: float = 0.88) -> np.ndarray:
    out = np.asarray(rgb, dtype=np.float64).copy()
    if out.max() > 1.5:
        out = out / 255.0
    m = (np.asarray(mask) > 0)[..., None].astype(np.float64)
    highlight = np.ones_like(out)
    highlight[..., 0] = color[0]
    highlight[..., 1] = color[1]
    highlight[..., 2] = color[2]
    dimmed = out * 0.42
    return np.clip(dimmed * (1 - m) + highlight * m * alpha + out * m * (1 - alpha), 0.0, 1.0)


def _binary_legend(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask) > 0
    rgb = np.zeros(m.shape + (3,), dtype=np.float64)
    rgb[..., 0] = np.where(m, 1.0, 0.04)
    rgb[..., 1] = np.where(m, 0.95, 0.06)
    rgb[..., 2] = np.where(m, 0.12, 0.12)
    return rgb


def threshold_views(cube: np.ndarray, score: np.ndarray, percentile: float = 99.0) -> dict:
    rgb = false_color(cube)
    score = np.asarray(score, dtype=np.float64)
    percentile = float(np.clip(percentile, 0.0, 100.0))
    thr = float(np.percentile(score, percentile))
    mask = score > thr
    return {
        "overlay": encode_png(overlay(rgb, mask)),
        "binary": encode_png(_binary_legend(mask)),
        "threshold": thr,
        "flagged": int(mask.sum()),
        "fraction": float(mask.mean()),
        "percentile": percentile,
    }


def preview_bundle(cube: np.ndarray, gt: np.ndarray | None = None, score: np.ndarray | None = None, percentile: float = 99.0) -> dict:
    rgb = false_color(cube)
    payload = {
        "rgb": encode_png(rgb),
        "shape": {"height": int(cube.shape[0]), "width": int(cube.shape[1]), "bands": int(cube.shape[2])},
    }
    if gt is not None:
        payload["gt"] = encode_png(np.repeat((gt > 0).astype(np.float64)[..., None], 3, axis=2))
        payload["overlay_gt"] = encode_png(overlay(rgb, gt))
    if score is not None:
        payload["heatmap"] = encode_png(apply_colormap(score))
        payload.update(threshold_views(cube, score, percentile))
    return payload
