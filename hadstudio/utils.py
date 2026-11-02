"""Shared array helpers for hyperspectral cubes."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def as_float_cube(cube: np.ndarray) -> np.ndarray:
    arr = np.asarray(cube)
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3-D cube (H, W, B), got shape {arr.shape}")
    return arr.astype(np.float64, copy=False)


def flatten_pixels(cube: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int]]:
    cube = as_float_cube(cube)
    h, w, bands = cube.shape
    return cube.reshape(-1, bands), (h, w, bands)


def reshape_map(values: np.ndarray, height: int, width: int) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(height, width)


def regularized_pinv(cov: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    cov = np.asarray(cov, dtype=np.float64)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("Covariance must be a square matrix")
    eye = np.eye(cov.shape[0], dtype=np.float64)
    scale = float(np.trace(cov) / max(cov.shape[0], 1))
    return np.linalg.pinv(cov + ridge * (scale + 1e-12) * eye)


def mahalanobis_sq(pixels: np.ndarray, mean: np.ndarray, cov_inv: np.ndarray) -> np.ndarray:
    centered = np.asarray(pixels, dtype=np.float64) - np.asarray(mean, dtype=np.float64)
    return np.einsum("ij,jk,ik->i", centered, cov_inv, centered)


def minmax_scale(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    return (arr - lo) / (hi - lo + eps)


def safe_entropy(image: np.ndarray, bins: int = 64) -> float:
    data = np.asarray(image, dtype=np.float64).ravel()
    data = data[np.isfinite(data)]
    if data.size == 0:
        return 0.0
    hist, _ = np.histogram(data, bins=bins, density=False)
    p = hist.astype(np.float64)
    p = p[p > 0]
    p /= p.sum()
    return float(-(p * np.log2(p)).sum())


def choose_false_color_bands(n_bands: int) -> tuple[int, int, int]:
    if n_bands <= 0:
        raise ValueError("Cube has no bands")
    if n_bands == 1:
        return 0, 0, 0
    if n_bands == 2:
        return 1, 0, 0
    red = min(n_bands - 1, int(round(0.72 * (n_bands - 1))))
    green = min(n_bands - 1, int(round(0.42 * (n_bands - 1))))
    blue = min(n_bands - 1, int(round(0.12 * (n_bands - 1))))
    return red, green, blue


def percentile_stretch(image: np.ndarray, low: float = 2.0, high: float = 98.0) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim == 2:
        lo, hi = np.percentile(arr, [low, high])
        return np.clip((arr - lo) / (hi - lo + 1e-12), 0.0, 1.0)
    out = np.zeros_like(arr)
    for b in range(arr.shape[-1]):
        lo, hi = np.percentile(arr[..., b], [low, high])
        out[..., b] = np.clip((arr[..., b] - lo) / (hi - lo + 1e-12), 0.0, 1.0)
    return out


def to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255.0).astype(np.uint8)


def ensure_binary_gt(gt: np.ndarray, height: int, width: int) -> np.ndarray:
    mask = np.asarray(gt)
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = mask.reshape(height, width)
    return (mask > 0).astype(np.uint8)


def unique_sorted(values: Iterable[int]) -> list[int]:
    return sorted({int(v) for v in values})
