"""Radiometric and spatial preprocessing for hyperspectral cubes."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from scipy.ndimage import median_filter

from .utils import as_float_cube


def parse_band_selection(text: str | list[int] | None, count: int) -> list[int]:
    """Parse zero-based indices or inclusive ranges, for example 0,1,20-25."""
    if text is None or text == "":
        return []
    if isinstance(text, (list, tuple)):
        bands = sorted({int(b) for b in text})
        if bands and (bands[0] < 0 or bands[-1] >= count):
            raise ValueError(f"Band selection is outside the valid range 0–{count - 1}.")
        return bands
    result: set[int] = set()
    cleaned = str(text).replace("–", "-").replace("—", "-")
    for token in filter(None, re.split(r"[,;\s]+", cleaned.strip())):
        if not re.fullmatch(r"\d+(?:-\d+)?", token):
            raise ValueError(f"Invalid band selection '{token}'. Use zero-based indices such as 0,1,20-25.")
        ends = [int(v) for v in token.split("-")]
        start, end = ends[0], ends[-1]
        if end < start or start < 0 or end >= count:
            raise ValueError(f"Band selection '{token}' is outside the valid range 0–{count - 1}.")
        result.update(range(start, end + 1))
    return sorted(result)


def minmax_normalize(cube: np.ndarray) -> np.ndarray:
    cube = as_float_cube(cube)
    lo = cube.min(axis=(0, 1), keepdims=True)
    hi = cube.max(axis=(0, 1), keepdims=True)
    return (cube - lo) / (hi - lo + 1e-12)


def zscore_normalize(cube: np.ndarray) -> np.ndarray:
    cube = as_float_cube(cube)
    mean = cube.mean(axis=(0, 1), keepdims=True)
    std = cube.std(axis=(0, 1), keepdims=True)
    return (cube - mean) / (std + 1e-12)


def l2_normalize_spectra(cube: np.ndarray) -> np.ndarray:
    cube = as_float_cube(cube)
    norms = np.linalg.norm(cube, axis=2, keepdims=True)
    return cube / (norms + 1e-12)


def parse_spatial_crop(text: str | list[int] | None, height: int, width: int) -> tuple[int, int, int, int] | None:
    """Parse an inclusive crop such as '0-99, 0-99' or '0,100,0,100' into slice bounds."""
    if text is None or text == "":
        return None
    if isinstance(text, (list, tuple)):
        if len(text) != 4:
            raise ValueError("Crop needs four integers: row start, row end, col start, col end.")
        r0, r1, c0, c1 = [int(v) for v in text]
    else:
        cleaned = str(text).replace("–", "-").replace("—", "-")
        parts = [part.strip() for part in re.split(r"[,;]+", cleaned) if part.strip()]
        if len(parts) == 2 and all("-" in part for part in parts):
            r0, r1 = [int(v) for v in parts[0].split("-", 1)]
            c0, c1 = [int(v) for v in parts[1].split("-", 1)]
            r1 += 1
            c1 += 1
        elif len(parts) == 4:
            r0, r1, c0, c1 = [int(v) for v in parts]
        else:
            raise ValueError("Crop as rows, cols — for example 0-99, 0-99.")
    r0, r1 = sorted((r0, r1))
    c0, c1 = sorted((c0, c1))
    r0 = max(0, r0)
    c0 = max(0, c0)
    r1 = min(height, r1 if r1 > r0 else r1 + 1)
    c1 = min(width, c1 if c1 > c0 else c1 + 1)
    if r1 - r0 < 2 or c1 - c0 < 2:
        raise ValueError("Crop must keep at least a 2×2 window.")
    return r0, r1, c0, c1


def apply_spatial_crop(cube: np.ndarray, crop: tuple[int, int, int, int] | None) -> np.ndarray:
    if not crop:
        return cube
    r0, r1, c0, c1 = crop
    return cube[r0:r1, c0:c1]


def remove_bands(cube: np.ndarray, bands: list[int] | None) -> np.ndarray:
    cube = as_float_cube(cube)
    if not bands:
        return cube
    keep = [i for i in range(cube.shape[2]) if i not in set(int(b) for b in bands)]
    if not keep:
        raise ValueError("All bands would be removed")
    return cube[:, :, keep]


def spatial_median(cube: np.ndarray, size: int = 3) -> np.ndarray:
    cube = as_float_cube(cube)
    size = max(1, int(size))
    if size <= 1:
        return cube
    return median_filter(cube, size=(size, size, 1))


def snv(cube: np.ndarray) -> np.ndarray:
    """Standard normal variate correction along the spectral axis."""
    cube = as_float_cube(cube)
    mean = cube.mean(axis=2, keepdims=True)
    std = cube.std(axis=2, keepdims=True)
    return (cube - mean) / (std + 1e-12)


PREPROCESSORS = {
    "none": lambda cube, **_: as_float_cube(cube),
    "minmax": lambda cube, **_: minmax_normalize(cube),
    "zscore": lambda cube, **_: zscore_normalize(cube),
    "l2": lambda cube, **_: l2_normalize_spectra(cube),
    "snv": lambda cube, **_: snv(cube),
}


def apply_preprocess(cube: np.ndarray, method: str = "minmax", params: dict[str, Any] | None = None) -> np.ndarray:
    params = params or {}
    cube = as_float_cube(cube)
    removed = parse_band_selection(params.get("remove_bands"), cube.shape[2])
    cube = remove_bands(cube, removed)
    cube = apply_spatial_crop(cube, parse_spatial_crop(params.get("crop"), cube.shape[0], cube.shape[1]))
    if params.get("median_size", 0):
        cube = spatial_median(cube, int(params["median_size"]))
    fn = PREPROCESSORS.get(method, PREPROCESSORS["none"])
    return fn(cube)
