"""Dual-window neighborhood utilities used by LRX, UNRS, CRD, and local-summation detectors."""

from __future__ import annotations

import numpy as np


def dual_window_mask(inner: int, outer: int) -> np.ndarray:
    if inner < 0 or outer <= 0 or outer <= inner:
        raise ValueError("Require 0 <= inner < outer")
    width = 2 * outer + 1
    yy, xx = np.ogrid[-outer : outer + 1, -outer : outer + 1]
    chebyshev = np.maximum(np.abs(yy), np.abs(xx))
    return (chebyshev > inner) & (chebyshev <= outer)


def iter_valid_centers(height: int, width: int, outer: int):
    for i in range(outer, height - outer):
        for j in range(outer, width - outer):
            yield i, j


def extract_window(
    cube: np.ndarray,
    row: int,
    col: int,
    inner: int,
    outer: int,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    cube = np.asarray(cube)
    h, w, _ = cube.shape
    r0, r1 = row - outer, row + outer + 1
    c0, c1 = col - outer, col + outer + 1
    if r0 < 0 or c0 < 0 or r1 > h or c1 > w:
        return np.empty((0, cube.shape[-1])), np.empty((0, 2), dtype=int)

    patch = cube[r0:r1, c0:c1]
    if mask is None:
        mask = dual_window_mask(inner, outer)
    keep = mask
    samples = patch[keep]
    coords = np.argwhere(keep)
    coords[:, 0] += r0
    coords[:, 1] += c0
    return samples.reshape(-1, cube.shape[-1]), coords


def neighbor_distances(center: np.ndarray, samples: np.ndarray) -> np.ndarray:
    delta = samples - center[None, :]
    return np.sqrt(np.sum(delta * delta, axis=1) + 1e-12)
