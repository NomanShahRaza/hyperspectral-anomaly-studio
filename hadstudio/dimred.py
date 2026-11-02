"""Dimensionality reduction for hyperspectral preprocessing."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.decomposition import PCA, FastICA, FactorAnalysis, TruncatedSVD
from sklearn.manifold import Isomap

from .utils import as_float_cube, flatten_pixels


def _components(n_bands: int, n_components: int) -> int:
    return max(1, min(int(n_components), n_bands, 64))


def pca_reduce(cube: np.ndarray, n_components: int = 10) -> dict[str, Any]:
    pixels, (h, w, bands) = flatten_pixels(cube)
    k = _components(bands, n_components)
    model = PCA(n_components=k, svd_solver="auto")
    scores = model.fit_transform(pixels)
    return {
        "cube": scores.reshape(h, w, k),
        "explained": model.explained_variance_ratio_.tolist(),
        "method": "pca",
        "n_components": k,
    }


def ica_reduce(cube: np.ndarray, n_components: int = 10) -> dict[str, Any]:
    pixels, (h, w, bands) = flatten_pixels(cube)
    k = _components(min(bands, pixels.shape[0] - 1), n_components)
    try:
        model = FastICA(n_components=k, whiten="unit-variance", max_iter=400, random_state=0)
        scores = model.fit_transform(pixels)
    except Exception:
        model = PCA(n_components=k, svd_solver="auto")
        scores = model.fit_transform(pixels)
    return {"cube": scores.reshape(h, w, k), "method": "ica", "n_components": k}


def svd_reduce(cube: np.ndarray, n_components: int = 10) -> dict[str, Any]:
    pixels, (h, w, bands) = flatten_pixels(cube)
    k = _components(bands, n_components)
    model = TruncatedSVD(n_components=k, random_state=0)
    scores = model.fit_transform(pixels)
    return {
        "cube": scores.reshape(h, w, k),
        "explained": model.explained_variance_ratio_.tolist(),
        "method": "svd",
        "n_components": k,
    }


def factor_analysis_reduce(cube: np.ndarray, n_components: int = 10) -> dict[str, Any]:
    pixels, (h, w, bands) = flatten_pixels(cube)
    k = _components(bands, n_components)
    model = FactorAnalysis(n_components=k, max_iter=400, random_state=0)
    scores = model.fit_transform(pixels)
    return {"cube": scores.reshape(h, w, k), "method": "factor_analysis", "n_components": k}


def isomap_reduce(cube: np.ndarray, n_components: int = 10, n_neighbors: int = 20) -> dict[str, Any]:
    pixels, (h, w, bands) = flatten_pixels(cube)
    k = _components(bands, n_components)
    n_samples = pixels.shape[0]
    # Isomap is O(n^2). Fit on a stratified subsample, then interpolate with nearest neighbors.
    max_fit = 2500
    neighbors = max(3, min(int(n_neighbors), n_samples - 1, 50))
    if n_samples <= max_fit:
        model = Isomap(n_components=k, n_neighbors=neighbors)
        scores = model.fit_transform(pixels)
    else:
        rng = np.random.default_rng(0)
        idx = rng.choice(n_samples, size=max_fit, replace=False)
        model = Isomap(n_components=k, n_neighbors=neighbors)
        fitted = model.fit(pixels[idx])
        scores = fitted.transform(pixels)
    return {"cube": scores.reshape(h, w, k), "method": "isomap", "n_components": k}


REDUCERS = {
    "none": lambda cube, **_: {"cube": as_float_cube(cube), "method": "none", "n_components": as_float_cube(cube).shape[2]},
    "pca": pca_reduce,
    "ica": ica_reduce,
    "svd": svd_reduce,
    "factor_analysis": factor_analysis_reduce,
    "isomap": isomap_reduce,
}


def apply_dimred(cube: np.ndarray, method: str = "pca", n_components: int = 10, **kwargs: Any) -> dict[str, Any]:
    fn = REDUCERS.get(method)
    if fn is None:
        raise ValueError(f"Unknown dimensionality reduction method: {method}")
    return fn(as_float_cube(cube), n_components=n_components, **kwargs)
