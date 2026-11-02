"""Statistics-based detectors: GRX, LRX, RSAD, D-RX, DWT-RX, FrFE.

GRX/LRX follow Spectral Python ``spectral.rx``.
D-RX, DWT-RX, and FrFE follow the FrFE MATLAB demo (Tao et al., 2019):
``demo.m``, ``FRFEorder.m``, KU Leuven ``Disfrft.m``, ``center_standard.m``,
and ``RX.m``.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pywt

from scipy.linalg import circulant

from ..utils import flatten_pixels, mahalanobis_sq, regularized_pinv, reshape_map


def _odd_width(value: int, fallback: int) -> int:
    value = int(value) if value is not None else fallback
    if value < 1:
        value = fallback
    if value % 2 == 0:
        value += 1
    return value


def _fit_spy_window(height: int, width: int, inner: int, outer: int) -> tuple[int, int]:
    limit = min(height, width)
    if limit % 2 == 0:
        limit -= 1
    outer = min(outer, max(3, limit))
    if outer % 2 == 0:
        outer -= 1
    inner = min(inner, max(1, outer - 2))
    if inner % 2 == 0:
        inner -= 1
    if inner < 1:
        inner = 1
    return inner, outer


def _spy_rx(cube: np.ndarray, window: tuple[int, int] | None = None) -> np.ndarray:
    """Spectral Python ``rx(X)`` / ``rx(X, window=(inner, outer))``. No ``cov`` argument."""
    from spectral.algorithms.detectors import rx

    cube = np.ascontiguousarray(np.asarray(cube, dtype=np.float64))
    if window is None:
        return np.asarray(rx(cube), dtype=np.float64)
    return np.asarray(rx(cube, window=window), dtype=np.float64)


def matlab_rx(cube: np.ndarray) -> np.ndarray:
    """Line-for-line port of FrFE ``RX.m``.

    MATLAB (bands × pixels)::

        X_mean = mean(X.').';
        X = X - repmat(X_mean, [1 M]);
        Sigma = (X * X') / M;
        Sigma_inv = inv(Sigma);
        D(m) = X(:, m)' * Sigma_inv * X(:, m);

    This is not Spectral Python ``rx``. Covariance is the population
    matrix ``XX'/M`` (divide by pixel count, not ``M-1``) and the inverse
    is unregularized, matching ``inv(Sigma)``.
    """
    cube = np.asarray(cube, dtype=np.float64)
    height, width, bands = cube.shape
    pixels = cube.reshape(height * width, bands).T
    n_pixels = pixels.shape[1]
    centered = pixels - pixels.mean(axis=1, keepdims=True)
    sigma = (centered @ centered.T) / n_pixels
    try:
        sigma_inv = np.linalg.inv(sigma)
    except np.linalg.LinAlgError:
        sigma_inv = np.linalg.pinv(sigma)
    scores = np.einsum("ij,ij->j", sigma_inv @ centered, centered, optimize=True)
    return scores.reshape(height, width)


def _rx_from_stats(pixels: np.ndarray, background: np.ndarray | None = None, ridge: float = 1e-6) -> np.ndarray:
    bg = pixels if background is None else background
    mean = bg.mean(axis=0)
    cov = np.cov(bg, rowvar=False)
    cov_inv = regularized_pinv(cov, ridge=ridge)
    return mahalanobis_sq(pixels, mean, cov_inv)


def _matlab_demo_cube(cube: np.ndarray) -> np.ndarray:
    """``DataTest = DataTest ./ max(DataTest(:))`` from FrFE ``demo.m``."""
    cube = np.asarray(cube, dtype=np.float64)
    peak = float(np.max(cube))
    if peak > 0:
        cube = cube / peak
    return cube


def global_rx(cube: np.ndarray, **_) -> np.ndarray:
    """Global RX: Spectral Python ``rx(X)``."""
    return _spy_rx(cube)


def local_rx(cube: np.ndarray, inner: int = 5, outer: int = 21, **_) -> np.ndarray:
    """Local RX: Spectral Python ``rx(X, window=(inner, outer))``.

    ``inner`` and ``outer`` are odd window widths in pixels. Background
    statistics use the outer window excluding the inner guard window.
    """
    cube = np.ascontiguousarray(np.asarray(cube, dtype=np.float64))
    h, w, _ = cube.shape
    inner, outer = _fit_spy_window(h, w, _odd_width(inner, 5), _odd_width(outer, 21))
    return _spy_rx(cube, window=(inner, outer))


def rsad(
    cube: np.ndarray,
    alpha: float = 0.01,
    max_iter: int = 20,
    n_blocks: int | None = None,
    n_sel: int = 40,
    **_,
) -> np.ndarray:
    """Original LAD ``rsad.m`` (Du & Zhang, 2011). ``4 * n_blocks > n_bands``."""
    from .matlab_bridge import run_lad_detector

    bands = int(np.asarray(cube).shape[2])
    blocks = int(n_blocks) if n_blocks not in (None, "") else max(80, bands // 4 + 1)
    if 4 * blocks <= bands:
        blocks = bands // 4 + 1
    return run_lad_detector(
        "rsad",
        cube,
        alpha=float(alpha),
        max_iter=int(max_iter),
        n_blocks=blocks,
        n_sel=max(1, int(n_sel)),
    )


def derivative_rx(cube: np.ndarray, lag: int = 4, **_) -> np.ndarray:
    """Original FrFE ``demo.m`` D-RX through ``RX.m``."""
    from .matlab_bridge import run_frfe_detector

    lag = max(1, int(lag))
    if lag >= np.asarray(cube).shape[2]:
        raise ValueError("D-RX lag must be smaller than the number of bands.")
    return run_frfe_detector("drx", cube, lag=lag)


def dwt_rx(cube: np.ndarray, **_) -> np.ndarray:
    """Original FrFE ``demo.m`` DWT-RX through ``dwt`` + ``RX.m``."""
    from .matlab_bridge import run_frfe_detector

    return run_frfe_detector("dwtrx", cube)


@lru_cache(maxsize=8)
def _disfrft_basis(n: int) -> np.ndarray:
    """Eigenvectors of the DFT from KU Leuven ``Disfrft.m`` / ``make_E`` (Candan 2000)."""
    if n < 3:
        raise ValueError("FrFE needs at least three spectral features.")
    approx = min(max(2.0, n / 2.0), n - 1)
    stencil = np.array([1.0], dtype=np.float64)
    operator = np.zeros(n, dtype=np.float64)
    for k in range(1, int(approx / 2) + 1):
        stencil = np.convolve(stencil, [1.0, -2.0, 1.0])
        wrapped = np.zeros(n, dtype=np.float64)
        wrapped[np.r_[np.arange(n - k, n), np.arange(k + 1)]] = stencil
        wrapped[0] = 0.0
        numerators = np.repeat(np.r_[1.0, np.arange(1, k)], 2)
        coefficient = 2.0 * ((-1) ** (k - 1)) * np.prod(numerators / np.arange(1, 2 * k + 1))
        operator += coefficient * wrapped
    hermite = circulant(operator) + np.diag(np.fft.fft(operator).real)
    half = n // 2
    parity = (np.eye(n - 1) + np.fliplr(np.eye(n - 1))) / np.sqrt(2.0)
    start = n - half - 1
    parity[start:, start:] *= -1
    if n % 2 == 0:
        parity[half - 1, half - 1] = 1.0
    transform = np.eye(n)
    transform[1:, 1:] = parity
    block = transform @ hermite @ transform.T
    _, even_vectors = np.linalg.eigh(block[: half + 1, : half + 1])
    _, odd_vectors = np.linalg.eigh(block[half + 1 :, half + 1 :])
    vectors = np.zeros((n, n), dtype=np.float64)
    vectors[: half + 1, : half + 1] = even_vectors[:, ::-1]
    vectors[half + 1 :, half + 1 :] = odd_vectors[:, ::-1]
    vectors = transform @ vectors
    even_idx = list(range(half + 1))
    odd_idx = list(range(half + 1, n))
    order_idx: list[int] = []
    for i in range(half + 1):
        order_idx.append(even_idx[i])
        if i < len(odd_idx):
            order_idx.append(odd_idx[i])
    return vectors[:, order_idx]


def disfrft(spectra: np.ndarray, order: float) -> np.ndarray:
    """Batch port of KU Leuven ``Disfrft(f, a)``. ``spectra`` is pixels×bands."""
    spectra = np.asarray(spectra, dtype=np.float64)
    bands = spectra.shape[1]
    basis = _disfrft_basis(bands)
    shift = (np.arange(bands) + bands // 2) % bands
    powers = np.arange(bands, dtype=np.float64)
    if bands % 2 == 0:
        powers[-1] += 1
    phase = np.exp(-0.5j * np.pi * float(order) * powers)
    transformed = ((spectra[:, shift] @ basis) * phase) @ basis.T
    out = np.empty_like(transformed)
    out[:, shift] = transformed
    return out


def center_standard(cube: np.ndarray) -> np.ndarray:
    """MATLAB ``center_standard.m``: global mean and sample-std of all voxels."""
    cube = np.asarray(cube, dtype=np.float64)
    temp = cube.ravel()
    sigma = float(np.sqrt(np.cov(temp, ddof=1)))
    if not np.isfinite(sigma) or sigma == 0.0:
        sigma = 1.0
    return (cube - float(temp.mean())) / sigma


def matlab_image_entropy(image: np.ndarray) -> float:
    """Image Processing Toolbox ``entropy`` on a double image (256-bin ``im2uint8``)."""
    quantized = np.rint(np.clip(np.asarray(image, dtype=np.float64), 0.0, 1.0) * 255.0).astype(np.uint8)
    counts = np.bincount(quantized.ravel(), minlength=256).astype(np.float64)
    positive = counts[counts > 0]
    if positive.size == 0:
        return 0.0
    probability = positive / quantized.size
    return float(-(probability * np.log2(probability)).sum())


def _frfe_select_order(pixels: np.ndarray, height: int, width: int, bands: int) -> float:
    """``FRFEorder.m``: search p=0:0.1:1, then ``order = index/10``."""
    best_score = -np.inf
    best_index = 1
    for index, order in enumerate(np.round(np.arange(0.0, 1.01, 0.1), decimals=1), start=1):
        features = center_standard(np.abs(disfrft(pixels, order)).reshape(height, width, bands))
        score = max(matlab_image_entropy(features[:, :, band]) for band in range(bands))
        if score > best_score:
            best_score = score
            best_index = index
    return best_index / 10.0


def _parse_frfe_order(order) -> float | None:
    if order is None:
        return None
    text = str(order).strip().lower()
    if text in {"", "auto", "none"}:
        return None
    return float(order)


def frfe(cube: np.ndarray, order: str | float = "auto", **_) -> np.ndarray:
    """Original FrFE ``FRFEorder.m`` + ``Disfrft.m`` + ``center_standard.m`` + ``RX.m``."""
    from .matlab_bridge import run_frfe_detector

    applied = _parse_frfe_order(order)
    return run_frfe_detector("frfe", cube, order="auto" if applied is None else applied)
