"""Filtering-based detectors: original AED MATLAB and Laplacian anomaly detectors.

AED executes the original yousefan/AED-algorithm ``demo.m`` path:

- ``matlab_aed/run_aed_detector.m``
- ``matlab_aed/functions/morph_detect.m``
- ``matlab_aed/functions/EMAP_xdk.m``
- ``matlab_aed/functions/RRX_detect.m``
- ``matlab_aed/functions/RF.m``

MATLAB uses the vendored Dalla Mura ``attribute_profile`` MEX. Octave uses
``attribute_profile.m`` (area opening / closing) with the same call signature.

Lc / Lcs / Lq / Lqs execute the original Verdoja LAD MATLAB files:

- ``matlab_lad/lad_C.m``
- ``matlab_lad/lad_C_S.m``
- ``matlab_lad/lad_Q.m``
- ``matlab_lad/lad_Q_S.m``
"""

from __future__ import annotations

from .matlab_bridge import run_aed_detector, run_lad_detector


def aed(
    cube: np.ndarray,
    para: int = 25,
    n_dim: int = 3,
    attr_type: str = "a",
    sigma_s: float = 5.0,
    sigma_r: float = 0.5,
    r: int = 2,
    lamda: float = 100.0,
    **_,
) -> np.ndarray:
    """Original AED ``demo.m``: PCA, area attribute filter, RF refinement."""
    return run_aed_detector(
        cube,
        para=int(para),
        n_dim=int(n_dim),
        attr_type=str(attr_type or "a"),
        sigma_s=float(sigma_s),
        sigma_r=float(sigma_r),
        r=int(r),
        lamda=float(lamda),
    )


def lc(cube: np.ndarray, **_) -> np.ndarray:
    """Original ``lad_C.m`` (Cauchy / spectral Laplacian)."""
    return run_lad_detector("lc", cube)


def lcs(cube: np.ndarray, **_) -> np.ndarray:
    """Original ``lad_C_S.m`` (Cauchy / 4-connected spatial)."""
    return run_lad_detector("lcs", cube)


def lq(cube: np.ndarray, **_) -> np.ndarray:
    """Original ``lad_Q.m`` (partial-correlation Laplacian)."""
    return run_lad_detector("lq", cube)


def lqs(cube: np.ndarray, **_) -> np.ndarray:
    """Original ``lad_Q_S.m`` (partial-correlation / 4-connected spatial)."""
    return run_lad_detector("lqs", cube)
