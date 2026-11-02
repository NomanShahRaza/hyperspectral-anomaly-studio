"""Subspace detectors: CSD, UNRS, LSUNRSORAD.

CSD runs original GatorSense ``csd_anomaly.m``.
LSUNRSORAD runs original ``fun_LSUNRS_ED_ORAD.m``.
UNRS runs original ``UNRS_ID_Detect.m``.
"""

from __future__ import annotations

import numpy as np


def csd(cube: np.ndarray, n_dim_bg: int | None = None, n_dim_tgt: int | None = None, tgt_orth: bool = True, **_) -> np.ndarray:
    """Original GatorSense ``csd_anomaly.m``."""
    from .matlab_bridge import run_gator_detector

    bands = int(np.asarray(cube).shape[2])
    n_bg = max(1, bands // 3) if n_dim_bg in (None, 0) else int(n_dim_bg)
    n_tgt = 0 if n_dim_tgt in (None, 0) else int(n_dim_tgt)
    return run_gator_detector("csd", cube, n_dim_bg=n_bg, n_dim_tgt=n_tgt, tgt_orth=bool(tgt_orth))


def unrs(cube: np.ndarray, inner: int = 3, outer: int = 5, lam: float = 1e-2, **_) -> np.ndarray:
    """Original ``UNRS_ID_Detect.m`` (Li & Du / Zephyr Hou)."""
    from .matlab_bridge import run_lsunrs_detector

    return run_lsunrs_detector("unrs", cube, inner=inner, outer=outer, lam=lam)


def lsunrsorad(
    cube: np.ndarray,
    inner: int = 1,
    outer: int = 7,
    lam: float = 1e-3,
    **_,
) -> np.ndarray:
    """Original ``fun_LSUNRS_ED_ORAD.m`` (Tan / Zephyr Hou)."""
    from .matlab_bridge import run_lsunrs_detector

    return run_lsunrs_detector("lsunrsorad", cube, inner=inner, outer=outer, lam=lam)
