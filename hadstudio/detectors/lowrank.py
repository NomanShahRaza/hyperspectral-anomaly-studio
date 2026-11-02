"""Low-rank detector: original LRASR / LRR MATLAB (Xu et al., 2016).

The public demo (axiqia ``main.m``) builds a K-means dictionary from the
lowest-RX pixels in each cluster, then scores with the original ``lrra.m``
solver. ``LADMAP.m`` is the paper LRASR ADMM and is vendored but not used
by ``main.m``.
"""

from __future__ import annotations

import numpy as np

from .matlab_bridge import run_lrasr_detector


def _soft(x: np.ndarray, tau: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)


def _svt(x: np.ndarray, tau: float) -> np.ndarray:
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    s = np.maximum(s - tau, 0.0)
    return (u * s) @ vt


def lrasr(
    cube: np.ndarray,
    n_clusters: int = 15,
    atoms_per_cluster: int = 25,
    lam: float = 0.1,
    **_,
) -> np.ndarray:
    """Original axiqia ``main.m`` + ``lrra.m`` (dictionary LRR, L2,1 residual)."""
    return run_lrasr_detector(
        cube,
        n_clusters=int(n_clusters),
        atoms_per_cluster=int(atoms_per_cluster),
        lam=float(lam),
    )
