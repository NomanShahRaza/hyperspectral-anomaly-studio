"""Representation-based detectors: CRD, LSAD, LSAD-CR-IDW, PAB-DC.

CRD runs original ``CRD_Detect.m``. LSAD runs original ``fun_LSAD.m``.
LSAD-CR-IDW runs original ``fun_LSAD_CR_IDW.m``.
PAB-DC follows Huyan et al. (2019): unsupervised background / potential-anomaly
dictionaries plus low-rank and sparse representation.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from ..utils import flatten_pixels, reshape_map
from .lowrank import _soft, _svt
from .statistics import global_rx


def crd(cube: np.ndarray, inner: int = 3, outer: int = 5, lam: float = 1e-2, **_) -> np.ndarray:
    """Original ``CRD_Detect.m`` (Li & Du / Zephyr Hou)."""
    from .matlab_bridge import run_lsunrs_detector

    return run_lsunrs_detector("crd", cube, inner=inner, outer=outer, lam=lam)


def lsad(cube: np.ndarray, win_size: int = 5, **_) -> np.ndarray:
    """Original ``fun_LSAD.m`` (RemoteSensingPaperCode.m uses win_size = 5)."""
    from .matlab_bridge import run_lsunrs_detector

    return run_lsunrs_detector("lsad", cube, win_size=int(win_size), outer=int(win_size))


def lsad_cr_idw(
    cube: np.ndarray,
    inner: int = 1,
    outer: int = 7,
    lam: float = 1e-2,
    **_,
) -> np.ndarray:
    """Original ``fun_LSAD_CR_IDW.m`` (Tan / Zephyr Hou)."""
    from .matlab_bridge import run_lsunrs_detector

    return run_lsunrs_detector("lsad_cr_idw", cube, inner=inner, outer=outer, lam=lam)


def _lrsr(bg_dic: np.ndarray, tg_dic: np.ndarray, data: np.ndarray, n_iter: int = 30) -> np.ndarray:
    """ALM low-rank / sparse representation used by PAB-DC (Huyan et al.)."""
    d = np.hstack([bg_dic, tg_dic])
    x = data
    z = np.zeros((d.shape[1], x.shape[1]))
    j = np.zeros_like(z)
    e = np.zeros_like(x)
    y1 = np.zeros_like(x)
    y2 = np.zeros_like(z)
    mu = 1e-3
    dtd = np.linalg.pinv(d.T @ d + 1e-4 * np.eye(d.shape[1]))
    for _ in range(int(n_iter)):
        z = dtd @ (d.T @ (x - e + y1 / mu) + (j - y2 / mu))
        j = _svt(z + y2 / mu, 0.1 / mu)
        e = _soft(x - d @ z + y1 / mu, 0.08 / mu)
        y1 = y1 + mu * (x - d @ z - e)
        y2 = y2 + mu * (z - j)
        mu = min(mu * 1.4, 1e6)
    n_bg = bg_dic.shape[1]
    anomaly_coeff = z[n_bg:, :]
    return np.sum(np.abs(anomaly_coeff), axis=0) + np.sum(e * e, axis=0)


def pab_dc(
    cube: np.ndarray,
    n_clusters: int = 6,
    bg_atoms: int = 80,
    tg_atoms: int = 40,
    coarse_percentile: float = 95.0,
    **_,
) -> np.ndarray:
    """Unsupervised PAB-DC: RX seeds potential anomalies, then LRSR dictionaries."""
    pixels, (h, w, _) = flatten_pixels(cube)
    coarse = global_rx(cube).ravel()
    potential = coarse >= np.percentile(coarse, float(coarse_percentile))
    background = ~potential
    if background.sum() < 16:
        background[:] = True
        potential[:] = False
        potential[np.argsort(coarse)[-max(8, tg_atoms) :]] = True
        background = ~potential

    k = max(2, min(int(n_clusters), background.sum() // 8, 10))
    km = KMeans(n_clusters=k, n_init=6, random_state=0)
    labels = km.fit_predict(pixels[background])
    bg_list = []
    per = max(4, int(bg_atoms) // k)
    bg_pixels = pixels[background]
    for c in range(k):
        cluster = bg_pixels[labels == c]
        if cluster.size == 0:
            continue
        mean = cluster.mean(axis=0)
        order = np.argsort(np.linalg.norm(cluster - mean, axis=1))[: min(per, cluster.shape[0])]
        bg_list.append(cluster[order])
    bg_dic = np.vstack(bg_list).T if bg_list else pixels[background][:bg_atoms].T

    tg = pixels[potential]
    if tg.shape[0] == 0:
        tg = pixels[np.argsort(coarse)[-int(tg_atoms) :]]
    if tg.shape[0] > int(tg_atoms):
        tkm = KMeans(n_clusters=min(5, tg.shape[0]), n_init=4, random_state=1)
        t_labels = tkm.fit_predict(tg)
        tg_list = []
        per_t = max(2, int(tg_atoms) // max(1, t_labels.max() + 1))
        for c in np.unique(t_labels):
            cluster = tg[t_labels == c]
            mean = cluster.mean(axis=0)
            order = np.argsort(-np.linalg.norm(cluster - mean, axis=1))[: min(per_t, cluster.shape[0])]
            tg_list.append(cluster[order])
        tg_dic = np.vstack(tg_list).T
    else:
        tg_dic = tg.T
    score = _lrsr(bg_dic, tg_dic, pixels.T)
    return reshape_map(score, h, w)
