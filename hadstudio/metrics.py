"""Detection-map evaluation: ROC, AUC, and operating-point metrics.

Global RX and other detectors keep the previous sklearn ROC.
D-RX, DWT-RX, and FrFE use the FrFE MATLAB ``demo.m`` ROC so they match
the published table: 5000 linspace thresholds, ``score > tau``, then

    sum((PF(1:end-1)-PF(2:end)).*(PD(2:end)+PD(1:end-1))/2)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import auc, roc_curve

MATLAB_ROC_THRESHOLDS = 5000
MATLAB_ROC_DETECTORS = {"drx", "dwtrx", "frfe"}


def sklearn_roc(score: np.ndarray, gt: np.ndarray) -> dict[str, Any]:
    y_score = np.asarray(score, dtype=np.float64).ravel()
    y_true = (np.asarray(gt) > 0).astype(np.uint8).ravel()
    if y_score.size != y_true.size:
        raise ValueError("Score map and ground truth must have the same number of pixels")
    if y_true.size == 0 or y_true.min() == y_true.max():
        return {
            "auc": None,
            "fpr": [],
            "tpr": [],
            "thresholds": [],
            "note": "Ground truth contains only one class",
            "roc": "sklearn",
        }
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    return {
        "auc": float(auc(fpr, tpr)),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
        "anomaly_pixels": int(y_true.sum()),
        "background_pixels": int((1 - y_true).sum()),
        "roc": "sklearn",
    }


def matlab_roc(score: np.ndarray, gt: np.ndarray, n_tau: int = MATLAB_ROC_THRESHOLDS) -> dict[str, Any]:
    """Port of the FrFE MATLAB demo ROC / AUC."""
    y_score = np.asarray(score, dtype=np.float64).ravel()
    anomaly = (np.asarray(gt) > 0).ravel()
    if y_score.size != anomaly.size:
        raise ValueError("Score map and ground truth must have the same number of pixels")
    if anomaly.size == 0 or anomaly.min() == anomaly.max():
        return {
            "auc": None,
            "fpr": [],
            "tpr": [],
            "thresholds": [],
            "note": "Ground truth contains only one class",
            "roc": "matlab-demo",
        }
    normal = ~anomaly
    n_normal = int(normal.sum())
    n_anomaly = int(anomaly.sum())
    taus = np.linspace(float(y_score.min()), float(y_score.max()), int(n_tau))
    pf = np.empty(taus.size, dtype=np.float64)
    pd = np.empty(taus.size, dtype=np.float64)
    batch = 256
    for start in range(0, taus.size, batch):
        stop = min(taus.size, start + batch)
        flagged = y_score[None, :] > taus[start:stop, None]
        pf[start:stop] = flagged[:, normal].sum(axis=1) / n_normal
        pd[start:stop] = flagged[:, anomaly].sum(axis=1) / n_anomaly
    area = float(np.sum((pf[:-1] - pf[1:]) * (pd[1:] + pd[:-1]) / 2))
    return {
        "auc": area,
        "fpr": pf.tolist(),
        "tpr": pd.tolist(),
        "thresholds": taus.tolist(),
        "anomaly_pixels": n_anomaly,
        "background_pixels": n_normal,
        "roc": "matlab-demo",
        "n_thresholds": int(n_tau),
    }


def evaluate_detection(score: np.ndarray, gt: np.ndarray, detector_id: str | None = None) -> dict[str, Any]:
    if detector_id in MATLAB_ROC_DETECTORS:
        return matlab_roc(score, gt)
    return sklearn_roc(score, gt)


def threshold_map(score: np.ndarray, method: str = "otsu") -> np.ndarray:
    arr = np.asarray(score, dtype=np.float64)
    if method == "percentile":
        thr = np.percentile(arr, 99.0)
    else:
        try:
            from skimage.filters import threshold_otsu

            thr = float(threshold_otsu(arr))
        except Exception:
            thr = float(np.percentile(arr, 99.0))
    return (arr >= thr).astype(np.uint8)
