"""Clustering-based detectors: CBAD and FCBAD.

These call the original GatorSense MATLAB files:

- ``matlab_gator/anomaly_detectors/cbad_anomaly.m``
- ``matlab_gator/anomaly_detectors/fcbad_anomaly.m``
- ``matlab_gator/util/img_det.m``

Executed through ``matlab.engine`` (MATLAB) or ``oct2py`` (GNU Octave).
Needs Statistics Toolbox / ``kmeans``; FCBAD also needs Fuzzy Logic Toolbox ``fcm``.
The demo uses ``n_cluster = 8``.
"""

from __future__ import annotations

import numpy as np

from .matlab_bridge import run_gator_detector


def cbad(cube: np.ndarray, n_clusters: int = 8, **_) -> np.ndarray:
    """Run original ``cbad_anomaly.m``."""
    return run_gator_detector("cbad", cube, n_clusters=n_clusters)


def fcbad(cube: np.ndarray, n_clusters: int = 8, **_) -> np.ndarray:
    """Run original ``fcbad_anomaly.m``."""
    return run_gator_detector("fcbad", cube, n_clusters=n_clusters)
