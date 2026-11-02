"""Detector catalogue used by the HAD Studio API and HTML client."""

from __future__ import annotations

from typing import Any, Callable

from . import clustering, filtering, kernel, lowrank, representation, statistics, subspace

DetectorFn = Callable[..., Any]

DETECTORS: dict[str, dict[str, Any]] = {
    "grx": {
        "name": "Global RX (GRX)",
        "family": "statistics",
        "fn": statistics.global_rx,
        "speed": "fast",
        "defaults": {},
        "reference": "Reed & Yu, IEEE TASSP, 1990",
        "source": "Spectral Python rx(X)",
    },
    "lrx": {
        "name": "Local RX (LRX)",
        "family": "statistics",
        "fn": statistics.local_rx,
        "speed": "medium",
        "defaults": {"inner": 5, "outer": 21},
        "reference": "Reed & Yu, IEEE TASSP, 1990",
        "source": "Spectral Python rx(X, window=(inner, outer))",
    },
    "rsad": {
        "name": "RSAD",
        "family": "statistics",
        "fn": statistics.rsad,
        "speed": "medium",
        "defaults": {"alpha": 0.01, "max_iter": 20, "n_sel": 40},
        "reference": "Du & Zhang, IEEE TGRS, 2011",
        "source": "Original rsad.m via matlab.engine / oct2py",
    },
    "drx": {
        "name": "Derivative RX (D-RX)",
        "family": "statistics",
        "fn": statistics.derivative_rx,
        "speed": "fast",
        "defaults": {"lag": 4},
        "reference": "Tao et al., IEEE JSTARS, 2019",
        "source": "Original FrFE demo.m + RX.m via matlab.engine / oct2py",
    },
    "dwtrx": {
        "name": "DWT-RX",
        "family": "statistics",
        "fn": statistics.dwt_rx,
        "speed": "medium",
        "defaults": {},
        "reference": "Tao et al., IEEE JSTARS, 2019",
        "source": "Original FrFE demo.m + RX.m via matlab.engine / oct2py",
    },
    "frfe": {
        "name": "FrFE",
        "family": "statistics",
        "fn": statistics.frfe,
        "speed": "slow",
        "defaults": {"order": "auto"},
        "reference": "Tao et al., IEEE JSTARS, 2019",
        "source": "Original FRFEorder.m + Disfrft.m + RX.m via matlab.engine / oct2py",
    },
    "csd": {
        "name": "CSD",
        "family": "subspace",
        "fn": subspace.csd,
        "speed": "fast",
        "defaults": {"n_dim_bg": 3, "tgt_orth": True},
        "reference": "Schaum, IEEE Aerospace Conf., 2004",
        "source": "Original GatorSense csd_anomaly.m via matlab.engine / oct2py",
    },
    "unrs": {
        "name": "UNRS",
        "family": "subspace",
        "fn": subspace.unrs,
        "speed": "slow",
        "defaults": {"inner": 3, "outer": 5, "lam": 0.01},
        "reference": "Li & Du, IEEE IGARSS, 2013",
        "source": "Original UNRS_ID_Detect.m via matlab.engine / oct2py",
    },
    "lsunrsorad": {
        "name": "LSUNRSORAD",
        "family": "subspace",
        "fn": subspace.lsunrsorad,
        "speed": "slow",
        "defaults": {"inner": 1, "outer": 7, "lam": 1e-3, "n_neighbors": 20, "outlier_ratio": 0.2},
        "reference": "Tan et al., Remote Sensing, 2019",
        "source": "Original fun_LSUNRS_ED_ORAD.m via matlab.engine / oct2py",
    },
    "kifd": {
        "name": "KIFD",
        "family": "kernel",
        "fn": kernel.kifd,
        "speed": "slow",
        "defaults": {"kpca_dim": 300, "tree_size": "auto", "tree_num": 1000},
        "reference": "Li et al., IEEE TGRS, 2020",
        "source": "Original KIFD demo.m via matlab.engine / oct2py",
    },
    "aed": {
        "name": "AED",
        "family": "filtering",
        "fn": filtering.aed,
        "speed": "medium",
        "defaults": {"para": 25, "n_dim": 3},
        "reference": "Kang et al., IEEE TGRS, 2017",
        "source": "Original AED demo.m via matlab.engine / oct2py",
    },
    "lc": {
        "name": "Lc",
        "family": "filtering",
        "fn": filtering.lc,
        "speed": "medium",
        "defaults": {},
        "reference": "Verdoja et al., Mach. Vis. Appl., 2020",
        "source": "Original lad_C.m via matlab.engine / oct2py",
    },
    "lcs": {
        "name": "Lcs",
        "family": "filtering",
        "fn": filtering.lcs,
        "speed": "medium",
        "defaults": {},
        "reference": "Verdoja et al., Mach. Vis. Appl., 2020",
        "source": "Original lad_C_S.m via matlab.engine / oct2py",
    },
    "lq": {
        "name": "Lq",
        "family": "filtering",
        "fn": filtering.lq,
        "speed": "medium",
        "defaults": {},
        "reference": "Verdoja et al., Mach. Vis. Appl., 2020",
        "source": "Original lad_Q.m via matlab.engine / oct2py",
    },
    "lqs": {
        "name": "Lqs",
        "family": "filtering",
        "fn": filtering.lqs,
        "speed": "medium",
        "defaults": {},
        "reference": "Verdoja et al., Mach. Vis. Appl., 2020",
        "source": "Original lad_Q_S.m via matlab.engine / oct2py",
    },
    "cbad": {
        "name": "CBAD",
        "family": "clustering",
        "fn": clustering.cbad,
        "speed": "fast",
        "defaults": {"n_clusters": 8},
        "reference": "Carlotto, IEEE TGRS, 2005",
        "source": "Original GatorSense cbad_anomaly.m via matlab.engine / oct2py",
    },
    "fcbad": {
        "name": "FCBAD",
        "family": "clustering",
        "fn": clustering.fcbad,
        "speed": "medium",
        "defaults": {"n_clusters": 8},
        "reference": "Hytla et al., JARS, 2009",
        "source": "Original GatorSense fcbad_anomaly.m via matlab.engine / oct2py",
    },
    "lrasr": {
        "name": "LRASR",
        "family": "lowrank",
        "fn": lowrank.lrasr,
        "speed": "slow",
        "defaults": {"n_clusters": 15, "atoms_per_cluster": 25, "lam": 0.1},
        "reference": "Xu et al., IEEE TGRS, 2016",
        "source": "Original main.m + lrra.m via matlab.engine / oct2py",
    },
    "crd": {
        "name": "CRD",
        "family": "representation",
        "fn": representation.crd,
        "speed": "slow",
        "defaults": {"inner": 3, "outer": 5, "lam": 0.000001},
        "reference": "Li & Du, IEEE TGRS, 2015",
        "source": "Original CRD_Detect.m via matlab.engine / oct2py",
    },
    "lsad": {
        "name": "LSAD",
        "family": "representation",
        "fn": representation.lsad,
        "speed": "slow",
        "defaults": {"win_size": 5},
        "reference": "Du et al., spectral-spatial local summation HAD",
        "source": "Original fun_LSAD.m via matlab.engine / oct2py",
    },
    "lsad_cr_idw": {
        "name": "LSAD-CR-IDW",
        "family": "representation",
        "fn": representation.lsad_cr_idw,
        "speed": "slow",
        "defaults": {"inner": 1, "outer": 7, "lam": 1e-2},
        "reference": "Tan et al., Remote Sensing, 2019",
        "source": "Original fun_LSAD_CR_IDW.m via matlab.engine / oct2py",
    },
    "pab_dc": {
        "name": "PAB-DC",
        "family": "representation",
        "fn": representation.pab_dc,
        "speed": "medium",
        "defaults": {"n_clusters": 6, "bg_atoms": 80, "tg_atoms": 40},
        "reference": "Huyan et al., IEEE TGRS, 2019",
        "source": "huyanning/Hyperspectral-Anomaly-Detection",
    },
}

FAMILIES = {
    "statistics": "Statistics-based",
    "subspace": "Subspace-based",
    "kernel": "Kernel-based",
    "filtering": "Filtering-based",
    "clustering": "Clustering-based",
    "lowrank": "Low-rank-based",
    "representation": "Representation-based",
}

DESCRIPTIONS = {
    "grx": "Global Reed–Xiaoli detector using a background Mahalanobis distance.",
    "lrx": "Spectral Python windowed RX: y = rx(X, window=(inner, outer)). Background is the outer window minus the inner guard window.",
    "rsad": "Executes original LAD rsad.m (Du & Zhang random 2x2 blocks, majority vote).",
    "drx": "Executes original FrFE demo.m D-RX through RX.m.",
    "dwtrx": "Executes original FrFE demo.m DWT-RX through dwt + RX.m.",
    "frfe": "Executes original FRFEorder.m + Disfrft.m + center_standard.m + RX.m.",
    "csd": "Executes original GatorSense csd_anomaly.m.",
    "unrs": "Executes original UNRS_ID_Detect.m (dual-window regularized subspace, L1 residual).",
    "lsunrsorad": "Executes original fun_LSUNRS_ED_ORAD.m.",
    "kifd": "Executes original KIFD demo.m (Gaussian KPCA K=300, M=3% of pixels, q=1000 trees).",
    "aed": "Executes original AED demo.m (PCA, area attribute filter, domain-transform RF).",
    "lc": "Executes original Verdoja lad_C.m (Cauchy Laplacian on bands).",
    "lcs": "Executes original Verdoja lad_C_S.m (Cauchy Laplacian with 4-neighbors).",
    "lq": "Executes original Verdoja lad_Q.m (partial-correlation Laplacian).",
    "lqs": "Executes original Verdoja lad_Q_S.m (partial-correlation Laplacian with 4-neighbors).",
    "cbad": "Executes original GatorSense cbad_anomaly.m via matlab.engine or oct2py.",
    "fcbad": "Executes original GatorSense fcbad_anomaly.m via matlab.engine or oct2py (needs fcm).",
    "lrasr": "Executes original axiqia main.m (K-means / lowest-RX dictionary) then lrra.m (nuclear + L2,1 residual).",
    "crd": "Executes original CRD_Detect.m (dual-window collaborative representation, L2 residual).",
    "lsad": "Executes original fun_LSAD.m (local summation of 5-PC Mahalanobis scores).",
    "lsad_cr_idw": "Executes original fun_LSAD_CR_IDW.m.",
    "pab_dc": "Prior-based background and target dictionary coding.",
}


_ORIGINAL_M = {
    "cbad",
    "fcbad",
    "csd",
    "drx",
    "dwtrx",
    "frfe",
    "lsunrsorad",
    "lsad_cr_idw",
    "unrs",
    "crd",
    "lsad",
    "lc",
    "lcs",
    "lq",
    "lqs",
    "rsad",
    "kifd",
    "lrasr",
    "aed",
}


def _implementation_label(detector_id: str, spec: dict[str, Any]) -> str:
    if detector_id in _ORIGINAL_M:
        return "matlab.engine / oct2py · original .m"
    return "Python"


def _is_ready(detector_id: str, spec: dict[str, Any]) -> bool:
    return True


SOURCE_URLS = {
    "grx": "https://www.spectralpython.net/class_func_ref.html#spectral.algorithms.detectors.rx",
    "lrx": "https://www.spectralpython.net/class_func_ref.html#spectral.algorithms.detectors.rx",
    "rsad": "https://github.com/fverdoja/LAD-Laplacian-Anomaly-Detector",
    "drx": "https://github.com/xudongzhao461/Hyperspectral-Anomaly-Detection-by-Fractional-Fourier-Entropy",
    "dwtrx": "https://github.com/xudongzhao461/Hyperspectral-Anomaly-Detection-by-Fractional-Fourier-Entropy",
    "frfe": "https://github.com/xudongzhao461/Hyperspectral-Anomaly-Detection-by-Fractional-Fourier-Entropy",
    "csd": "https://github.com/GatorSense/hsi_toolkit",
    "unrs": "https://github.com/zephyrhours/Hyperspectral-Anomaly-Detection-LSUNRSORAD-and-LSAD-CR-IDW-",
    "lsunrsorad": "https://github.com/zephyrhours/Hyperspectral-Anomaly-Detection-LSUNRSORAD-and-LSAD-CR-IDW-",
    "kifd": "https://github.com/PuhongDuan/Hyperspectral-anomaly-detection-with-kernel-isolation-forest",
    "aed": "https://github.com/yousefan/AED-algorithm",
    "lc": "https://github.com/fverdoja/LAD-Laplacian-Anomaly-Detector",
    "lcs": "https://github.com/fverdoja/LAD-Laplacian-Anomaly-Detector",
    "lq": "https://github.com/fverdoja/LAD-Laplacian-Anomaly-Detector",
    "lqs": "https://github.com/fverdoja/LAD-Laplacian-Anomaly-Detector",
    "cbad": "https://github.com/GatorSense/hsi_toolkit/blob/master/anomaly_detectors/cbad_anomaly.m",
    "fcbad": "https://github.com/GatorSense/hsi_toolkit/blob/master/anomaly_detectors/fcbad_anomaly.m",
    "lrasr": "https://github.com/axiqia/Anomaly-Detection-in-Hyperspectral-Images-Based-on-Low-Rank-and-Sparse-Representation",
    "crd": "https://github.com/zephyrhours/Hyperspectral-Anomaly-Detection-LSUNRSORAD-and-LSAD-CR-IDW-",
    "lsad": "https://github.com/zephyrhours/Hyperspectral-Anomaly-Detection-LSUNRSORAD-and-LSAD-CR-IDW-",
    "lsad_cr_idw": "https://github.com/zephyrhours/Hyperspectral-Anomaly-Detection-LSUNRSORAD-and-LSAD-CR-IDW-",
    "pab_dc": "https://github.com/huyanning/Hyperspectral-Anomaly-Detection",
}


def list_detectors() -> list[dict[str, Any]]:
    items = []
    for key, spec in DETECTORS.items():
        items.append(
            {
                "id": key,
                "name": spec["name"],
                "family": spec["family"],
                "family_name": FAMILIES[spec["family"]],
                "speed": spec["speed"],
                "defaults": spec["defaults"],
                "reference": spec["reference"],
                "source": spec["source"],
                "url": SOURCE_URLS.get(key, ""),
                "description": DESCRIPTIONS.get(key, spec["reference"]),
                "implementation": _implementation_label(key, spec),
                "ready": _is_ready(key, spec),
            }
        )
    return items


def run_detector(detector_id: str, cube, params: dict[str, Any] | None = None):
    if detector_id not in DETECTORS:
        raise KeyError(f"Unknown detector: {detector_id}")
    spec = DETECTORS[detector_id]
    merged = dict(spec["defaults"])
    if params:
        merged.update({k: v for k, v in params.items() if v is not None and v != ""})
    score = spec["fn"](cube, **merged)
    return score, merged
