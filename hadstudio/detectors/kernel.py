"""Kernel-based detector: original KIFD MATLAB (Li / Duan et al., IEEE TGRS 2020)."""

from __future__ import annotations

import math

import numpy as np

from .matlab_bridge import run_kifd_detector


def default_tree_size(height: int, width: int) -> int:
    """Paper default: M = floor(3% × rows × columns), at least 2."""
    return max(2, int(math.floor(3 * height * width / 100)))


def kifd(
    cube: np.ndarray,
    tree_num: int = 1000,
    tree_size=None,
    kpca_train: int = 10000,
    kpca_dim: int = 300,
    kernel_param: float = 1.0,
    **_,
) -> np.ndarray:
    """Original KIFD ``demo.m``: Gaussian KPCA, global iForest, then local iForest."""
    height, width = cube.shape[:2]
    if tree_size in (None, "", "auto"):
        m = default_tree_size(height, width)
    else:
        m = max(2, int(tree_size))
    return run_kifd_detector(
        cube,
        tree_num=int(tree_num),
        tree_size=m,
        kpca_train=int(kpca_train),
        kpca_dim=int(kpca_dim),
        kernel_param=float(kernel_param),
    )
