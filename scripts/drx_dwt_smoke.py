import sys
from pathlib import Path

import numpy as np
import pywt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hadstudio.detectors.statistics import derivative_rx, dwt_rx, matlab_rx

rng = np.random.default_rng(0)
cube = rng.normal(size=(8, 8, 32))
lag = 4
expected = np.abs(cube[:, :, lag:] - cube[:, :, :-lag])
approx, _ = pywt.dwt(cube, "db1", mode="symmetric", axis=2)
assert expected.shape[2] == 28
assert approx.shape[2] == 16

# Tiny RX.m check: population cov, inv, no ridge.
toy = rng.normal(size=(4, 3, 5))
pixels = toy.reshape(-1, 5).T
centered = pixels - pixels.mean(axis=1, keepdims=True)
sigma = (centered @ centered.T) / pixels.shape[1]
manual = np.einsum("ij,ij->j", np.linalg.inv(sigma) @ centered, centered).reshape(4, 3)
np.testing.assert_allclose(matlab_rx(toy), manual, rtol=1e-12, atol=1e-12)

score_d = derivative_rx(cube, lag=4)
score_w = dwt_rx(cube)
assert score_d.shape == (8, 8)
assert score_w.shape == (8, 8)
print("drx", score_d.shape, float(score_d.mean()))
print("dwtrx", score_w.shape, float(score_w.mean()))
print("matlab_rx ok")
print("ok")
