"""Run original GatorSense .m files from Python.

Backends, in order:
1. ``matlab.engine`` — official MATLAB Engine API (requires a local MATLAB)
2. ``oct2py`` — Python-to-Octave bridge (requires GNU Octave)
3. ``matlab`` / ``octave`` on PATH as a subprocess fallback
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

MATLAB_ROOT = Path(__file__).resolve().parent / "matlab_gator"
FRFE_ROOT = Path(__file__).resolve().parent / "matlab_frfe"
LSUNRS_ROOT = Path(__file__).resolve().parent / "matlab_lsunrs"
LAD_ROOT = Path(__file__).resolve().parent / "matlab_lad"
KIFD_ROOT = Path(__file__).resolve().parent / "matlab_kifd"
LRASR_ROOT = Path(__file__).resolve().parent / "matlab_lrasr"
AED_ROOT = Path(__file__).resolve().parent / "matlab_aed"
MATLAB_PATHS = (
    MATLAB_ROOT,
    MATLAB_ROOT / "anomaly_detectors",
    MATLAB_ROOT / "util",
    FRFE_ROOT,
    LSUNRS_ROOT,
    LAD_ROOT,
    KIFD_ROOT,
    LRASR_ROOT,
    AED_ROOT,
    AED_ROOT / "functions",
)

_SESSION_LOCK = threading.Lock()
_SESSION: tuple[str, object] | None = None


def _octave_executable() -> str | None:
    for name in ("octave-cli", "octave"):
        found = shutil.which(name)
        if found:
            return found
    env = os.environ.get("OCTAVE_EXECUTABLE") or os.environ.get("OCTAVE")
    if env and Path(env).is_file():
        return env
    roots = [
        Path(r"C:\Program Files\GNU Octave"),
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "GNU Octave",
    ]
    for root in roots:
        if not root.is_dir():
            continue
        matches = sorted(root.rglob("octave-cli.exe"), reverse=True)
        if matches:
            return str(matches[0])
    return None


def _matlab_executable() -> str | None:
    found = shutil.which("matlab")
    if found:
        return found
    env = os.environ.get("MATLABROOT")
    if env:
        candidate = Path(env) / "bin" / "matlab.exe"
        if candidate.is_file():
            return str(candidate)
    root = Path(r"C:\Program Files\MATLAB")
    if root.is_dir():
        for child in root.iterdir():
            candidate = child / "bin" / "matlab.exe"
            if candidate.is_file():
                return str(candidate)
    return None


def _add_original_paths_matlab(eng) -> None:
    for path in MATLAB_PATHS:
        eng.addpath(str(path), nargout=0)


def _add_original_paths_oct2py(oc) -> None:
    for path in MATLAB_PATHS:
        oc.addpath(str(path))


def _start_matlab_engine():
    import matlab.engine

    eng = matlab.engine.start_matlab("-nodesktop")
    _add_original_paths_matlab(eng)
    return "matlab.engine", eng


def _start_oct2py():
    exe = _octave_executable()
    if exe:
        os.environ["OCTAVE_EXECUTABLE"] = exe
    from oct2py import Oct2Py

    oc = Oct2Py(executable=exe) if exe else Oct2Py()
    _add_original_paths_oct2py(oc)
    try:
        oc.eval(
            "pkg load statistics; pkg load fuzzy-logic-toolkit; "
            "pkg load signal; pkg load image;"
        )
    except Exception:
        pass
    return "oct2py", oc


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def find_matlab_engine() -> tuple[str, str] | None:
    """Describe the first backend that can execute the original .m files."""
    if _has_module("matlab.engine") and _matlab_executable():
        return "matlab.engine", _matlab_executable() or "matlab"
    exe = _octave_executable()
    if _has_module("oct2py") and exe:
        return "oct2py", exe
    exe = _matlab_executable()
    if exe:
        return "matlab", exe
    exe = _octave_executable()
    if exe:
        return "octave", exe
    return None


def require_matlab_engine() -> tuple[str, str]:
    found = find_matlab_engine()
    if found is None:
        raise RuntimeError(
            "MATLAB-origin detectors execute the original .m files through "
            "matlab.engine (needs a local MATLAB) or oct2py (needs GNU Octave). "
            "Those libraries do not interpret .m files by themselves. "
            "Install MATLAB + `pip install matlabengine`, or install Octave + `pip install oct2py`, "
            "then restart HAD Studio. This machine has neither backend."
        )
    return found


def _open_session() -> tuple[str, object]:
    global _SESSION
    with _SESSION_LOCK:
        if _SESSION is not None:
            return _SESSION
        last_error = None
        if _matlab_executable():
            try:
                _SESSION = _start_matlab_engine()
                return _SESSION
            except Exception as exc:
                last_error = exc
        if _octave_executable():
            try:
                _SESSION = _start_oct2py()
                return _SESSION
            except Exception as exc:
                last_error = exc
        found = find_matlab_engine()
        if found is not None:
            _SESSION = found
            return _SESSION
        if last_error is not None:
            raise RuntimeError(f"Could not start a MATLAB/Octave Python engine. {last_error}")
        require_matlab_engine()
        raise RuntimeError("Could not start a MATLAB/Octave Python engine.")


def _reshape_score(score: np.ndarray, cube: np.ndarray) -> np.ndarray:
    score = np.asarray(score, dtype=np.float64)
    if score.ndim == 2 and score.shape == cube.shape[:2]:
        return score
    return np.reshape(score, cube.shape[:2], order="F")


def _payload(cube: np.ndarray, extras: dict[str, object]) -> dict[str, object]:
    payload: dict[str, object] = {"hsi_img": cube}
    payload.update(extras)
    return payload


def _run_wrapper(wrapper: str, cube: np.ndarray, extras: dict[str, object], timeout: int = 600) -> np.ndarray:
    global _SESSION
    cube = np.asarray(cube, dtype=np.float64)
    if cube.ndim != 3:
        raise ValueError("Original MATLAB detectors expect a 3-D cube.")
    kind, handle = _open_session()
    try:
        with tempfile.TemporaryDirectory(prefix="hadstudio_m_") as tmp:
            tmp_path = Path(tmp)
            input_path = str(tmp_path / "input.mat")
            output_path = str(tmp_path / "output.mat")
            savemat(input_path, _payload(cube, extras))
            if kind == "matlab.engine":
                getattr(handle, wrapper)(input_path, output_path, nargout=0)
            elif kind == "oct2py":
                try:
                    handle.timeout = timeout
                except Exception:
                    pass
                getattr(handle, wrapper)(input_path, output_path, nout=0)
            else:
                roots = ";".join(f"addpath('{path.as_posix()}')" for path in MATLAB_PATHS)
                command = (
                    f"{roots}; {wrapper}('{Path(input_path).as_posix()}',"
                    f"'{Path(output_path).as_posix()}');"
                )
                exe = str(handle)
                args = [exe, "-batch", command] if kind == "matlab" else [exe, "--no-gui", "--eval", command]
                completed = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
                if completed.returncode != 0:
                    detail = (completed.stderr or completed.stdout or "no MATLAB output").strip()
                    raise RuntimeError(f"{wrapper} failed ({kind}). {detail[:800]}")
            if not Path(output_path).is_file():
                raise RuntimeError(f"{wrapper} produced no output.mat")
            score = np.asarray(loadmat(output_path)["out"], dtype=np.float64)
    except Exception:
        with _SESSION_LOCK:
            _SESSION = None
        raise
    return _reshape_score(score, cube)


def run_gator_detector(detector: str, cube: np.ndarray, n_clusters: int = 8, **params) -> np.ndarray:
    """Execute original GatorSense ``cbad_anomaly.m``, ``fcbad_anomaly.m``, or ``csd_anomaly.m``."""
    extras: dict[str, object] = {
        "detector": detector,
        "n_clusters": max(1, int(n_clusters)),
        "n_dim_bg": int(params.get("n_dim_bg") or 0),
        "n_dim_tgt": int(params.get("n_dim_tgt") or 0),
        "tgt_orth": 1 if params.get("tgt_orth", True) else 0,
    }
    return _run_wrapper("run_gator_detector", cube, extras)


def run_frfe_detector(detector: str, cube: np.ndarray, **params) -> np.ndarray:
    """Execute original FrFE ``RX.m`` / ``FRFEorder.m`` / ``Disfrft.m`` scoring."""
    extras: dict[str, object] = {"detector": detector, "lag": max(1, int(params.get("lag") or 4))}
    order = params.get("order", "auto")
    if order not in (None, "", "auto"):
        extras["order"] = float(order)
    return _run_wrapper("run_frfe_detector", cube, extras, timeout=1800)


def run_lsunrs_detector(detector: str, cube: np.ndarray, **params) -> np.ndarray:
    """Execute original UNRS / CRD / LSAD / LSUNRSORAD / LSAD-CR-IDW ``.m`` files."""
    extras = {
        "detector": detector,
        "win_out": max(3, int(params.get("outer") or params.get("win_size") or 5)),
        "win_in": max(1, int(params.get("inner") or 3)),
        "lambda": float(params.get("lam") if params.get("lam") not in (None, "") else 0.01),
    }
    if params.get("win_size") not in (None, ""):
        win_size = max(3, int(params["win_size"]))
        if win_size % 2 == 0:
            win_size += 1
        extras["win_size"] = win_size
    if extras["win_out"] % 2 == 0:
        extras["win_out"] += 1
    if extras["win_in"] % 2 == 0:
        extras["win_in"] += 1
    if extras["win_in"] >= extras["win_out"]:
        extras["win_in"] = max(1, extras["win_out"] - 2)
    return _run_wrapper("run_lsunrs_detector", cube, extras, timeout=1800)


def run_lad_detector(detector: str, cube: np.ndarray, **params) -> np.ndarray:
    """Execute original Verdoja LAD / RSAD ``.m`` files."""
    extras: dict[str, object] = {
        "detector": detector,
        "alpha": float(params.get("alpha") if params.get("alpha") not in (None, "") else 0.01),
        "max_iter": max(1, int(params.get("max_iter") or 20)),
        "n_blocks": max(1, int(params.get("n_blocks") or 80)),
        "n_sel": max(1, int(params.get("n_sel") or 40)),
    }
    return _run_wrapper("run_lad_detector", cube, extras, timeout=1800)


def run_kifd_detector(cube: np.ndarray, **params) -> np.ndarray:
    """Execute original KIFD ``demo.m`` scoring (kpca + iForest + Local_iforest)."""
    extras: dict[str, object] = {
        "tree_num": max(1, int(params.get("tree_num") or 1000)),
        "kpca_train": max(8, int(params.get("kpca_train") or 10000)),
        "kpca_dim": max(1, int(params.get("kpca_dim") or 300)),
        "kernel_param": float(params.get("kernel_param") or 1),
    }
    tree_size = params.get("tree_size")
    if tree_size not in (None, "", "auto"):
        extras["tree_size"] = max(2, int(tree_size))
    return _run_wrapper("run_kifd_detector", cube, extras, timeout=3600)


def run_aed_detector(cube: np.ndarray, **params) -> np.ndarray:
    """Execute original AED ``demo.m`` scoring (PCA, attribute filter, RF)."""
    extras: dict[str, object] = {
        "para": max(1, int(params.get("para") or 25)),
        "n_dim": max(1, int(params.get("n_dim") or 3)),
        "attr_type": str(params.get("attr_type") or "a"),
        "sigma_s": float(params.get("sigma_s") if params.get("sigma_s") not in (None, "") else 5),
        "sigma_r": float(params.get("sigma_r") if params.get("sigma_r") not in (None, "") else 0.5),
        "r": max(1, int(params.get("r") or 2)),
        "lamda": max(1, float(params.get("lamda") or 100)),
    }
    return _run_wrapper("run_aed_detector", cube, extras, timeout=1800)


def run_lrasr_detector(cube: np.ndarray, **params) -> np.ndarray:
    """Execute original axiqia ``main.m`` dictionary + ``lrra.m`` scoring."""
    extras: dict[str, object] = {
        "n_clusters": max(2, int(params.get("n_clusters") or 15)),
        "atoms_per_cluster": max(1, int(params.get("atoms_per_cluster") or 25)),
        "lambda": float(params.get("lam") if params.get("lam") not in (None, "") else 0.1),
    }
    return _run_wrapper("run_lrasr_detector", cube, extras, timeout=1800)
