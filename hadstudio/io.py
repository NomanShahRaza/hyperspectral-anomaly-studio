"""Load hyperspectral cubes and ground-truth maps from common research formats."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

CUBE_KEYS = (
    "data",
    "hsi",
    "img",
    "image",
    "cube",
    "X",
    "x",
    "Sandiego",
    "sandiego",
    "SanDiego",
    "sadiego",
    "san_diego",
    "abu",
    "hydice",
    "salinas",
    "Salians_syn",
    "d",
    "hyperimg",
)

GT_KEYS = (
    "gt",
    "GT",
    "map",
    "groundtruth",
    "groundTruth",
    "PlaneGT",
    "Salians_syn_gt",
    "hsi_gt",
    "mask",
    "anomap",
    "anomaly_map",
)


def _is_array(value: Any) -> bool:
    return isinstance(value, np.ndarray) and value.ndim >= 2 and value.size > 1 and value.dtype != object


def _squeeze_mat(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value)
    while arr.ndim > 3 and 1 in arr.shape:
        arr = np.squeeze(arr)
    return arr


def _first_array(*candidates: np.ndarray | None) -> np.ndarray | None:
    for item in candidates:
        if item is not None:
            return item
    return None


def _orient_cube(arr: np.ndarray) -> np.ndarray:
    arr = _squeeze_mat(arr)
    if arr.ndim == 2:
        return arr[..., None]
    if arr.ndim != 3:
        raise ValueError(f"Cannot interpret array of shape {arr.shape} as an HSI cube")

    dims = arr.shape
    # Two closest dimensions are treated as the spatial plane; the remaining axis is spectral.
    pairs = ((0, 1), (0, 2), (1, 2))
    spatial = min(pairs, key=lambda ij: abs(dims[ij[0]] - dims[ij[1]]))
    spectral = ({0, 1, 2} - set(spatial)).pop()
    close_spatial = abs(dims[spatial[0]] - dims[spatial[1]]) <= max(8, 0.35 * max(dims[spatial[0]], dims[spatial[1]]))
    if not close_spatial and 8 <= dims[2] <= 512:
        spectral = 2
    elif dims[spectral] > max(dims[spatial[0]], dims[spatial[1]]) and 8 <= dims[2] <= 512:
        spectral = 2

    if spectral == 2:
        return arr
    if spectral == 0:
        return np.transpose(arr, (1, 2, 0))
    return np.transpose(arr, (0, 2, 1))


def _walk_matlab(obj: Any, prefix: str, out: dict[str, np.ndarray]) -> None:
    if _is_array(obj):
        out[prefix or "data"] = _squeeze_mat(obj)
        return
    fields = getattr(obj, "_fieldnames", None)
    if fields:
        for field in fields:
            _walk_matlab(getattr(obj, field), field if not prefix else f"{prefix}.{field}", out)
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            if str(key).startswith("__"):
                continue
            _walk_matlab(value, str(key), out)
        return
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        for idx, item in enumerate(obj.ravel()[:8]):
            _walk_matlab(item, prefix or f"item{idx}", out)


def _arrays_from_mat(path: Path) -> dict[str, np.ndarray]:
    from scipy.io import loadmat

    try:
        raw = loadmat(str(path), squeeze_me=False, struct_as_record=False)
        arrays: dict[str, np.ndarray] = {}
        _walk_matlab(raw, "", arrays)
        if arrays:
            return arrays
    except NotImplementedError:
        pass

    import h5py

    arrays = {}
    with h5py.File(path, "r") as handle:
        def _visit(name, obj):
            if hasattr(obj, "shape") and len(obj.shape) >= 2:
                arrays[name.split("/")[-1]] = np.array(obj)

        handle.visititems(_visit)
    return arrays


def _pick_by_keys(arrays: dict[str, np.ndarray], keys: tuple[str, ...], min_ndim: int) -> np.ndarray | None:
    lowered = {k.lower(): v for k, v in arrays.items()}
    for key in keys:
        if key in arrays and arrays[key].ndim >= min_ndim:
            return arrays[key]
        if key.lower() in lowered and lowered[key.lower()].ndim >= min_ndim:
            return lowered[key.lower()]
    return None


def _largest_array(arrays: dict[str, np.ndarray], ndim: int) -> np.ndarray | None:
    candidates = [v for v in arrays.values() if v.ndim == ndim]
    if not candidates:
        return None
    return max(candidates, key=lambda a: a.size)


def load_matlab(path: Path) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    arrays = _arrays_from_mat(path)
    cube = _first_array(_pick_by_keys(arrays, CUBE_KEYS, 3), _largest_array(arrays, 3))
    if cube is None:
        raise ValueError(f"No 3-D hyperspectral cube found in {path.name}")
    cube = _orient_cube(cube)
    gt = _pick_by_keys(arrays, GT_KEYS, 2)
    if gt is not None:
        gt = np.squeeze(gt)
        if gt.ndim == 3:
            gt = gt[..., 0] if gt.shape[-1] < min(gt.shape[:2]) else gt[0]
    meta = {"source": str(path), "variables": {k: list(v.shape) for k, v in arrays.items()}}
    return cube.astype(np.float64), (None if gt is None else np.asarray(gt)), meta


def load_numpy(path: Path) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    if path.suffix.lower() == ".npz":
        bundle = np.load(path, allow_pickle=False)
        arrays = {k: bundle[k] for k in bundle.files}
        cube = _first_array(_pick_by_keys(arrays, CUBE_KEYS, 3), _largest_array(arrays, 3))
        if cube is None:
            raise ValueError(f"No 3-D cube in {path.name}")
        gt = _pick_by_keys(arrays, GT_KEYS, 2)
        return _orient_cube(cube).astype(np.float64), gt, {"source": str(path), "variables": list(arrays)}
    cube = _orient_cube(np.load(path))
    return cube.astype(np.float64), None, {"source": str(path)}


def load_tiff(path: Path) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    import tifffile

    cube = np.asarray(tifffile.imread(str(path)))
    return _orient_cube(cube).astype(np.float64), None, {"source": str(path)}


def load_envi(path: Path) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    import spectral

    hdr = path if path.suffix.lower() == ".hdr" else path.with_suffix(".hdr")
    img = spectral.open_image(str(hdr))
    cube = np.asarray(img.load())
    return _orient_cube(cube).astype(np.float64), None, {"source": str(hdr)}


def load_mask(path: str | Path) -> np.ndarray:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".mat":
        arrays = _arrays_from_mat(path)
    elif suffix == ".npz":
        bundle = np.load(path, allow_pickle=False)
        arrays = {k: bundle[k] for k in bundle.files}
    elif suffix == ".npy":
        arrays = {"array": np.load(path)}
    elif suffix in {".tif", ".tiff"}:
        import tifffile

        arrays = {"array": np.asarray(tifffile.imread(str(path)))}
    else:
        raise ValueError(f"Unsupported ground-truth type: {suffix}")
    gt = _first_array(_pick_by_keys(arrays, GT_KEYS, 2), _largest_array(arrays, 2))
    if gt is None:
        cubeish = _largest_array(arrays, 3)
        if cubeish is not None:
            cubeish = _orient_cube(cubeish)
            gt = cubeish[..., 0]
    if gt is None:
        raise ValueError(f"No 2-D reference mask found in {path.name}")
    gt = np.squeeze(gt)
    if gt.ndim == 3:
        gt = gt[..., 0] if gt.shape[-1] < min(gt.shape[:2]) else gt[0]
    return np.asarray(gt)


def load_cube(path: str | Path, gt_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".mat":
        cube, gt, meta = load_matlab(path)
    elif suffix in {".npy", ".npz"}:
        cube, gt, meta = load_numpy(path)
    elif suffix in {".tif", ".tiff"}:
        cube, gt, meta = load_tiff(path)
    elif suffix in {".hdr", ".img", ".dat"}:
        cube, gt, meta = load_envi(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    if gt_path:
        gt = load_mask(gt_path) if Path(gt_path).is_file() else None

    height, width, bands = cube.shape
    if gt is not None:
        from .utils import ensure_binary_gt

        gt = ensure_binary_gt(gt, height, width)

    return {
        "cube": cube,
        "gt": gt,
        "meta": meta,
        "shape": {"height": height, "width": width, "bands": bands},
        "name": path.stem,
    }
