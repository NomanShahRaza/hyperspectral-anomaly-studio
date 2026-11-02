"""HAD Studio web application."""

from __future__ import annotations

import json
import re
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from hadstudio.catalog import fetch_catalog_scene, load_catalog, public_catalog
from hadstudio.detectors.registry import FAMILIES, list_detectors, run_detector
from hadstudio.dimred import REDUCERS, apply_dimred
from hadstudio.io import load_cube
from hadstudio.library import (
    find_scene,
    list_scenes,
    load_record_arrays,
    preview_path,
    record_from_loaded,
    seed_from_disk,
    stable_id,
    upsert_scene,
)
from hadstudio.metrics import evaluate_detection
from hadstudio.preprocess import PREPROCESSORS, apply_preprocess, parse_spatial_crop
from hadstudio.synthetic import make_demo_scene
from hadstudio.utils import minmax_scale
from hadstudio.visualize import preview_bundle, threshold_views

ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT / "frontend"
RUNS = ROOT / "runs"
UPLOADS = ROOT / "uploads"
RUNS.mkdir(exist_ok=True)
UPLOADS.mkdir(exist_ok=True)

SESSIONS: dict[str, dict[str, Any]] = {}
MAX_RESULTS = 20
_UUID_FILE = re.compile(r"^[0-9a-f]{32}_", re.IGNORECASE)


def _safe_filename(name: str | None, fallback: str) -> str:
    raw = Path(name or fallback).name
    raw = raw.replace("..", "").strip() or fallback
    while _UUID_FILE.match(raw):
        raw = _UUID_FILE.sub("", raw)
    return raw


def _purge_duplicate_uploads() -> None:
    groups: dict[str, list[Path]] = {}
    for path in UPLOADS.iterdir():
        if path.is_file():
            groups.setdefault(_safe_filename(path.name, path.name).lower(), []).append(path)
    for paths in groups.values():
        newest = max(paths, key=lambda item: item.stat().st_mtime)
        original = _safe_filename(newest.name, newest.name)
        dest = UPLOADS / original
        for path in paths:
            if path != newest:
                path.unlink(missing_ok=True)
        if newest.exists() and newest.name != original:
            if dest.exists() and dest != newest:
                dest.unlink()
            newest.rename(dest)


def _purge_stale_runs() -> None:
    for path in RUNS.iterdir():
        if path.is_file() and path.suffix.lower() in {".npy", ".json"}:
            path.unlink(missing_ok=True)


_purge_duplicate_uploads()
_purge_stale_runs()
seed_from_disk()

app = FastAPI(title="HAD Studio", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/assets", StaticFiles(directory=FRONTEND / "assets"), name="assets")


def _session_or_404(session_id: str) -> dict[str, Any]:
    if session_id in SESSIONS:
        return SESSIONS[session_id]
    record = find_scene(session_id)
    if record is None:
        raise HTTPException(404, "Session not found. Load a cube first.")
    try:
        return _hydrate_session(record)
    except Exception as exc:
        raise HTTPException(400, f"Could not reopen saved scene: {exc}") from exc


def _library_preview(scene_id: str) -> dict[str, str]:
    path = preview_path(scene_id)
    if not path.is_file():
        return {}
    return {"rgb": f"/api/library/{scene_id}/preview.png"}


def _public_library_scene(record: dict[str, Any]) -> dict[str, Any]:
    shape = record.get("shape") or {"height": 0, "width": 0, "bands": 0}
    return {
        "id": record["id"],
        "name": record.get("name") or record["id"],
        "shape": shape,
        "original_bands": int(record.get("original_bands") or shape.get("bands") or 0),
        "has_gt": bool(record.get("has_gt")),
        "origin": record.get("origin") or "local",
        "catalog_id": record.get("catalog_id"),
        "preview": _library_preview(record["id"]),
    }


def _remember_result(session: dict[str, Any], record: dict[str, Any]) -> None:
    session["results"][record["id"]] = record
    extra = list(session["results"])[:-MAX_RESULTS]
    for key in extra:
        session["results"].pop(key, None)
    session["last_result"] = record


def _hydrate_session(record: dict[str, Any]) -> dict[str, Any]:
    scene_id = record["id"]
    if scene_id in SESSIONS:
        return SESSIONS[scene_id]
    loaded = load_record_arrays(record)
    session = _new_session(loaded, scene_id=scene_id)
    session["origin"] = record.get("origin") or "local"
    session["history"].append("Reopened saved scene.")
    return session


def _persist_loaded(
    loaded: dict[str, Any],
    *,
    scene_id: str,
    origin: str,
    cube_path: Path | None = None,
    gt_path: Path | None = None,
    catalog_id: str | None = None,
    crop: list[int] | None = None,
) -> dict[str, Any]:
    record = record_from_loaded(
        loaded,
        scene_id=scene_id,
        origin=origin,
        cube_path=cube_path,
        gt_path=gt_path,
        catalog_id=catalog_id,
        crop=crop,
    )
    return upsert_scene(record, cube=loaded["cube"])


def _working_cube(session: dict[str, Any]) -> np.ndarray:
    if session.get("reduced") is not None:
        return session["reduced"]
    if session.get("processed") is not None:
        return session["processed"]
    return session["cube"]


def _new_session(payload: dict[str, Any], scene_id: str | None = None) -> dict[str, Any]:
    session_id = stable_id(scene_id or payload.get("name") or uuid.uuid4().hex)
    if session_id in SESSIONS:
        existing = SESSIONS[session_id]
        existing.update({
            "name": payload.get("name") or existing["name"],
            "cube": payload["cube"],
            "processed": None,
            "reduced": None,
            "gt": payload.get("gt"),
            "meta": payload.get("meta") or existing.get("meta") or {},
        })
        return existing
    session = {
        "id": session_id,
        "name": payload["name"],
        "cube": payload["cube"],
        "processed": None,
        "reduced": None,
        "gt": payload.get("gt"),
        "meta": payload.get("meta") or {},
        "preprocess": "none",
        "dimred": "none",
        "history": [],
        "results": {},
        "origin": payload.get("origin"),
    }
    SESSIONS[session_id] = session
    return session


def _aligned_gt(session: dict[str, Any]):
    gt = session.get("gt")
    if gt is None:
        return None
    cube = _working_cube(session)
    if gt.shape == cube.shape[:2]:
        return gt
    crop = session.get("crop")
    if crop:
        r0, r1, c0, c1 = crop
        cropped = gt[r0:r1, c0:c1]
        if cropped.shape == cube.shape[:2]:
            return cropped
    return None


def _recommended_remove_bands(session: dict[str, Any]) -> str:
    catalog_id = session.get("catalog_id") or (find_scene(session["id"]) or {}).get("catalog_id") or session.get("id")
    spec = next((item for item in load_catalog() if item.get("id") == catalog_id), None)
    return str((spec or {}).get("recommended_remove_bands") or "")


def _public_session(session: dict[str, Any], include_preview: bool = True) -> dict[str, Any]:
    cube = _working_cube(session)
    last = session.get("last_result")
    score = last["score"] if last else None
    gt = _aligned_gt(session)
    body = {
        "id": session["id"],
        "name": session["name"],
        "shape": {"height": int(cube.shape[0]), "width": int(cube.shape[1]), "bands": int(cube.shape[2])},
        "original_bands": int(session["cube"].shape[2]),
        "has_gt": gt is not None or session.get("gt") is not None,
        "origin": session.get("origin") or (find_scene(session["id"]) or {}).get("origin") or "local",
        "catalog_id": (find_scene(session["id"]) or {}).get("catalog_id"),
        "recommended_remove_bands": _recommended_remove_bands(session),
        "preprocess": session["preprocess"],
        "dimred": session["dimred"],
        "history": session["history"][-12:],
        "results": [
            {
                "id": key,
                "detector": item["detector"],
                "name": item["name"],
                "dataset": item.get("dataset") or session["name"],
                "auc": item.get("metrics", {}).get("auc"),
                "elapsed": item.get("elapsed"),
                "params": item.get("params"),
            }
            for key, item in session["results"].items()
        ],
        "last_result": None
        if not last
        else {
            "id": last["id"],
            "detector": last["detector"],
            "name": last["name"],
            "elapsed": last["elapsed"],
            "params": last["params"],
            "metrics": {k: last["metrics"][k] for k in last["metrics"] if k != "thresholds"},
        },
    }
    if include_preview:
        body["preview"] = preview_bundle(cube, gt, score)
    return body


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")


@app.get("/api/catalog")
def catalog():
    return {
        "detectors": list_detectors(),
        "families": FAMILIES,
        "preprocess": list(PREPROCESSORS),
        "dimred": list(REDUCERS),
        "datasets": public_catalog(),
        "library": [_public_library_scene(item) for item in list_scenes()],
    }


@app.get("/api/library")
def library_list():
    return {"scenes": [_public_library_scene(item) for item in list_scenes()]}


@app.get("/api/library/{scene_id}/preview.png")
def library_preview(scene_id: str):
    path = preview_path(scene_id)
    if not path.is_file():
        raise HTTPException(404, "Preview not found")
    return FileResponse(path, media_type="image/png")


def _reset_analysis(session: dict[str, Any]) -> dict[str, Any]:
    session["processed"] = None
    session["reduced"] = None
    session["preprocess"] = "none"
    session["dimred"] = "none"
    session["results"] = {}
    session.pop("last_result", None)
    session.pop("preview", None)
    session.pop("crop", None)
    session["history"].append("Workspace cleared.")
    return session


@app.get("/api/session/{session_id}")
def get_session(session_id: str):
    return _public_session(_session_or_404(session_id))


@app.post("/api/session/{session_id}/reset")
def reset_session(session_id: str):
    return _public_session(_reset_analysis(_session_or_404(session_id)))


@app.post("/api/datasets/fetch")
def fetch_dataset(payload: dict[str, Any]):
    catalog_id = (payload or {}).get("catalog_id")
    if not catalog_id:
        raise HTTPException(400, "catalog_id is required")
    existing = find_scene(str(catalog_id))
    if existing is not None and existing.get("has_gt"):
        session = _hydrate_session(existing)
        session["history"].append(f"Opened saved catalogue scene {session['name']}.")
        return _public_session(session)
    try:
        loaded = fetch_catalog_scene(str(catalog_id))
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(400, f"Could not import catalogue scene: {exc}") from exc
    from hadstudio.catalog import CACHE, load_catalog

    spec = next((item for item in load_catalog() if item["id"] == catalog_id), {})
    files = spec.get("files") or []
    cube_path = CACHE / Path(files[0]["filename"]).name if files else None
    gt_path = CACHE / Path(files[1]["filename"]).name if len(files) > 1 else None
    record = _persist_loaded(
        loaded,
        scene_id=str(catalog_id),
        origin="catalog",
        cube_path=cube_path if cube_path and cube_path.is_file() else None,
        gt_path=gt_path if gt_path and gt_path.is_file() else None,
        catalog_id=str(catalog_id),
        crop=list(spec["crop"]) if spec.get("crop") else None,
    )
    session = _new_session(loaded, scene_id=record["id"])
    session["origin"] = "catalog"
    session["history"].append(f"Imported catalogue scene {loaded['name']}.")
    if loaded.get("gt") is not None:
        session["history"].append("Ground-truth map attached.")
    return _public_session(session)


@app.post("/api/demo")
def load_demo():
    payload = make_demo_scene()
    record = _persist_loaded(payload, scene_id="demo", origin="demo")
    session = _new_session(payload, scene_id=record["id"])
    session["origin"] = "demo"
    session["history"].append("Loaded built-in synthetic demo scene.")
    return _public_session(session)


@app.post("/api/upload")
async def upload(cube: UploadFile = File(...), gt: UploadFile | None = File(None)):
    cube_path = UPLOADS / _safe_filename(cube.filename, "cube.mat")
    cube_path.write_bytes(await cube.read())
    gt_path = None
    if gt is not None and gt.filename:
        gt_path = UPLOADS / _safe_filename(gt.filename, "gt.mat")
        gt_path.write_bytes(await gt.read())
    _purge_duplicate_uploads()
    cube_path = UPLOADS / _safe_filename(cube.filename, "cube.mat")
    if gt_path is not None:
        gt_path = UPLOADS / _safe_filename(gt.filename, "gt.mat")
    try:
        loaded = load_cube(cube_path, gt_path)
    except Exception as exc:
        raise HTTPException(400, f"Could not read cube: {exc}") from exc
    scene_id = stable_id(cube_path.stem)
    record = _persist_loaded(loaded, scene_id=scene_id, origin="local", cube_path=cube_path, gt_path=gt_path)
    session = _new_session(loaded, scene_id=record["id"])
    session["origin"] = "local"
    session["history"].append(f"Loaded {cube.filename} with shape {tuple(loaded['cube'].shape)}.")
    if loaded["gt"] is not None:
        session["history"].append("Ground-truth map attached.")
    return _public_session(session)


@app.post("/api/preprocess")
def preprocess(payload: dict[str, Any]):
    session = _session_or_404(payload["session_id"])
    method = payload.get("method", "minmax")
    params = payload.get("params") or {}
    session["processed"] = apply_preprocess(session["cube"], method=method, params=params)
    session["reduced"] = None
    session["preprocess"] = method
    session["history"].append(f"Preprocess: {method}.")
    return _public_session(session)


@app.post("/api/reduce")
def reduce(payload: dict[str, Any]):
    session = _session_or_404(payload["session_id"])
    source = session.get("processed") if session.get("processed") is not None else session["cube"]
    method = payload.get("method", "pca")
    n_components = int(payload.get("n_components", 10))
    result = apply_dimred(source, method=method, n_components=n_components)
    session["reduced"] = result["cube"]
    session["dimred"] = method
    session["history"].append(f"Dimensionality reduction: {method} → {result['n_components']} components.")
    body = _public_session(session)
    body["dimred_info"] = {k: result[k] for k in result if k != "cube"}
    return body


@app.post("/api/run")
def run_pipeline(payload: dict[str, Any]):
    session = _session_or_404(payload["session_id"])
    prep = payload.get("preprocess", "minmax")
    prep_params = payload.get("prep_params") or {}
    session["crop"] = parse_spatial_crop(prep_params.get("crop"), session["cube"].shape[0], session["cube"].shape[1])
    session["processed"] = apply_preprocess(session["cube"], method=prep, params=prep_params)
    session["preprocess"] = prep
    dr = payload.get("dimred", "pca")
    n_components = int(payload.get("n_components", 10))
    session["reduced"] = apply_dimred(session["processed"], method=dr, n_components=n_components)["cube"]
    session["dimred"] = dr
    detector_id = payload.get("detector")
    if not detector_id:
        raise HTTPException(400, "Detector id is required")
    t0 = time.perf_counter()
    try:
        score, used = run_detector(detector_id, session["reduced"], payload.get("params") or {})
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(500, f"{detector_id} failed: {exc}") from exc
    elapsed = time.perf_counter() - t0
    score = np.asarray(score, dtype=np.float64)
    metrics = (
        evaluate_detection(score, _aligned_gt(session), detector_id=detector_id)
        if _aligned_gt(session) is not None
        else {}
    )
    result_id = uuid.uuid4().hex[:10]
    record = {
        "id": result_id,
        "detector": detector_id,
        "name": next(d["name"] for d in list_detectors() if d["id"] == detector_id),
        "dataset": session["name"],
        "score": score,
        "params": used,
        "elapsed": round(elapsed, 3),
        "metrics": metrics,
    }
    _remember_result(session, record)
    session["history"].append(f"Pipeline {prep} → {dr} → {record['name']} in {elapsed:.2f}s.")
    return _public_session(session)


@app.post("/api/detect")
def detect(payload: dict[str, Any]):
    session = _session_or_404(payload["session_id"])
    detector_id = payload.get("detector")
    if not detector_id:
        raise HTTPException(400, "Detector id is required")
    cube = _working_cube(session)
    params = payload.get("params") or {}
    t0 = time.perf_counter()
    try:
        score, used = run_detector(detector_id, cube, params)
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(500, f"{detector_id} failed: {exc}") from exc
    elapsed = time.perf_counter() - t0
    score = np.asarray(score, dtype=np.float64)
    metrics = (
        evaluate_detection(score, _aligned_gt(session), detector_id=detector_id)
        if _aligned_gt(session) is not None
        else {}
    )
    result_id = uuid.uuid4().hex[:10]
    record = {
        "id": result_id,
        "detector": detector_id,
        "name": next(d["name"] for d in list_detectors() if d["id"] == detector_id),
        "dataset": session["name"],
        "score": score,
        "params": used,
        "elapsed": round(elapsed, 3),
        "metrics": metrics,
    }
    _remember_result(session, record)
    session["history"].append(f"Ran {record['name']} in {elapsed:.2f}s.")
    return _public_session(session)


@app.post("/api/threshold")
def threshold_preview(payload: dict[str, Any]):
    session = _session_or_404(payload["session_id"])
    last = session.get("last_result")
    if not last:
        raise HTTPException(400, "Run a detector before adjusting the overlay threshold.")
    percentile = float(payload.get("percentile", 99))
    views = threshold_views(_working_cube(session), last["score"], percentile)
    if session.get("preview") is None:
        session["preview"] = {}
    session["preview"].update({"overlay": views["overlay"], "binary": views["binary"]})
    return views


@app.post("/api/spectrum")
def spectrum(payload: dict[str, Any]):
    session = _session_or_404(payload["session_id"])
    row = int(payload.get("row", 0))
    col = int(payload.get("col", 0))
    cube = _working_cube(session)
    h, w, _ = cube.shape
    row = min(max(row, 0), h - 1)
    col = min(max(col, 0), w - 1)
    values = np.asarray(cube[row, col], dtype=float).tolist()
    score = None
    last = session.get("last_result")
    if last is not None:
        score_map = np.asarray(last["score"])
        if 0 <= row < score_map.shape[0] and 0 <= col < score_map.shape[1]:
            score = float(score_map[row, col])
    return {"row": row, "col": col, "spectrum": values, "score": score}


@app.get("/api/export/{session_id}/{result_id}")
def export_result(session_id: str, result_id: str):
    session = _session_or_404(session_id)
    if result_id not in session["results"]:
        raise HTTPException(404, "Result not found")
    record = session["results"][result_id]
    path = RUNS / f"{session_id}_{result_id}.json"
    path.write_text(
        json.dumps(
            {
                "detector": record["detector"],
                "name": record["name"],
                "params": record["params"],
                "elapsed": record["elapsed"],
                "metrics": {k: record["metrics"].get(k) for k in ("auc", "anomaly_pixels", "background_pixels")},
                "score": minmax_scale(record["score"]).tolist(),
            }
        ),
        encoding="utf-8",
    )
    return FileResponse(path, filename=f"{record['detector']}_{result_id}.json")


@app.exception_handler(Exception)
async def fallback_error(_, exc: Exception):
    if isinstance(exc, HTTPException):
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
    return JSONResponse({"detail": str(exc)}, status_code=500)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8765, reload=False)
