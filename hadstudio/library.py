"""Persist imported scenes so they remain in Loaded scenes after restart."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from PIL import Image

from .catalog import CACHE, load_catalog
from .io import load_cube, load_mask
from .utils import ensure_binary_gt
from .visualize import false_color, to_uint8_rgb

ROOT = Path(__file__).resolve().parent.parent
LIBRARY = ROOT / "library"
INDEX = LIBRARY / "index.json"
PREVIEWS = LIBRARY / "previews"
UPLOADS = ROOT / "uploads"
CUBE_SUFFIXES = {".mat", ".npy", ".npz", ".tif", ".tiff", ".hdr", ".img", ".dat"}

_SAFE_ID = re.compile(r"[^a-z0-9._-]+")


def stable_id(key: str, fallback: str = "scene") -> str:
    raw = _SAFE_ID.sub("-", (key or fallback).strip().lower()).strip("-.")
    return (raw or fallback)[:80]


def _rel(path: Path | None) -> str | None:
    if path is None:
        return None
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def resolve_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _read_index() -> dict[str, Any]:
    if not INDEX.is_file():
        return {"scenes": []}
    try:
        data = json.loads(INDEX.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"scenes": []}
    scenes = data.get("scenes")
    return {"scenes": scenes if isinstance(scenes, list) else []}


def _write_index(data: dict[str, Any]) -> None:
    LIBRARY.mkdir(parents=True, exist_ok=True)
    tmp = INDEX.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, INDEX)


def list_scenes() -> list[dict[str, Any]]:
    return list(_read_index()["scenes"])


def find_scene(scene_id: str) -> dict[str, Any] | None:
    return next((item for item in list_scenes() if item.get("id") == scene_id), None)


def preview_path(scene_id: str) -> Path:
    PREVIEWS.mkdir(parents=True, exist_ok=True)
    return PREVIEWS / f"{stable_id(scene_id)}.png"


def save_preview(scene_id: str, cube) -> Path:
    dest = preview_path(scene_id)
    Image.fromarray(to_uint8_rgb(false_color(cube))).save(dest, format="PNG")
    return dest


def upsert_scene(record: dict[str, Any], cube=None) -> dict[str, Any]:
    scene_id = stable_id(str(record.get("id") or record.get("name") or "scene"))
    record = {**record, "id": scene_id}
    if cube is not None:
        save_preview(scene_id, cube)
        shape = {
            "height": int(cube.shape[0]),
            "width": int(cube.shape[1]),
            "bands": int(cube.shape[2]),
        }
        record["shape"] = record.get("shape") or shape
        record.setdefault("original_bands", shape["bands"])
    data = _read_index()
    scenes = [item for item in data["scenes"] if item.get("id") != scene_id]
    scenes.insert(0, record)
    data["scenes"] = scenes
    _write_index(data)
    return record


def load_record_arrays(record: dict[str, Any]) -> dict[str, Any]:
    if record.get("origin") == "demo":
        from .synthetic import make_demo_scene

        loaded = make_demo_scene()
        loaded["name"] = record.get("name") or loaded["name"]
        return loaded
    cube_path = resolve_path(record.get("cube_path"))
    if cube_path is None or not cube_path.is_file():
        raise FileNotFoundError("Scene file is missing. Import it again.")
    loaded = load_cube(cube_path)
    gt_file = resolve_path(record.get("gt_path"))
    if gt_file is not None and gt_file.is_file():
        loaded["gt"] = load_mask(gt_file)
    crop = record.get("crop")
    if crop:
        r0, r1, c0, c1 = [int(v) for v in crop]
        loaded["cube"] = loaded["cube"][r0:r1, c0:c1]
        gt = loaded.get("gt")
        if gt is not None and gt.shape != loaded["cube"].shape[:2] and gt.shape[0] >= r1 and gt.shape[1] >= c1:
            loaded["gt"] = gt[r0:r1, c0:c1]
    if loaded.get("gt") is not None:
        loaded["gt"] = ensure_binary_gt(loaded["gt"], loaded["cube"].shape[0], loaded["cube"].shape[1])
    loaded["name"] = record.get("name") or loaded["name"]
    return loaded


def record_from_loaded(
    loaded: dict[str, Any],
    *,
    scene_id: str,
    origin: str,
    cube_path: Path | None = None,
    gt_path: Path | None = None,
    catalog_id: str | None = None,
    crop: list[int] | None = None,
) -> dict[str, Any]:
    cube = loaded["cube"]
    gt = loaded.get("gt")
    return {
        "id": stable_id(scene_id),
        "name": loaded.get("name") or scene_id,
        "origin": origin,
        "catalog_id": catalog_id,
        "cube_path": _rel(cube_path),
        "gt_path": _rel(gt_path),
        "crop": crop,
        "has_gt": gt is not None,
        "shape": {
            "height": int(cube.shape[0]),
            "width": int(cube.shape[1]),
            "bands": int(cube.shape[2]),
        },
        "original_bands": int(cube.shape[2]),
    }


def catalog_paths(spec: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    names = [Path(entry["filename"]).name for entry in spec.get("files") or [] if entry.get("filename")]
    if not names:
        names = [f"{spec['id']}.mat"]
    for name in names:
        found = None
        for folder in (CACHE, UPLOADS):
            path = folder / name
            if path.is_file() and path.stat().st_size >= 64:
                found = path
                break
        if found is not None:
            paths.append(found)
    return paths


def _catalog_file(spec: dict[str, Any]) -> Path | None:
    paths = catalog_paths(spec)
    return paths[0] if paths else None


def seed_from_disk() -> None:
    """Register already-downloaded catalogue files and leftover local uploads."""
    known_files: set[str] = set()
    catalog_names: set[str] = set()
    for spec in load_catalog():
        for entry in spec.get("files") or []:
            if entry.get("filename"):
                catalog_names.add(Path(entry["filename"]).name.lower())
        if not spec.get("downloadable"):
            continue
        scene_id = spec["id"]
        existing = find_scene(scene_id)
        paths = catalog_paths(spec)
        if existing and preview_path(scene_id).is_file() and existing.get("shape") and existing.get("has_gt"):
            cube_path = resolve_path(existing.get("cube_path"))
            if cube_path and cube_path.is_file():
                known_files.add(cube_path.resolve().as_posix())
                continue
        if not paths:
            continue
        try:
            from .catalog import fetch_catalog_scene

            loaded = fetch_catalog_scene(scene_id)
            upsert_scene(
                record_from_loaded(
                    loaded,
                    scene_id=scene_id,
                    origin="catalog",
                    cube_path=paths[0],
                    gt_path=paths[1] if len(paths) > 1 else None,
                    catalog_id=scene_id,
                    crop=list(spec["crop"]) if spec.get("crop") else None,
                ),
                cube=loaded["cube"],
            )
            known_files.add(paths[0].resolve().as_posix())
        except Exception:
            continue

    if not UPLOADS.is_dir():
        return
    for path in sorted(UPLOADS.iterdir()):
        if not path.is_file() or path.suffix.lower() not in CUBE_SUFFIXES:
            continue
        if path.name.lower() in catalog_names or path.resolve().as_posix() in known_files:
            continue
        scene_id = stable_id(path.stem)
        existing = find_scene(scene_id)
        if existing and preview_path(scene_id).is_file() and existing.get("shape"):
            continue
        try:
            loaded = load_cube(path)
            upsert_scene(
                record_from_loaded(loaded, scene_id=scene_id, origin="local", cube_path=path),
                cube=loaded["cube"],
            )
        except Exception:
            continue
