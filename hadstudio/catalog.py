"""Research dataset catalogue and provider-scene import."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .io import load_cube, load_mask
from .utils import ensure_binary_gt

ROOT = Path(__file__).resolve().parent
CATALOG_PATH = ROOT / "dataset_catalog.json"
CACHE = ROOT.parent / "uploads" / "catalog_cache"


def load_catalog() -> list[dict]:
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def public_catalog() -> list[dict]:
    keys = (
        "id",
        "name",
        "collection",
        "description",
        "source",
        "notes",
        "downloadable",
        "format",
        "synthetic",
        "recommended_remove_bands",
    )
    return [{key: item.get(key) for key in keys} for item in load_catalog()]


def _mirrors(url: str) -> list[str]:
    urls = [url]
    prefix = "https://raw.githubusercontent.com/"
    if url.startswith(prefix):
        user, repo, commit, *parts = url[len(prefix):].split("/")
        path = "/".join(parts)
        urls.append(f"https://cdn.jsdelivr.net/gh/{user}/{repo}@{commit}/{path}")
        urls.append(f"https://github.com/{user}/{repo}/raw/{commit}/{path}")
        if commit not in {"master", "main"}:
            urls.append(f"https://raw.githubusercontent.com/{user}/{repo}/master/{path}")
            urls.append(f"https://cdn.jsdelivr.net/gh/{user}/{repo}@master/{path}")
    return urls


def _download(url: str, dest: Path) -> None:
    errors: list[str] = []
    for candidate in _mirrors(url):
        req = Request(candidate, headers={"User-Agent": "HADStudio/1.0"})
        try:
            with urlopen(req, timeout=180) as response, dest.open("wb") as out:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            if dest.is_file() and dest.stat().st_size >= 64:
                return
            errors.append(f"{candidate}: empty response")
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            errors.append(f"{candidate}: {exc}")
    raise ValueError("Download failed. " + " | ".join(errors[:3]))


def fetch_catalog_scene(catalog_id: str) -> dict:
    spec = next((item for item in load_catalog() if item["id"] == catalog_id), None)
    if spec is None:
        raise ValueError("Unknown catalogue dataset.")
    if not spec.get("downloadable"):
        raise ValueError("This scene must be downloaded from the provider repository.")
    CACHE.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for entry in spec["files"]:
        dest = CACHE / Path(entry["filename"]).name
        if not dest.is_file() or dest.stat().st_size < 64:
            _download(entry["url"], dest)
        if dest.stat().st_size < 64:
            raise ValueError(f"Download failed for {dest.name}")
        paths.append(dest)
    loaded = load_cube(paths[0])
    cube = loaded["cube"]
    gt = loaded.get("gt")
    if len(paths) > 1:
        gt = load_mask(paths[1])
    crop = spec.get("crop")
    if crop:
        r0, r1, c0, c1 = [int(v) for v in crop]
        cube = cube[r0:r1, c0:c1]
        if gt is not None and gt.shape != cube.shape[:2] and gt.shape[0] >= r1 and gt.shape[1] >= c1:
            gt = gt[r0:r1, c0:c1]
    if gt is not None:
        gt = ensure_binary_gt(gt, cube.shape[0], cube.shape[1])
    return {
        "name": spec["name"],
        "cube": cube,
        "gt": gt,
        "meta": {**(loaded.get("meta") or {}), "catalog_id": catalog_id, "source": spec["source"]},
    }
