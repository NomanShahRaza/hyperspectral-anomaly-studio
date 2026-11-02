import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hadstudio.library import find_scene, load_record_arrays, seed_from_disk
from hadstudio.metrics import evaluate_detection

seed_from_disk()

for scene_id in ("aviris-i", "salinas-synthetic"):
    record = find_scene(scene_id)
    print("record", scene_id, record.get("has_gt") if record else None, record.get("gt_path") if record else None)
    if record:
        loaded = load_record_arrays(record)
        gt = loaded.get("gt")
        print("  loaded", loaded["cube"].shape, None if gt is None else (gt.shape, int(gt.sum())))
        if gt is not None:
            dummy = loaded["cube"][..., 0]
            metrics = evaluate_detection(dummy, gt)
            print("  auc", metrics.get("auc"), "fpr", len(metrics.get("fpr") or []))
