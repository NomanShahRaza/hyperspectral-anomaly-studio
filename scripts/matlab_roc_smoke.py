import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hadstudio.detectors.registry import run_detector
from hadstudio.library import find_scene, load_record_arrays
from hadstudio.metrics import evaluate_detection

loaded = load_record_arrays(find_scene("abu-airport-1"))
cube, gt = loaded["cube"], loaded["gt"]
for det, expected in (("drx", 0.8456), ("dwtrx", 0.8645)):
    score, _ = run_detector(det, cube)
    metrics = evaluate_detection(score, gt, detector_id=det)
    auc = metrics["auc"]
    print(det, f"{auc:.4f}", "table", expected, "n", metrics.get("n_thresholds"), "roc", metrics["roc"])
    assert metrics["roc"] == "matlab-demo"
    assert abs(auc - expected) < 5e-4, (det, auc, expected)
score, _ = run_detector("grx", cube)
metrics = evaluate_detection(score, gt, detector_id="grx")
print("grx", f"{metrics['auc']:.4f}", "roc", metrics["roc"])
assert metrics["roc"] == "sklearn"
print("ok")
