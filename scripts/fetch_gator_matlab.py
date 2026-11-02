from pathlib import Path
from urllib.request import Request, urlopen

DEST = Path(__file__).resolve().parents[1] / "hadstudio" / "detectors" / "matlab_gator"
(DEST / "util").mkdir(parents=True, exist_ok=True)
BASE = "https://raw.githubusercontent.com/GatorSense/hsi_toolkit/master/"
files = {
    "anomaly_detectors/cbad_anomaly.m": "anomaly_detectors/cbad_anomaly.m",
    "anomaly_detectors/fcbad_anomaly.m": "anomaly_detectors/fcbad_anomaly.m",
    "anomaly_detectors/csd_anomaly.m": "anomaly_detectors/csd_anomaly.m",
    "util/img_det.m": "util/img_det.m",
    "util/pca.m": "util/pca.m",
}
for dest_name, rel in files.items():
    data = urlopen(Request(BASE + rel, headers={"User-Agent": "HADStudio/1.0"}), timeout=60).read()
    path = DEST / dest_name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    print(dest_name, len(data))
