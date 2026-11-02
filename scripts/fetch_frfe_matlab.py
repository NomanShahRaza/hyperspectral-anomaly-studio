from pathlib import Path
from urllib.request import Request, urlopen

DEST = Path(__file__).resolve().parents[1] / "hadstudio" / "detectors" / "matlab_frfe"
DEST.mkdir(parents=True, exist_ok=True)
BASE = "https://cdn.jsdelivr.net/gh/xudongzhao461/Hyperspectral-Anomaly-Detection-by-Fractional-Fourier-Entropy@67f6bf059a51a2e6789ca34041b1f2a51ef272c9/"
for name in ("demo.m", "RX.m", "center_standard.m", "FRFEorder.m"):
    data = urlopen(Request(BASE + name, headers={"User-Agent": "HADStudio/1.0"}), timeout=60).read()
    (DEST / name).write_bytes(data)
    print(name, len(data))
