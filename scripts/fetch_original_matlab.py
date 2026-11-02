"""Download original MATLAB sources used by HAD Studio."""

from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
UA = {"User-Agent": "HADStudio/1.0"}


def _get(url: str) -> bytes:
    return urlopen(Request(url, headers=UA), timeout=90).read()


def _write(dest: Path, data: bytes) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    print(dest.relative_to(ROOT), len(data))


files = {
    ROOT / "hadstudio/detectors/matlab_gator/anomaly_detectors/csd_anomaly.m": (
        "https://raw.githubusercontent.com/GatorSense/hsi_toolkit/master/anomaly_detectors/csd_anomaly.m"
    ),
    ROOT / "hadstudio/detectors/matlab_frfe/Disfrft.m": (
        "http://nalag.cs.kuleuven.be/research/software/FRFT/Disfrft.m"
    ),
    ROOT / "hadstudio/detectors/matlab_lsunrs/fun_LSUNRS_ED_ORAD.m": (
        "https://cdn.jsdelivr.net/gh/zephyrhours/Hyperspectral-Anomaly-Detection-LSUNRSORAD-and-LSAD-CR-IDW-@master/fun_LSUNRS_ED_ORAD.m"
    ),
    ROOT / "hadstudio/detectors/matlab_lsunrs/fun_LSAD_CR_IDW.m": (
        "https://cdn.jsdelivr.net/gh/zephyrhours/Hyperspectral-Anomaly-Detection-LSUNRSORAD-and-LSAD-CR-IDW-@master/fun_LSAD_CR_IDW.m"
    ),
    ROOT / "hadstudio/detectors/matlab_lad/lad_C.m": (
        "https://raw.githubusercontent.com/fverdoja/LAD-Laplacian-Anomaly-Detector/master/lad_C.m"
    ),
    ROOT / "hadstudio/detectors/matlab_lad/lad_C_S.m": (
        "https://raw.githubusercontent.com/fverdoja/LAD-Laplacian-Anomaly-Detector/master/lad_C_S.m"
    ),
    ROOT / "hadstudio/detectors/matlab_lad/lad_Q.m": (
        "https://raw.githubusercontent.com/fverdoja/LAD-Laplacian-Anomaly-Detector/master/lad_Q.m"
    ),
    ROOT / "hadstudio/detectors/matlab_lad/lad_Q_S.m": (
        "https://raw.githubusercontent.com/fverdoja/LAD-Laplacian-Anomaly-Detector/master/lad_Q_S.m"
    ),
    ROOT / "hadstudio/detectors/matlab_lad/rsad.m": (
        "https://raw.githubusercontent.com/fverdoja/LAD-Laplacian-Anomaly-Detector/master/rsad.m"
    ),
}

for dest, url in files.items():
    _write(dest, _get(url))
