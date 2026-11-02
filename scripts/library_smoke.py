import json
import urllib.request
from pathlib import Path

lib = json.load(urllib.request.urlopen("http://127.0.0.1:8765/api/library"))
print("scenes", len(lib["scenes"]))
for scene in lib["scenes"]:
    print(scene["id"], scene["name"], scene["origin"], scene["shape"])

root = Path(__file__).resolve().parents[1]
print("runs", len(list((root / "runs").glob("*"))))
print("previews", len(list((root / "library" / "previews").glob("*.png"))))

if lib["scenes"]:
    first = lib["scenes"][0]["id"]
    opened = json.load(urllib.request.urlopen(f"http://127.0.0.1:8765/api/session/{first}"))
    print("opened", opened["name"], opened["shape"], opened["has_gt"])
