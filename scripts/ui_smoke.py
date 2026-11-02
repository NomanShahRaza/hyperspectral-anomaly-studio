import json
import urllib.request

base = "http://127.0.0.1:8765"
html = urllib.request.urlopen(base + "/").read().decode("utf-8", "replace")
for needle in (
    "Research catalogue",
    "Use detector",
    "Method details",
    "loadedDatasets",
    "modelFamilyFilter",
    'option value="pca"',
):
    print(needle, needle in html)
js = urllib.request.urlopen(base + "/assets/app.js?v=9").read().decode()
print("js_ok", "function renderModels" in js and "Import scene" in js)
cat = json.load(urllib.request.urlopen(base + "/api/catalog"))
print("datasets", len(cat["datasets"]), cat["datasets"][0]["name"], cat["datasets"][0].get("collection"))
print("detectors", len(cat["detectors"]), "desc", bool(cat["detectors"][0].get("description")), "ready", cat["detectors"][0].get("ready"))
print("families", sorted({d["family"] for d in cat["detectors"]}))
req = urllib.request.Request(base + "/api/demo", data=b"{}", method="POST", headers={"Content-Type": "application/json"})
demo = json.load(urllib.request.urlopen(req))
print("demo", demo["name"], demo["shape"])
sess = json.load(urllib.request.urlopen(base + "/api/session/" + demo["id"]))
print("session", sess["name"], sess["has_gt"])
