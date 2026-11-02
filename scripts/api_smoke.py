import json
import urllib.request

BASE = "http://127.0.0.1:8765"


def post(path, payload):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.load(resp)


def get(path):
    with urllib.request.urlopen(BASE + path) as resp:
        return resp.read()


catalog = json.loads(get("/api/catalog"))
print("detectors", len(catalog["detectors"]))
demo = post("/api/demo", {})
print("demo", demo["shape"], "png", demo["preview"]["rgb"].startswith("data:image/png"))
prep = post("/api/preprocess", {"session_id": demo["id"], "method": "minmax", "params": {"median_size": 0}})
print("prep", prep["preprocess"])
red = post("/api/reduce", {"session_id": demo["id"], "method": "pca", "n_components": 8})
print("pca", red["shape"])
det = post("/api/detect", {"session_id": demo["id"], "detector": "grx", "params": {}})
print("grx auc", det["last_result"]["metrics"]["auc"], "s", det["last_result"]["elapsed"])
print("heatmap", bool(det["preview"].get("heatmap")))
html = get("/")
print("html", b"HAD Studio" in html)
