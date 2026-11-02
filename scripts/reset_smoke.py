import json
import urllib.request

base = "http://127.0.0.1:8765"

def post(url):
    req = urllib.request.Request(url, data=b"{}", method="POST", headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))

lib = json.load(urllib.request.urlopen(base + "/api/library"))
scene_id = lib["scenes"][0]["id"]
opened = json.load(urllib.request.urlopen(base + f"/api/session/{scene_id}"))
cleared = post(base + f"/api/session/{scene_id}/reset")
print("opened", opened["name"], bool(opened.get("last_result")), len(opened.get("results") or []))
print("cleared", cleared["name"], cleared.get("last_result"), cleared.get("results"), "heatmap" in (cleared.get("preview") or {}))
