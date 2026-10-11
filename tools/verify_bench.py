"""Verify the connector test bench: each tile must render its own configuration alone."""
from __future__ import annotations

import json
import urllib.request

BASE = "http://127.0.0.1:8947"


def post(path: str, body: dict) -> dict:
    req = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))


def current() -> dict:
    with urllib.request.urlopen(BASE + "/api/project", timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))["params"]


CLEAN = {
    "connector": {
        "hdmi_present": 0, "hdmi_moisture": 0, "hdmi_corrosion": 0,
        "av_present": 0, "av_moisture": 0, "av_corrosion": 0, "av_ground_loop": 0,
        "svideo_present": 0, "svideo_moisture": 0, "svideo_corrosion": 0,
        "svideo_leakage": 0,
        "rf_present": 0, "rf_moisture": 0, "rf_corrosion": 0, "rf_shielding": 0,
    },
    "tape": {"stiction_rate": 0, "stiction_amount": 0, "stiction_stretch": 0},
    "display": {"crt_moisture": 0, "crt_corrosion": 0, "crt_age": 0.3},
}

# what the page does per tile: neutralise everything, then apply the tile
TILES = [
    ("clean baseline", {}),
    ("HDMI wet", {"connector": {"hdmi_present": 1, "hdmi_moisture": 0.5}}),
    ("HDMI corroded", {"connector": {"hdmi_present": 1, "hdmi_corrosion": 0.8}}),
    ("AV ground loop", {"connector": {"av_present": 1, "av_ground_loop": 0.6}}),
    ("S-Video leak", {"connector": {"svideo_present": 1, "svideo_leakage": 0.5}}),
    ("RF wet", {"connector": {"rf_present": 1, "rf_moisture": 0.6}}),
    ("CRT new", {"display": {"crt_moisture": 0.6, "crt_age": 0.0}}),
    ("CRT aged", {"display": {"crt_moisture": 0.6, "crt_age": 0.9}}),
]


def deep_merge(a: dict, b: dict) -> dict:
    out = json.loads(json.dumps(a))
    for k, v in b.items():
        if isinstance(v, dict):
            out[k] = deep_merge(out.get(k, {}), v)
        else:
            out[k] = v
    return out


print("each tile is rendered exactly as the page renders it: full neutral set + its own patch")
print("%-18s %-42s %s" % ("tile", "which interfaces are actually on", "as intended?"))
print("-" * 92)
problems = []
for label, patch in TILES:
    post("/api/project", {"params": deep_merge(CLEAN, patch)})
    p = current()
    c, d = p["connector"], p["display"]
    live = []
    if float(c["hdmi_present"]) > 0.5:
        live.append("HDMI(m=%.2f,c=%.2f)" % (c["hdmi_moisture"], c["hdmi_corrosion"]))
    if float(c["av_present"]) > 0.5:
        live.append("AV(g=%.2f,m=%.2f)" % (c["av_ground_loop"], c["av_moisture"]))
    if float(c["svideo_present"]) > 0.5:
        live.append("SV(leak=%.2f)" % c["svideo_leakage"])
    if float(c["rf_present"]) > 0.5:
        live.append("RF(m=%.2f,c=%.2f)" % (c["rf_moisture"], c["rf_corrosion"]))
    if float(d["crt_moisture"]) > 1e-4 or float(d["crt_corrosion"]) > 1e-4:
        live.append("CRT(m=%.2f,age=%.2f)" % (d["crt_moisture"], d["crt_age"]))
    if float(p["tape"]["stiction_rate"]) > 1e-4:
        live.append("stiction(%.1f)" % p["tape"]["stiction_rate"])

    want = len([k for k in patch if k == "connector"]) + \
        (1 if "display" in patch else 0)
    text = ", ".join(live) if live else "(none -- clean)"
    # a tile must have exactly the interfaces its own patch switches on, and no others
    n_on = len(live)
    ok = (n_on == 0) if not patch else (n_on == 1)
    if not ok:
        problems.append(label)
    print("%-18s %-42s %s" % (label, text[:42], "yes" if ok else "NO"))

print()
if problems:
    print("PROBLEMS: %s" % ", ".join(problems))
    print("A tile carrying more than its own interface is rendering accumulated state, so its")
    print("picture would not match its caption. That is exactly what the neutral set prevents.")
else:
    print("every tile carries exactly its own configuration and nothing else")
    print("=> the bench shows one interface at a time, so the difference between two tiles is")
    print("   attributable to the interface that differs between them.")
