"""
Final pass on the reproduction: stop the chroma injection, and render the effect against its own control.

The banding is not from the highlight mechanisms. The source is a **smooth single-colour** sodium-lit scene, so
horizontal magenta and green stripes can only come from a stage that **injects chroma that was not there** -- and the
candidates are all in the composite separation: `cross_color` and `cross_luma` (crosstalk between luma and chroma),
`color_noise`, and the chroma averaging. They are set to zero here, which is the honest setting for an input that has
no chroma detail to misplace.

**A control is rendered beside it** -- the identical path with every highlight mechanism at zero -- because a
reproduction that cannot be compared against its own baseline cannot be judged, and because the last several
attempts all failed for the same reason: effects were being tuned while a larger artefact sat underneath them,
unmeasured.

Both are rendered at 0.5 scale with the same seed, and both are written to disk with the settings file.
"""
from __future__ import annotations

import io
import json
import os
import urllib.request
import uuid

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "tools", "ntscrs_ref")
SRC = os.path.join(OUT, "clean_source.jpg")
BASE = "http://127.0.0.1:8941"


def get(path, timeout=180):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return json.loads(r.read())


def get_raw(path, timeout=180):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return r.read()


def post(path, obj, timeout=900):
    req = urllib.request.Request(BASE + path, data=json.dumps(obj).encode(),
                                headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def upload(path):
    boundary = "----atsim" + uuid.uuid4().hex
    with open(path, "rb") as fh:
        data = fh.read()
    body = b"".join([
        ("--%s\r\n" % boundary).encode(),
        ('Content-Disposition: form-data; name="file"; filename="%s"\r\n'
         % os.path.basename(path)).encode(),
        b"Content-Type: image/jpeg\r\n\r\n", data, b"\r\n",
        ("--%s--\r\n" % boundary).encode(),
    ])
    req = urllib.request.Request(BASE + "/api/upload", data=body,
                                headers={"Content-Type": "multipart/form-data; boundary=" + boundary},
                                method="POST")
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


upload(SRC)

# **The chroma injection is off.**  An input with no chroma detail cannot have chroma misplaced in it, so the
# crosstalk terms and the chroma noise belong at zero for this subject; leaving them at their defaults is what put
# the magenta and green stripes over a single-colour scene.
QUIET = {
    "format": {"color_noise": 0.0, "cross_color": 0.0, "cross_luma": 0.0, "edge_boost": 0.0,
               "comb_depth": 0.0, "chroma_vertical_averaging": 0.0,
               "chroma_bandwidth": 1.6, "luma_bandwidth": 4.5, "chroma_gain": 1.0},
    "player": {"color_noise": 0.0, "cross_color": 0.0, "cross_luma": 0.0, "comb_depth": 0.0,
               "chroma_vertical_averaging": 0.0, "chroma_bandwidth": 1.6, "luma_bandwidth": 4.5},
    "codec": {"dv_enabled": 0.0},
    "display": {"enabled": 1.0, "gamma": 1.0, "saturation": 0.60},
}

EFFECT = {k: dict(v) for k, v in QUIET.items()}
EFFECT["camera"] = {
    "halation": 0.50, "ccd_smear": 0.45, "flare_defocus": 0.85, "lens_flare": 0.22,
    "spike_spread": 0.65, "chroma_aberration": 0.60, "axis_offset": 0.35,
    "longitudinal_ca": 0.30, "overflow_drain": 0.18, "exposure": 0.0,
}
CONTROL = {k: dict(v) for k, v in QUIET.items()}
CONTROL["camera"] = {"exposure": 0.0, "halation": 0.0, "ccd_smear": 0.0, "flare_defocus": 0.0,
                     "lens_flare": 0.0, "spike_spread": 0.0, "chroma_aberration": 0.0,
                     "axis_offset": 0.0, "longitudinal_ca": 0.0, "overflow_drain": 0.0}


def render(params, name):
    post("/api/project", {"reset_params": True})
    post("/api/project", {"params": params, "seed": 20240101, "standard": "ntsc"})
    with urllib.request.urlopen(BASE + "/api/frame?t=0.0&scale=0.5&q=95", timeout=900) as r:
        im = Image.open(io.BytesIO(r.read())).convert("RGB")
    im.save(os.path.join(OUT, name))
    a = np.asarray(im, np.float32) / 255.0
    coef = np.array([0.299, 0.587, 0.114], np.float32)
    # the banding, as the amount of chroma that varies from row to row where the source does not
    row_rb = (a[..., 0] - a[..., 2]).mean(axis=1)
    print("   %-10s mean %s  R-B %+.4f  blown %.4f  row-to-row R-B variation %.5f"
          % (name.replace("repro-", "").replace(".png", ""),
             np.round(a.reshape(-1, 3).mean(axis=0), 4),
             float(a[..., 0].mean() - a[..., 2].mean()),
             float((a.max(axis=2) > 0.97).mean()),
             float(np.abs(np.diff(row_rb)).mean())))
    return a


print("   %s" % ("control and effect, same path, same seed"))
render(CONTROL, "repro-control.png")
render(EFFECT, "repro-effect_render.png")

# the settings file for the effect
post("/api/project", {"reset_params": True})
post("/api/project", {"params": EFFECT, "seed": 20240101, "standard": "ntsc"})
cfg = json.loads(get_raw("/api/export"))
out = os.path.join(OUT, "repro-highlight.json")
with open(out, "w", encoding="utf-8") as fh:
    json.dump(cfg, fh, ensure_ascii=False, indent=2)
print()
print("   wrote %s: %d parameter(s) across %d layer(s)"
      % (os.path.relpath(out, ROOT), cfg["param_count"], len(cfg["params"])))
for layer, vals in sorted(cfg["params"].items()):
    print("      %-9s %s" % (layer, {k: (round(v, 3) if isinstance(v, float) else v)
                                     for k, v in sorted(vals.items())}))
