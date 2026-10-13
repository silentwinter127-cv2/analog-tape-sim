"""
Quantify how far a clean input drifts under the model's neutral settings, and attribute the drift.

The claim to be tested is specific: **a clean, correctly exposed picture passing through the chain at neutral
settings should come out close to itself.** It does not -- a sodium-lit snow scene comes out with magenta and green
banding -- and until that is fixed, no highlight effect can be validated, because the artefacts are larger than the
effect being looked for.

Two numbers are used, and they answer different questions:

  * **colour drift**, the difference between the output's mean channel values and the source's, which says the
    processing changed the picture's colour globally;
  * **structure added**, the local detail the output has that the source does not, which says it *invented*
    something -- banding, noise, crosstalk -- rather than merely shifting what was there.

The source is downsampled to a fixed small size before both are measured, so the comparison is between the same
pixels and the render's resolution cannot flatter it.

Then every format and player parameter is swept **one at a time to zero**, which is what attributes the drift.
"""
from __future__ import annotations

import io
import json
import os
import urllib.request

import numpy as np
from PIL import Image

BASE = "http://127.0.0.1:8941"
COEF = np.array([0.299, 0.587, 0.114], np.float32)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "tools", "ntscrs_ref")
SMALL = (120, 90)


def get(path, timeout=180):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return json.loads(r.read())


def post(path, obj, timeout=900):
    req = urllib.request.Request(BASE + path, data=json.dumps(obj).encode(),
                                headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


_uploaded = False


def ensure_source():
    global _uploaded
    if _uploaded:
        return
    import uuid
    path = os.path.join(OUT, "clean_source.jpg")
    boundary = "----atsim" + uuid.uuid4().hex
    with open(path, "rb") as fh:
        data = fh.read()
    body = b"".join([
        ("--%s\r\n" % boundary).encode(),
        ('Content-Disposition: form-data; name="file"; filename="clean_source.jpg"\r\n').encode(),
        b"Content-Type: image/jpeg\r\n\r\n", data, b"\r\n",
        ("--%s--\r\n" % boundary).encode(),
    ])
    req = urllib.request.Request(BASE + "/api/upload", data=body,
                                headers={"Content-Type": "multipart/form-data; boundary=" + boundary},
                                method="POST")
    with urllib.request.urlopen(req, timeout=900) as r:
        json.loads(r.read())
    _uploaded = True


def frame(scale=0.30):
    with urllib.request.urlopen(BASE + "/api/frame?t=0.0&scale=%s&q=95" % scale, timeout=900) as r:
        return np.asarray(Image.open(io.BytesIO(r.read())).convert("RGB"), np.float32) / 255.0


def shrink(img):
    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).resize(SMALL, Image.BILINEAR)
    return np.asarray(im, np.float32) / 255.0


ensure_source()
KEEP = json.loads(json.dumps(get("/api/project").get("params") or {}))

src_full = np.asarray(Image.open(os.path.join(OUT, "clean_source.jpg")).convert("RGB"),
                      np.float32) / 255.0
SRC = shrink(src_full)
print("   the source, at the comparison size (%dx%d):" % SMALL)
print("      mean RGB %s   R-B %+.4f   local detail %.5f"
      % (np.round(SRC.reshape(-1, 3).mean(axis=0), 4),
         float(SRC[..., 0].mean() - SRC[..., 2].mean()),
         float(np.abs(np.diff(SRC @ COEF, axis=1)).mean())))

# The most neutral settings the model can be put in: every mechanism the schema offers that could act on a clean
# picture is switched off, so what is left is the signal path itself.
NEUTRAL = {
    "camera": {"exposure": 0.0, "color_cast": 0.0, "halation": 0.0, "ccd_smear": 0.0,
               "flare_defocus": 0.0, "lens_flare": 0.0, "spike_spread": 0.0,
               "chroma_aberration": 0.0, "longitudinal_ca": 0.0, "axis_offset": 0.0,
               "overflow_drain": 0.0, "field_curvature": 0.0, "coma": 0.0, "astigmatism": 0.0,
               "spherical_aberration": 0.0, "lens_distortion": 0.0, "lens_wear": 0.0,
               "lens_fog": 0.0, "bloom_amount": 0.0, "sharpness": 1.0, "defocus": 0.0},
    "format": {"chroma_bandwidth": 3.0, "luma_bandwidth": 6.0, "chroma_gain": 1.0,
               "color_noise": 0.0, "cross_color": 0.0, "cross_luma": 0.0, "comb_depth": 0.0,
               "edge_boost": 0.0, "chroma_vertical_averaging": 0.0},
    "player": {"chroma_bandwidth": 3.0, "luma_bandwidth": 6.0, "color_noise": 0.0,
               "cross_color": 0.0, "cross_luma": 0.0, "comb_depth": 0.0},
    "codec": {"dv_enabled": 0.0, "chroma_mode": 0.0},
    "tape": {"chroma_loss": 0.0, "luma_dropout": 0.0, "head_switch": 0.0},
    "display": {"enabled": 0.0},
    "connector": {},
    "rf": {},
}


def apply(extra=None):
    post("/api/project", {"params": KEEP})
    merged = {k: dict((KEEP.get(k) or {}), **v) for k, v in NEUTRAL.items()}
    if extra:
        for k, v in extra.items():
            merged[k] = dict(merged.get(k) or {}, **v)
    post("/api/project", {"params": merged})


def measure(img, label):
    s = shrink(img)
    src_mean = SRC.reshape(-1, 3).mean(axis=0)
    out_mean = s.reshape(-1, 3).mean(axis=0)
    drift = float(np.abs(out_mean - src_mean).max())
    added = float(np.abs(np.diff(s @ COEF, axis=1)).mean()) - float(np.abs(np.diff(SRC @ COEF, axis=1)).mean())
    hue = float(s[..., 0].mean() - s[..., 2].mean())
    print("   %-38s colour drift %.5f  structure added %+.5f  R-B %+.4f"
          % (label, drift, added, hue))
    return drift, added


print()
print("=== the whole chain at its most neutral ===")
apply()
d0, a0 = measure(frame(), "all artefacts off")

print()
print("=== and with the project's own defaults, which is what a preset gives ===")
post("/api/project", {"params": KEEP})
d1, a1 = measure(frame(), "the project's defaults")

print()
print("=== sweeping single parameters away from the defaults ===")
print("   (each row turns ONE parameter off, so the change is attributable)")
cands = []
for layer in ("format", "player", "tape", "codec"):
    cur = KEEP.get(layer) or {}
    for k, v in cur.items():
        if isinstance(v, (int, float)) and abs(float(v)) > 1e-6:
            cands.append((layer, k, float(v)))

results = []
for layer, k, v in cands:
    post("/api/project", {"params": KEEP})
    post("/api/project", {"params": {layer: {k: 0.0}}})
    img = frame()
    s = shrink(img)
    src_mean = SRC.reshape(-1, 3).mean(axis=0)
    drift = float(np.abs(s.reshape(-1, 3).mean(axis=0) - src_mean).max())
    added = float(np.abs(np.diff(s @ COEF, axis=1)).mean()) - float(np.abs(np.diff(SRC @ COEF, axis=1)).mean())
    results.append((added, drift, layer, k, v))

results.sort()
print("   %-10s %-26s %-9s %-14s %s" % ("layer", "parameter", "default", "structure added", "colour drift"))
for added, drift, layer, k, v in results[:12]:
    print("   %-10s %-26s %-9.3f %+-14.5f %.5f" % (layer, k, v, added, drift))

post("/api/project", {"params": KEEP})
print()
print("   the chain at its most neutral adds %.5f of structure; the defaults add %.5f" % (a0, a1))
print("   parameters restored")
