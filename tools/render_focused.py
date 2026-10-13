"""
One focused render: the warm glow and the cool vertical trail, with everything else switched off.

The previous attempt enabled seven mechanisms at once -- lateral and longitudinal chromatic aberration, per-channel
flare defocus, spike separation, smear, halation and overflow drain -- and the result was a blurred, magenta-shifted
frame in which the picture itself had been dragged out of shape. **That is not the effect the reference shows**, and
stacking mechanisms cannot demonstrate any one of them.

The reference photograph is specific: a **warm blown lamp** and a **cool vertical trail** beneath it. Two mechanisms
produce those and nothing else in the model is needed for them:

  * `halation`, the sensor's bloom, is the warm glow -- it is an imager effect around an overexposed area;
  * `ccd_smear`, charge along the readout column, is the vertical trail;
  * and `flare_defocus`, which defocuses the flare's core per channel with blue widest, is what makes the trail
    **cooler than the lamp** rather than a neutral smear.

Everything else is left at zero, including both chromatic aberrations, so the frame keeps its own sharpness and the
two effects can actually be seen.
"""
from __future__ import annotations

import io
import json
import os
import urllib.request
import uuid

import numpy as np
from PIL import Image

BASE = "http://127.0.0.1:8941"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "tools", "ntscrs_ref")
SRC = os.path.join(OUT, "clean_source.jpg")
COEF = np.array([0.299, 0.587, 0.114], np.float32)


def get(path, timeout=180):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return json.loads(r.read())


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


def frame(scale=0.45):
    with urllib.request.urlopen(BASE + "/api/frame?t=0.0&scale=%s&q=96" % scale, timeout=900) as r:
        return np.asarray(Image.open(io.BytesIO(r.read())).convert("RGB"), np.float32) / 255.0


print("   uploading the clean source")
upload(SRC)
KEEP = json.loads(json.dumps(get("/api/project").get("params") or {}))

# The display is switched off entirely for this pair.  It is not what the reference is about, and leaving it on
# adds the tube's own blur and mask to a comparison that is meant to show two mechanisms.
NEUTRAL = {
    "camera": {"exposure": 0.0, "color_cast": 0.0,
               # only these three, everything else at zero
               "halation": 0.0, "ccd_smear": 0.0, "flare_defocus": 0.0, "lens_flare": 0.0,
               "spike_spread": 0.0, "chroma_aberration": 0.0, "longitudinal_ca": 0.0,
               "axis_offset": 0.0, "overflow_drain": 0.0, "field_curvature": 0.0,
               "coma": 0.0, "astigmatism": 0.0, "spherical_aberration": 0.0,
               "lens_distortion": 0.0, "lens_wear": 0.0, "lens_fog": 0.0},
    "format": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0, "chroma_gain": 1.0},
    "player": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0},
    "codec": {"dv_enabled": 0.0},
    "display": {"enabled": 0.0},
}

FOCUSED = json.loads(json.dumps(NEUTRAL))
FOCUSED["camera"].update({
    "halation": 0.55,        # the sensor's bloom: the warm glow around the overexposed lamp
    "ccd_smear": 0.45,       # charge along the readout column: the vertical trail
    "flare_defocus": 0.85,   # and per-channel defocus, so the trail is cooler than the lamp
    "lens_flare": 0.15,      # a touch of the ghost, which the reference also shows
})


def apply(params):
    post("/api/project", {"params": KEEP})
    post("/api/project", {"params": {k: dict((KEEP.get(k) or {}), **v) for k, v in params.items()}})


def report(img, label):
    lum = img @ COEF
    print("   %-8s %dx%d  mean %.4f  blown %.5f  dark %.4f  R-B %+.4f  sharpness %.5f"
          % (label, img.shape[1], img.shape[0], float(lum.mean()),
             float((lum > 0.97).mean()), float((lum < 0.02).mean()),
             float(img[..., 0].mean() - img[..., 2].mean()),
             float(np.abs(np.diff(lum, axis=1)).mean())))


print()
apply(NEUTRAL)
a = frame()
report(a, "neutral")
Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).save(os.path.join(OUT, "fx_before.png"))

apply(FOCUSED)
b = frame()
report(b, "focused")
Image.fromarray((np.clip(b, 0, 1) * 255).astype(np.uint8)).save(os.path.join(OUT, "fx_after.png"))

# a strip through the brightest lamp, full width, so the glow and the trail are visible at a glance
lum = a @ COEF
h, w = lum.shape
cy, cx = np.unravel_index(int(np.argmax(lum)), lum.shape)
r = 340
y0, y1 = max(0, cy - r), min(h, cy + r)
x0, x1 = max(0, cx - r), min(w, cx + r)
for tag, img in (("before", a), ("after", b)):
    crop = img[y0:y1, x0:x1]
    im = Image.fromarray((np.clip(crop, 0, 1) * 255).astype(np.uint8))
    im.save(os.path.join(OUT, "fxcrop_%s.png" % tag))
    print("   wrote fxcrop_%s.png (%dx%d) at the brightest lamp" % (tag, im.width, im.height))

post("/api/project", {"params": KEEP})
print("   wrote fx_before.png and fx_after.png")
