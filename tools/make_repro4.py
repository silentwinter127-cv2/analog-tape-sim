"""
Build the reproduction for real: a clean modern source, the display linear, and the effect made visible.

**What is kept from the investigation.** The display's gamma accounts for more than half the drift on a clean source
(0.264 default against 0.110 at gamma 1.0), so it is set to **1.0** -- measured, not chosen. The chroma path is left
wide because the fault found earlier in it is fixed. And the banding is a **known, unresolved artefact of the chroma
path on a real photograph**; it is not hidden here, and the render is presented with that stated.

**What is changed from the last attempt.** The previous reproduction was too weak to see: the glow and the trail were
there in the numbers and not in the picture. The mechanisms are pushed up together, and the exposure is pulled **down**
rather than up, because the chain was already brightening the picture by 45% on its own (mean 0.85 against the
source's 0.59) and the blown area by three times.

**A control is rendered on the identical path with every mechanism at zero**, because a reproduction that cannot be
compared against its own baseline cannot be judged -- and because the last several attempts failed for exactly that
reason.

The settings file comes from the export endpoint, so it is exactly what a second person needs.
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
COEF = np.array([0.299, 0.587, 0.114], np.float32)


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
        ("--%s\r\n" % boundary).encode(),
    ])
    # the multipart terminator
    body = body[:-len(("--%s\r\n" % boundary).encode())] + ("--%s--\r\n" % boundary).encode()
    req = urllib.request.Request(BASE + "/api/upload", data=body,
                                headers={"Content-Type": "multipart/form-data; boundary=" + boundary},
                                method="POST")
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


print("   uploading the clean CC0 source")
print("   %s" % {k: v for k, v in upload(SRC).items() if k in ("ok", "width", "height")})

# The display is linear: gamma 1.0, measured as the single largest contributor to the drift on a clean source.
# The chroma path is left wide and quiet, since the fault found in it earlier is fixed and the crosstalk terms would
# inject chroma into a scene that has none.
COMMON = {
    "format": {"chroma_bandwidth": 1.6, "luma_bandwidth": 4.5, "chroma_gain": 1.0,
               "color_noise": 0.0, "cross_color": 0.0, "cross_luma": 0.0, "edge_boost": 0.0},
    "player": {"chroma_bandwidth": 1.6, "luma_bandwidth": 4.5,
               "color_noise": 0.0, "cross_color": 0.0, "cross_luma": 0.0},
    "codec": {"dv_enabled": 0.0},
    "display": {"enabled": 1.0, "gamma": 1.0, "saturation": 0.60},
}

# Exposure is pulled DOWN because the chain brightens the picture by 45% on its own.
EFFECT_CAMERA = {
    "exposure": -0.45,
    "halation": 0.85,
    "ccd_smear": 0.85,
    "flare_defocus": 1.00,
    "lens_flare": 0.55,
    "spike_spread": 0.85,
    "chroma_aberration": 0.85,
    "axis_offset": 0.45,
    "longitudinal_ca": 0.55,
    "overflow_drain": 0.35,
}
CONTROL_CAMERA = {
    "exposure": -0.45,
    "halation": 0.0, "ccd_smear": 0.0, "flare_defocus": 0.0, "lens_flare": 0.0,
    "spike_spread": 0.0, "chroma_aberration": 0.0, "axis_offset": 0.0,
    "longitudinal_ca": 0.0, "overflow_drain": 0.0,
}


def render(camera, name, scale=0.55):
    params = {k: dict(v) for k, v in COMMON.items()}
    params["camera"] = dict(camera)
    post("/api/project", {"reset_params": True})
    post("/api/project", {"params": params, "seed": 20240101, "standard": "ntsc"})
    with urllib.request.urlopen(BASE + "/api/frame?t=0.0&scale=%s&q=96" % scale, timeout=900) as r:
        im = Image.open(io.BytesIO(r.read())).convert("RGB")
    im.save(os.path.join(OUT, name))
    a = np.asarray(im, np.float32) / 255.0
    lum = a @ COEF
    row_rb = (a[..., 0] - a[..., 2]).mean(axis=1)
    return im, dict(mean=[round(float(v), 4) for v in a.reshape(-1, 3).mean(axis=0)],
                    rb=round(float(a[..., 0].mean() - a[..., 2].mean()), 4),
                    blown=round(float((lum > 0.97).mean()), 4),
                    banding=round(float(np.abs(np.diff(row_rb)).mean()), 5))


print()
im_c, m_c = render(CONTROL_CAMERA, "repro2-control.png")
print("   control (mechanisms at zero)  %s" % m_c)
im_e, m_e = render(EFFECT_CAMERA, "repro2-effect.png")
print("   effect                        %s" % m_e)

src = np.asarray(Image.open(SRC).convert("RGB"), np.float32) / 255.0
sl = src @ COEF
print("   the source                    mean %s  rb %+.4f  blown %.4f  banding %.5f"
      % ([round(float(v), 4) for v in src.reshape(-1, 3).mean(axis=0)],
         float(src[..., 0].mean() - src[..., 2].mean()),
         float((sl > 0.97).mean()),
         float(np.abs(np.diff((src[..., 0] - src[..., 2]).mean(axis=1))).mean())))

# the settings file, straight from the export endpoint
post("/api/project", {"reset_params": True})
params = {k: dict(v) for k, v in COMMON.items()}
params["camera"] = dict(EFFECT_CAMERA)
post("/api/project", {"params": params, "seed": 20240101, "standard": "ntsc"})
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
