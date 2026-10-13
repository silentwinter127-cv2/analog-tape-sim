"""
Render the effect on the closest-framed raw source available, and state what remains.

**Where the gap actually is.** Measured: the reference's lamp is a large multiple-bulb fixture covering about **18%** of
its frame; the raw photographs available have saturated regions from **0.05% to 0.62%**, and cropping the one whose
light is nearest the camera raises it to **3.02%** -- five times better than the best full frame, and still six times
short of the reference. **The mechanism is not the limit**: `source_glow` is calibrated so that on a realistic source
it reaches a halo of **0.53** where the reference's own profile needs **0.51**.

**So this render uses the best subject available and reports the comparison honestly**, rather than presenting it as a
match. It is the same chain as the accepted reproduction -- raw photograph, digital-component path, linear display, the
band off, the mechanisms at a visible strength -- with `source_glow` added at the calibrated value.

The crop is a field-of-view choice on a raw photograph, recorded with its own provenance and its own measurements, so
what was done to the input is stated rather than implied.
"""
from __future__ import annotations

import io
import json
import os
import urllib.request

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "tools", "ntscrs_ref")
SRC = os.path.join(OUT, "raw_light_close.jpg")
BASE = "http://127.0.0.1:8941"
COEF = np.array([0.299, 0.587, 0.114], np.float32)


def get_raw(path, timeout=180):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return r.read()


def post(path, obj, timeout=900):
    req = urllib.request.Request(BASE + path, data=json.dumps(obj).encode(),
                                headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def upload_raw(path):
    with open(path, "rb") as fh:
        data = fh.read()
    req = urllib.request.Request(
        BASE + "/api/upload", data=data, method="POST",
        headers={"Content-Type": "application/octet-stream",
                 "X-Filename": os.path.basename(path),
                 "Content-Length": str(len(data))})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


print("   uploading the crop of the raw photograph")
print("      %s" % {k: v for k, v in upload_raw(SRC).items() if k in ("ok", "still")})

COMMON = {
    "format": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0, "chroma_gain": 1.0,
               "color_noise": 0.0, "cross_color": 0.0, "cross_luma": 0.0, "edge_boost": 0.0},
    "player": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0,
               "color_noise": 0.0, "cross_color": 0.0, "cross_luma": 0.0},
    "codec": {"dv_enabled": 1.0},
    "tape": {"head_switching": 0.0},
    "display": {"enabled": 1.0, "gamma": 1.0, "saturation": 0.62},
}

EFFECT = {
    # the light mechanisms, per-unit-area and so unchanged by the frame's size
    "source_glow": 0.85,     # the scattered halo carrying the source's own colour -- calibrated against the reference
    "halation": 0.85, "ccd_smear": 0.80, "flare_defocus": 1.00, "lens_flare": 0.50,
    "spike_spread": 0.85, "overflow_drain": 0.35,
    # the aberrations, radial and in pixels, so scaled to this frame
    "chroma_aberration": 0.20, "longitudinal_ca": 0.12, "axis_offset": 0.20,
}


def apply(camera):
    params = {k: dict(v) for k, v in COMMON.items()}
    base = {"exposure": 0.0, "color_cast": 0.0, "halation": 0.0, "ccd_smear": 0.0,
            "chroma_aberration": 0.0, "lens_flare": 0.0, "spike_spread": 0.0,
            "flare_defocus": 0.0, "longitudinal_ca": 0.0, "axis_offset": 0.0,
            "overflow_drain": 0.0, "source_glow": 0.0, "lens_fog": 0.0, "lens_wear": 0.0,
            "bloom_amount": 0.0, "dead_sites": 0.0}
    params["camera"] = dict(base, **camera)
    post("/api/project", {"reset_params": True})
    post("/api/project", {"params": params, "seed": 20240101, "standard": "ntsc"})


def render(name, scale=0.6):
    with urllib.request.urlopen(BASE + "/api/frame?t=0.0&scale=%s&q=96" % scale, timeout=900) as r:
        im = Image.open(io.BytesIO(r.read())).convert("RGB")
    im.save(os.path.join(OUT, name))
    x = np.asarray(im, np.float32) / 255.0
    L = x @ COEF
    return im, dict(mean=round(float(L.mean()), 4), blown=round(float((L > 0.97).mean()), 5),
                    glow=round(float((L > 0.85).mean()), 5),
                    warm=round(float((x[..., 0] - x[..., 2]).mean()), 5))


apply({})
im_c, m_c = render("close-control.png")
print()
print("   control (mechanisms zero)  %s" % m_c)
apply(EFFECT)
im_e, m_e = render("close-effect.png")
print("   effect                     %s" % m_e)
print("   the effect against its control:")
for k in ("mean", "blown", "glow", "warm"):
    print("      %-6s %.5f -> %.5f  (%+.5f)" % (k, m_c[k], m_e[k], m_e[k] - m_c[k]))

# the profile below the core, against the reference's, which is the number that says whether it matches
def prof(img, rows=(0, 40, 60, 100, 160, 180)):
    H, W = img.shape[:2]
    lum = img @ COEF
    k = max(5, H // 40)
    box = np.lib.stride_tricks.sliding_window_view(lum, (k, k)).mean(axis=(2, 3))
    cy, cx = np.unravel_index(int(np.argmax(box)), box.shape)
    cy += k // 2
    cx += k // 2
    sc = H / 900.0
    out = []
    for d in rows:
        dd = int(d * sc)
        r0, r1 = min(H - 1, cy + dd), min(H, cy + dd + max(3, int(20 * sc)))
        c0, c1 = max(0, cx - max(3, int(6 * sc))), min(W, cx + max(4, int(7 * sc)))
        if r1 <= r0 or c1 <= c0:
            out.append(float("nan"))
            continue
        p = img[r0:r1, c0:c1].reshape(-1, 3).mean(axis=0)
        out.append(float(p[2] - p[0]))
    return out


print()
print("   the profile below the core, B-R:")
print("      %-12s %s" % ("the reference", "  ".join("+%-6d" % d for d in (0, 40, 60, 100, 160, 180))))
print("      %-12s %s" % ("", "  ".join("%+.3f" % v for v in
                                      [0.000, -0.009, -0.170, -0.510, 0.178, 0.226])))
for tag, name in (("control", "close-control.png"), ("effect", "close-effect.png")):
    b = np.asarray(Image.open(os.path.join(OUT, name)).convert("RGB"), np.float32) / 255.0
    print("      %-12s %s" % (tag, "  ".join("%+.3f" % v for v in prof(b))))

apply(EFFECT)
cfg = json.loads(get_raw("/api/export"))
with open(os.path.join(OUT, "repro-highlight.json"), "w", encoding="utf-8") as fh:
    json.dump(cfg, fh, ensure_ascii=False, indent=2)
print()
print("   wrote repro-highlight.json: %d parameter(s) across %d layer(s)"
      % (cfg["param_count"], len(cfg["params"])))
