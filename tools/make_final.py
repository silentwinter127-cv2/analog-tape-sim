"""
Final render: a night frame with a genuinely blown lamp, the band off, and the effect against its own control.

**Why this source and not the last one.** The sodium-lit snow scene has **no blown pixels at all** (`blown 0.0000`),
and every mechanism under test is a **highlight** mechanism -- halation is the bloom of a saturated photosite, the
smear is charge spilling from one, the ring is the anti-blooming drain pulling its neighbours down. On a frame with
nothing saturated they are all correctly inert, which is why the effect differed from its control by 0.011 in R-B and
0.0001 in blown area: **there was nothing for them to act on.**

This frame, from the DSC-W210 clip supplied earlier, has a genuinely blown street lamp on a dark background --
`blown 0.005`, `glow 0.010`, mean luma 0.047 -- which is both what the effect needs and what the reference
photographs show.

**The band is off** because head switching is a helical-scan artefact, intrinsic to VHS and absent from a
digital-component camera, and this material is from one.

**And the upload path is now correct**: `/api/upload` reads the raw body with the name in `X-Filename`, not
multipart, so every earlier upload in this series stored the multipart envelope as the file and the picture arrived
only because a JPEG decoder skips to the first valid header. That is recorded as its own defect.
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


COMMON = {
    "camera": {"exposure": 0.0, "color_cast": 0.0, "halation": 0.0, "ccd_smear": 0.0,
               "chroma_aberration": 0.0, "lens_flare": 0.0, "spike_spread": 0.0,
               "flare_defocus": 0.0, "longitudinal_ca": 0.0, "axis_offset": 0.0,
               "overflow_drain": 0.0, "lens_fog": 0.0, "lens_wear": 0.0,
               "bloom_amount": 0.0, "dead_sites": 0.0},
    "format": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0, "chroma_gain": 1.0,
               "color_noise": 0.0, "cross_color": 0.0, "cross_luma": 0.0, "edge_boost": 0.0},
    "player": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0,
               "color_noise": 0.0, "cross_color": 0.0, "cross_luma": 0.0},
    "codec": {"dv_enabled": 1.0},
    "tape": {"head_switching": 0.0},
    "display": {"enabled": 1.0, "gamma": 1.0, "saturation": 0.62},
}

# the effect: the glow, the trail, the coloured edge, and the ring outside a blowout
EFFECT = {"halation": 0.90, "ccd_smear": 0.85, "flare_defocus": 1.00, "lens_flare": 0.55,
          "spike_spread": 0.85, "chroma_aberration": 0.85, "axis_offset": 0.45,
          "longitudinal_ca": 0.65, "overflow_drain": 0.35}


def apply(camera):
    params = {k: dict(v) for k, v in COMMON.items()}
    params["camera"] = dict(COMMON["camera"], **camera)
    post("/api/project", {"reset_params": True})
    post("/api/project", {"params": params, "seed": 20240101, "standard": "ntsc"})
    return params


def render(name, scale=0.55):
    with urllib.request.urlopen(BASE + "/api/frame?t=0.0&scale=%s&q=96" % scale, timeout=900) as r:
        im = Image.open(io.BytesIO(r.read())).convert("RGB")
    im.save(os.path.join(OUT, name))
    x = np.asarray(im, np.float32) / 255.0
    L = x @ COEF
    rb = x[..., 0] - x[..., 2]
    # the glow: how much of the frame sits in the highlight range where the mechanisms act
    return im, dict(mean=[round(float(v), 4) for v in x.reshape(-1, 3).mean(axis=0)],
                    blown=round(float((L > 0.97).mean()), 5),
                    glow=round(float((L > 0.80).mean()), 5),
                    mid=round(float(((L > 0.55) & (L <= 0.80)).mean()), 5),
                    rb=round(float(x[..., 0].mean() - x[..., 2].mean()), 5),
                    row_std=round(float(np.std(rb.mean(axis=1))), 5))


print()
apply({})
im_c, m_c = render("final-control.png")
print("   control (mechanisms zero)  %s" % m_c)
apply(EFFECT)
im_e, m_e = render("final-effect.png")
print("   effect (mechanisms on)     %s" % m_e)

print()
print("   the effect against its control -- this is what makes it visible, if it is:")
for k in ("mean", "blown", "glow", "mid", "rb", "row_std"):
    if k == "mean":
        print("      %-9s %s -> %s" % (k, m_c[k], m_e[k]))
    else:
        print("      %-9s %.5f -> %.5f   (%+.5f)" % (k, m_c[k], m_e[k], m_e[k] - m_c[k]))

# and cropped around the lamp, where the effect lives
a = np.asarray(Image.open(os.path.join(OUT, "final-effect.png")).convert("RGB"), np.float32) / 255.0
L = a @ COEF
cy, cx = np.unravel_index(int(np.argmax(L)), L.shape)
r = 170
y0, y1 = max(0, cy - r), min(a.shape[0], cy + r)
x0, x1 = max(0, cx - r), min(a.shape[1], cx + r)
for tag, name in (("control", "final-control.png"), ("effect", "final-effect.png")):
    b = np.asarray(Image.open(os.path.join(OUT, name)).convert("RGB"), np.float32) / 255.0
    crop = b[y0:y1, x0:x1]
    im = Image.fromarray((np.clip(crop, 0, 1) * 255).astype(np.uint8))
    im = im.resize((im.width * 2, im.height * 2), Image.LANCZOS)
    im.save(os.path.join(OUT, "final-crop-%s.png" % tag))
print()
print("   the brightest pixel is at (row %d, col %d); crops written around it" % (cy, cx))

apply(EFFECT)
cfg = json.loads(get_raw("/api/export"))
with open(os.path.join(OUT, "repro-highlight.json"), "w", encoding="utf-8") as fh:
    json.dump(cfg, fh, ensure_ascii=False, indent=2)
print("   wrote repro-highlight.json: %d parameter(s) across %d layer(s)"
      % (cfg["param_count"], len(cfg["params"])))
for layer, vals in sorted(cfg["params"].items()):
    print("      %-9s %s" % (layer, {k: (round(v, 3) if isinstance(v, float) else v)
                                     for k, v in sorted(vals.items())}))
