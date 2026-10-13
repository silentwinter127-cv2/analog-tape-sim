"""
Dial the mechanisms to the raw frame's own scale, because a radial aberration scales with the picture's width.

**What the last render showed.** The control is clean and natural -- a white wall, an orange reflection, a railing --
and the effect put strong red-green-blue fringing across the whole crop. The cause is measurable rather than a matter
of taste: **lateral chromatic aberration is radial and defined in pixels**, and its coefficient was chosen on
320-pixel test material where a full-strength setting displaces the edge by about 2.7 px. **At 3840 px the same
coefficient displaces it by more than fourteen.** `chroma_aberration 0.85` and `longitudinal_ca 0.65` were therefore
roughly five times too strong for this material, and the fringing is them.

So the aberrations come down, the light mechanisms -- halation, the charge smear, the flare, the overflow ring, which
do **not** scale with the picture's width -- stay where they were, and a control is rendered on the identical path
again. The comparison against the source is stated for all three.
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
    "display": {"enabled": 1.0, "gamma": 1.0, "saturation": 0.60},
}

# The aberrations scale with the picture's width; the light mechanisms do not.  So they are set separately, and the
# aberration values are the ones measured as "a visible fringe" on 320-px material rather than "the maximum".
EFFECT = {
    "halation": 0.90,        # the glow, per-unit-area: unchanged by the frame's size
    "ccd_smear": 0.85,       # charge down the column: likewise
    "flare_defocus": 1.00,   # per-channel defocus of the flare
    "lens_flare": 0.50,
    "spike_spread": 0.85,
    "overflow_drain": 0.35,
    "chroma_aberration": 0.20,   # radial and in pixels: 0.85 was ~5x too strong at 3840 px
    "longitudinal_ca": 0.12,     # a per-channel blur, also in pixels
    "axis_offset": 0.20,
}


def apply(camera):
    params = {k: dict(v) for k, v in COMMON.items()}
    params["camera"] = dict(COMMON["camera"], **camera)
    post("/api/project", {"reset_params": True})
    post("/api/project", {"params": params, "seed": 20240101, "standard": "ntsc"})
    return params


def render(name, scale=0.5):
    with urllib.request.urlopen(BASE + "/api/frame?t=0.0&scale=%s&q=96" % scale, timeout=900) as r:
        im = Image.open(io.BytesIO(r.read())).convert("RGB")
    im.save(os.path.join(OUT, name))
    x = np.asarray(im, np.float32) / 255.0
    L = x @ COEF
    # the fringing, as the mean absolute R-B away from neutral: a clean picture has little, a fringed one a lot
    return im, dict(mean=round(float(L.mean()), 5),
                    blown=round(float((L > 0.97).mean()), 5),
                    glow=round(float((L > 0.85).mean()), 5),
                    fringe=round(float(np.abs(x[..., 0] - x[..., 2]).mean()), 5))


print()
apply({})
im_c, m_c = render("raw2-control.png")
print("   control (mechanisms zero)  %s" % m_c)
apply(EFFECT)
im_e, m_e = render("raw2-effect.png")
print("   effect (dialled to scale)  %s" % m_e)

print()
print("   the effect against its control:")
for k in ("mean", "blown", "glow", "fringe"):
    print("      %-7s %.5f -> %.5f   (%+.5f)" % (k, m_c[k], m_e[k], m_e[k] - m_c[k]))

print()
print("   and against the earlier over-driven version, so the correction is visible as a number:")
try:
    old = np.asarray(Image.open(os.path.join(OUT, "raw-effect.png")).convert("RGB"),
                     np.float32) / 255.0
    print("      fringing: over-driven %.5f -> dialled %.5f  (control is %.5f)"
          % (float(np.abs(old[..., 0] - old[..., 2]).mean()), m_e["fringe"], m_c["fringe"]))
except Exception as exc:                                                    # noqa: BLE001
    print("      (the earlier render is not available: %s)" % exc)

# crops around the same place, at the same scale, so the two pairs can be compared directly
for tag, name in (("control", "raw2-control.png"), ("effect", "raw2-effect.png")):
    b = np.asarray(Image.open(os.path.join(OUT, name)).convert("RGB"), np.float32) / 255.0
    L = b @ COEF
    k = 25
    box = np.lib.stride_tricks.sliding_window_view(L, (k, k)).mean(axis=(2, 3))
    cy, cx = np.unravel_index(int(np.argmax(box)), box.shape)
    cy += k // 2
    cx += k // 2
    r = 150
    y0, y1 = max(0, cy - r), min(b.shape[0], cy + r)
    x0, x1 = max(0, cx - r), min(b.shape[1], cx + r)
    im = Image.fromarray((np.clip(b[y0:y1, x0:x1], 0, 1) * 255).astype(np.uint8))
    im = im.resize((im.width * 3, im.height * 3), Image.LANCZOS)
    im.save(os.path.join(OUT, "raw2-crop-%s.png" % tag))

apply(EFFECT)
cfg = json.loads(get_raw("/api/export"))
with open(os.path.join(OUT, "repro-highlight.json"), "w", encoding="utf-8") as fh:
    json.dump(cfg, fh, ensure_ascii=False, indent=2)
print()
print("   wrote repro-highlight.json: %d parameter(s) across %d layer(s)"
      % (cfg["param_count"], len(cfg["params"])))
for layer, vals in sorted(cfg["params"].items()):
    print("      %-9s %s" % (layer, {k2: (round(v2, 3) if isinstance(v2, float) else v2)
                                     for k2, v2 in sorted(vals.items())}))
