"""
Deliver the reproduction on the photograph that can carry it, with the diagnosis recorded and the settings chosen.

**What is established and no longer being searched for.** The pole survives the chain when its **source** excess exceeds
the chain's own shift to the surround, which measures about **0.06**:

    refscene.png (the earlier one)   +0.0363  ->  overwhelmed, inverted at every setting
    this photograph                  +0.1459  ->  survives, and comes out at +0.2196
    the reference photograph         +0.2142  ->  survives at +0.2016

**So the eleven parameter candidates were excluded for a reason that is now measured rather than suspected**: the
quantity being asked for is not a parameter of the chain but a **property of the subject**, and no setting of a chain
that adds light can create a colour contrast the input does not have. **The earlier photograph's pole is a pale line on
a wall of nearly its own colour; this one is a cool pole against a warm building, and the measurement separates them by
a factor of four before the chain is applied.**

**And the geometry is locked to the source's pole** (columns 222..231, surround +/-40) because the free detector jumps
between structures as the render's brightness profile changes -- the failure its positive control predicted. With the
geometry fixed, the pole's excess across the light settings reads **+0.2196, +0.1189, +0.1067, +0.0925** -- ordered and
monotonic, which is what a usable measurement looks like.

**The settings kept are the reproduction's light group**, because the conservative choice (light at zero) produces a
render indistinguishable from the source and therefore demonstrates nothing. They do drive the frame brighter -- the
blown fraction goes from 0.40% to 3.07% and the fully-white fraction from 0.007% to 2.32% against the control -- and
**that is recorded as a property of this subject rather than tuned away**, since reducing the group lowers the pole's
excess toward the point where the original fault returns.

**The deliverable is therefore a settings file with its limitation written into it**, which is the honest form of a
result whose cause is understood: it reproduces the effect on material that has the contrast, and it says what that
material needs to have.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from atsim.core.params import merge_values                                   # noqa: E402
from atsim.core.profiles import FACTORY_PRESETS, SCHEMA                      # noqa: E402
from atsim.render.renderer import Renderer                                   # noqa: E402

OUT = os.path.join(ROOT, "tools", "ntscrs_ref")
COEF = np.array([0.299, 0.587, 0.114], np.float32)
BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])
pic = np.asarray(Image.open(os.path.join(OUT, "newscene.png")).convert("RGB"),
                 np.float32) / 255.0
POLE_L, POLE_R, OFF = 222, 231, 40


class Still:
    def __init__(self, p):
        self.height, self.width = p.shape[:2]
        self.fps, self.frame_count = 25.0, 3
        self._pic = p

    def frame(self, index):
        return self._pic


RASTER_OFF = {"scanlines": 0.0, "mask_depth": 0.0, "interlace_flicker": 0.0, "scan_pitch": 0.0,
              "spot_sigma": 0.0, "defocus": 0.0, "vignette": 0.0, "reflection": 0.0,
              "bloom": 0.0, "geometry_warp": 0.0, "overscan": 0.0, "curvature": 0.0,
              "breathing": 0.0, "crt_age": 0.0, "convergence": 0.0, "crt_moisture": 0.0,
              "crt_corrosion": 0.0}
ZERO = dict(halation=0.0, ccd_smear=0.0, flare_defocus=0.0, lens_flare=0.0, spike_spread=0.0,
            overflow_drain=0.0, chroma_aberration=0.0, longitudinal_ca=0.0, axis_offset=0.0,
            source_glow=0.0, halation_source_colour=0.0, shadow_depth=0.0, exposure=0.0,
            color_cast=0.0, lens_fog=0.0, lens_wear=0.0, bloom_amount=0.0, dead_sites=0.0)
ABERR = dict(chroma_aberration=0.6, longitudinal_ca=0.4, axis_offset=0.5)
EFFECT = dict(ABERR, source_glow=0.04, halation=0.25, ccd_smear=0.20, flare_defocus=0.35,
              lens_flare=0.30, spike_spread=0.40, overflow_drain=0.60)


def run(cam=None, name=None, scale=1.0):
    layers = {"camera": dict(ZERO, **(cam or {})),
              "format": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0, "chroma_gain": 1.0,
                         "color_noise": 0.0, "cross_color": 0.0, "cross_luma": 0.0,
                         "edge_boost": 0.0},
              "player": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0, "color_noise": 0.0,
                         "cross_color": 0.0, "cross_luma": 0.0},
              "codec": {"dv_enabled": 1.0}, "tape": {"head_switching": 0.0},
              "display": dict(RASTER_OFF, enabled=1.0, gamma=1.0, saturation=0.62)}
    p = merge_values(BASE, layers)
    r = Renderer(Still(pic), pic.shape[1], pic.shape[0], 25.0, p, seed=20240101)
    out = np.clip(np.asarray(r.render_frame(0), np.float32), 0, 1)
    if name:
        im = Image.fromarray((out * 255).astype(np.uint8))
        if scale != 1.0:
            im = im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS)
        im.save(os.path.join(OUT, name))
    return out, p


def locked(img, left=POLE_L, right=POLE_R, off=OFF, min_contrast=0.02):
    lum = img @ COEF
    Hh, Ww = lum.shape
    k = max(5, Hh // 40)
    box = np.lib.stride_tricks.sliding_window_view(lum, (k, k)).mean(axis=(2, 3))
    cy, _ = np.unravel_index(int(np.argmax(box)), box.shape)
    cy += k // 2
    rows = np.arange(min(Hh - 2, cy + 6), Hh)
    c0, c1 = max(0, left - off), max(1, left - 3)
    d0, d1 = min(Ww - 1, right + 3), min(Ww, right + off)
    pl = lum[rows][:, left:right + 1].mean(axis=1)
    sl = np.concatenate([lum[rows][:, c0:c1], lum[rows][:, d0:d1]], axis=1).mean(axis=1)
    keep = pl > sl + min_contrast
    if keep.sum() < 10:
        return None
    sel = rows[keep]
    ph = float(np.median([float((img[q, left:right + 1, 2] - img[q, left:right + 1, 0]).mean())
                          for q in sel]))
    sh = float(np.median([float(np.concatenate([
        img[q, c0:c1, 2] - img[q, c0:c1, 0],
        img[q, d0:d1, 2] - img[q, d0:d1, 0]]).mean()) for q in sel]))
    return dict(pole=ph, surround=sh, excess=ph - sh)


def stats(img):
    lum = img @ COEF
    q = locked(img)
    return dict(mean=float(lum.mean()), blown=float((lum > 0.97).mean()),
                glow=float((lum > 0.85).mean()),
                white=float(np.all(img >= 0.9999, axis=-1).mean()),
                excess=q["excess"] if q else None)


print("   the source:  mean %.5f  blown %.5f  glow %.5f  white %.5f  pole excess +0.1459"
      % (stats(pic)["mean"], stats(pic)["blown"], stats(pic)["glow"], stats(pic)["white"]))
c, _ = run(ABERR)
e, p = run(EFFECT, "newscene-effect.png")
sc, se = stats(c), stats(e)
print("   control:     mean %.5f  blown %.5f  glow %.5f  white %.5f  pole excess %+.4f"
      % (sc["mean"], sc["blown"], sc["glow"], sc["white"], sc["excess"]))
print("   effect:      mean %.5f  blown %.5f  glow %.5f  white %.5f  pole excess %+.4f"
      % (se["mean"], se["blown"], se["glow"], se["white"], se["excess"]))
print()
print("   the effect against its control:  blown %+.5f, glow %+.5f, pole excess %+.4f"
      % (se["blown"] - sc["blown"], se["glow"] - sc["glow"],
         (se["excess"] or 0) - (sc["excess"] or 0)))
# a crop around the lamp, which is where the effect lives
lum = e @ COEF
k = 31
box = np.lib.stride_tricks.sliding_window_view(lum, (k, k)).mean(axis=(2, 3))
cy, cx = np.unravel_index(int(np.argmax(box)), box.shape)
cy += k // 2
cx += k // 2
r = 200
y0, y1 = max(0, cy - r), min(e.shape[0], cy + r)
x0, x1 = max(0, cx - r), min(e.shape[1], cx + r)
for tag, arr in (("effect", e), ("control", c)):
    crop = np.clip(arr[y0:y1, x0:x1], 0, 1)
    im = Image.fromarray((crop * 255).astype(np.uint8))
    im = im.resize((im.width * 2, im.height * 2), Image.LANCZOS)
    im.save(os.path.join(OUT, "newscene-crop-%s.png" % tag))
print("   crops written around the lamp at (row %d, col %d)" % (cy, cx))

diff = {k2: {kk: vv for kk, vv in v.items() if BASE.get(k2, {}).get(kk) != vv}
        for k2, v in p.items()}
diff = {k2: v for k2, v in diff.items() if v}
with open(os.path.join(OUT, "repro-highlight.json"), "w", encoding="utf-8") as fh:
    json.dump({"params": diff, "param_count": sum(len(v) for v in diff.values()),
               "seed": 20240101, "standard": "ntsc",
               "source_note": "the input's pole must carry more colour contrast against its own surround than the "
                              "chain shifts that surround, about 0.06. Measured: +0.0363 inverts, +0.1459 survives.",
               "settings_note": "the light group is at the reproduction's values, which raise the blown fraction "
                                "from 0.40% to 3.07% on this subject; reducing it lowers the pole's excess toward "
                                "the range where the inversion returns."},
              fh, ensure_ascii=False, indent=2)
print("   wrote repro-highlight.json: %d parameter(s) across %d layer(s)"
      % (sum(len(v) for v in diff.values()), len(diff)))
