"""
Neutralise the display's terms one at a time, to find which carries the brightness-dependent blue.

**What is already established.** The pole's excess is inverted by the **display stage** -- disabling it restores the
sign (+0.0040 against -0.0319) and nothing else does; not the chroma bandwidth, the crosstalk, the codec, any of the ten
light mechanisms, or `saturation` (0.0 and 1.0 give identical figures to four decimals). **And the form of the fault is
measured**: the display moves the pole +0.058 in B-R and the **dark surround +0.062**, so both are lifted and the
surround crosses above the pole.

**And a likely candidate is visible in the code.** `display.color_temp` is applied as a **per-channel multiplier**,

    [1 + 0.10*t,  1 + 0.01*t,  1 - 0.12*t]

which at its default of **0.15** is **[1.015, 1.0015, 0.982]** -- i.e. it *reduces blue*. That cannot be what adds the
blue, but it shows the kind of term to look for: **a per-channel constant**. The other per-channel or
brightness-dependent candidates are the tone curve, `crt_age`, `convergence`, the mask, `bloom` and the reflection.

**And there is a gap worth closing first.** The reproduction's display dictionary does not mention `color_temp`,
`contrast`, `brightness`, `mask_pitch` or the trapezoid terms, so they are at the **base preset's** values rather than
at a considered one. **Measured against the baseline's values rather than against zero**, which is what this run does.
"""
from __future__ import annotations

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
W, H = 452, 382
LEFT, RIGHT, OFF = 270, 275, 24

src = np.asarray(Image.open(os.path.join(OUT, "refscene.png")).convert("RGB"),
                 np.float32) / 255.0
PIC = np.asarray(Image.fromarray((src * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR),
                 np.float32) / 255.0
BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])
print("   the base preset's display values for the terms the reproduction does not mention:")
for k in ("color_temp", "contrast", "brightness", "mask_pitch", "crt_age", "gamma",
          "trapezoid_x", "trapezoid_y", "convergence", "bloom", "reflection", "vignette"):
    sp = SCHEMA.layer("display").specs.get(k)
    if sp is not None:
        print("      %-14s default %-8s (range %s..%s)" % (k, sp.default, sp.minimum, sp.maximum))


class Still:
    def __init__(self):
        self.width, self.height, self.fps, self.frame_count = W, H, 25.0, 3

    def frame(self, index):
        return PIC


LIGHT = dict(halation=0.25, ccd_smear=0.20, flare_defocus=0.35, lens_flare=0.30,
             spike_spread=0.40, overflow_drain=0.60, chroma_aberration=0.6,
             longitudinal_ca=0.4, axis_offset=0.5, halation_source_colour=0.0,
             shadow_depth=0.0, source_glow=0.04, exposure=0.0, color_cast=0.0,
             lens_fog=0.0, lens_wear=0.0, bloom_amount=0.0, dead_sites=0.0)
RASTER_OFF = {"scanlines": 0.0, "mask_depth": 0.0, "interlace_flicker": 0.0, "scan_pitch": 0.0,
              "spot_sigma": 0.0, "defocus": 0.0, "vignette": 0.0, "reflection": 0.0,
              "bloom": 0.0, "geometry_warp": 0.0, "overscan": 0.0, "curvature": 0.0,
              "breathing": 0.0, "crt_age": 0.0, "convergence": 0.0, "crt_moisture": 0.0,
              "crt_corrosion": 0.0}


def run(disp=None, sat=0.62):
    layers = {"camera": dict(LIGHT),
              "format": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0, "chroma_gain": 1.0,
                         "color_noise": 0.0, "cross_color": 0.0, "cross_luma": 0.0,
                         "edge_boost": 0.0},
              "player": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0, "color_noise": 0.0,
                         "cross_color": 0.0, "cross_luma": 0.0},
              "codec": {"dv_enabled": 1.0}, "tape": {"head_switching": 0.0},
              "display": dict(RASTER_OFF, enabled=1.0, saturation=sat, **(disp or {}))}
    p = merge_values(BASE, layers)
    r = Renderer(Still(), W, H, 25.0, p, seed=20240101)
    return np.clip(np.asarray(r.render_frame(0), np.float32), 0, 1)


def pole(img):
    lum = img @ COEF
    Hh, Ww = img.shape[:2]
    k = max(5, Hh // 40)
    box = np.lib.stride_tricks.sliding_window_view(lum, (k, k)).mean(axis=(2, 3))
    cy, _ = np.unravel_index(int(np.argmax(box)), box.shape)
    cy += k // 2
    rows = np.arange(min(Hh - 2, cy + 8), Hh)
    c0, c1 = max(0, LEFT - OFF), max(1, LEFT - 3)
    d0, d1 = min(Ww - 1, RIGHT + 3), min(Ww, RIGHT + OFF)
    pl = lum[rows][:, LEFT:RIGHT + 1].mean(axis=1)
    sl = np.concatenate([lum[rows][:, c0:c1], lum[rows][:, d0:d1]], axis=1).mean(axis=1)
    keep = pl > sl + 0.02
    if keep.sum() < 10:
        return None
    sel = rows[keep]
    ph = float(np.median([float((img[q, LEFT:RIGHT + 1, 2] - img[q, LEFT:RIGHT + 1, 0]).mean())
                          for q in sel]))
    sh = float(np.median([float(np.concatenate([
        img[q, c0:c1, 2] - img[q, c0:c1, 0],
        img[q, d0:d1, 2] - img[q, d0:d1, 0]]).mean()) for q in sel]))
    return dict(pole=ph, surround=sh, excess=ph - sh, n=int(keep.sum()))


REF = pole(PIC)
print()
print("   the source:  pole %+.4f, surround %+.4f, EXCESS %+.4f" % (REF["pole"], REF["surround"], REF["excess"]))
print()
print("   %-40s %-11s %-11s %-11s %s" % ("display setting", "pole", "surround", "EXCESS", "sign"))


def line(label, **kw):
    q = pole(run(**kw))
    if q is None:
        print("   %-40s not visible" % label)
        return
    print("   %-40s %-+11.4f %-+11.4f %-+11.4f %s"
          % (label, q["pole"], q["surround"], q["excess"],
             "COOLER (as the source)" if q["excess"] > 0 else "warmer (inverted)"))


line("as it stands (color_temp at the default)")
for k, vals in (("color_temp", (0.0, 0.4)), ("contrast", (0.0, 1.0)), ("brightness", (0.0, 1.0)),
                ("crt_age", (0.0, 1.0)), ("convergence", (0.0, 1.0)), ("mask_depth", (0.0, 1.0)),
                ("bloom", (0.0, 1.0)), ("reflection", (0.0, 1.0)), ("vignette", (0.0, 1.0)),
                ("gamma", (0.6, 2.4))):
    for v in vals:
        line("%s = %.2f" % (k, v), disp={k: v})
    print()
line("display disabled", disp={"enabled": 0.0})
