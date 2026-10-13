"""
Test the chroma bandwidth at the pole's own location, which the invalidated sweep never did.

**Why this is the live candidate again.** The display's tone curve cannot account for the measured shift: `contrast`,
`brightness` and `saturation` are **affine** and so leave a channel difference alone, and `_col_temp` is a **flat
multiplier** `[1.015, 1.0015, 0.982]` whose effect on B-R is **proportional to the pixel's value** -- so it moves the
**brighter pole more than the darker wall**, which is the opposite direction from what was measured (the surround was
lifted **+0.062**, the pole **+0.058**). **No combination of those four terms produces a lift of +0.062 on a wall at
0.13-0.15.**

**And the arithmetic points at the chroma channel.** The pole is **6 px wide**. If the chroma bandwidth limits the
chroma channel to a scale comparable to that, then **the pole's colour and the wall's mix across the boundary** -- and
that mixing is exactly a mechanism that dilutes the wall's blue next to the pole. The earlier bandwidth sweeps were
taken at columns 308..322, i.e. **they measured the bandwidth's effect on a wall**, so those readings are void and the
question was never actually asked.

**So the bandwidth is swept now at the validated geometry**, and through both places it is applied -- the format stage's
and the player's -- because the two are separate limits on the same quantity and either could be the binding one.

**And the honest possibility this tests for**: a 6-px object at NTSC chroma bandwidth **may simply not keep its colour**,
in which case this is not a defect but a **subject-scale mismatch** -- the same class as the `source_glow` magnitude
earlier, where a coefficient right for a small source was two orders out for a large one.
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


def run(fmt=None, plr=None, disp=None):
    layers = {"camera": dict(LIGHT),
              "format": dict({"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0, "chroma_gain": 1.0,
                              "color_noise": 0.0, "cross_color": 0.0, "cross_luma": 0.0,
                              "edge_boost": 0.0}, **(fmt or {})),
              "player": dict({"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0, "color_noise": 0.0,
                              "cross_color": 0.0, "cross_luma": 0.0}, **(plr or {})),
              "codec": {"dv_enabled": 1.0}, "tape": {"head_switching": 0.0},
              "display": dict(RASTER_OFF, enabled=1.0, gamma=1.0, saturation=0.62,
                              **(disp or {}))}
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
print("   the source:  pole %+.4f  surround %+.4f  EXCESS %+.4f" % (REF["pole"], REF["surround"], REF["excess"]))
print()
print("   chroma bandwidth, swept at the pole's own columns for the first time")
print("   %-38s %-11s %-11s %-11s %s" % ("setting", "pole", "surround", "EXCESS", "sign"))


def line(label, **kw):
    q = pole(run(**kw))
    if q is None:
        print("   %-38s %-35s not visible" % (label, ""))
        return
    print("   %-38s %-+11.4f %-+11.4f %-+11.4f %s"
          % (label, q["pole"], q["surround"], q["excess"],
             "COOLER" if q["excess"] > 0 else "warmer (inverted)"))


line("as it stands (2.0 MHz both stages)")
for v in (0.4, 0.75, 1.2, 3.0, 6.0):
    line("format.chroma_bandwidth = %.2f" % v, fmt={"chroma_bandwidth": v})
print()
for v in (0.4, 0.75, 1.2, 3.0, 6.0):
    line("player.chroma_bandwidth = %.2f" % v, plr={"chroma_bandwidth": v})
print()
line("both stages at 6.0 (no limit at all)",
     fmt={"chroma_bandwidth": 6.0}, plr={"chroma_bandwidth": 6.0})
line("both stages at 0.4 (the narrowest)",
     fmt={"chroma_bandwidth": 0.4}, plr={"chroma_bandwidth": 0.4})
print()
line("chroma_gain = 0 (chroma removed entirely)", fmt={"chroma_gain": 0.0})
print()
print("   the decisive one: if removing the limit entirely still leaves the pole warmer than its surround,")
print("   then the bandwidth is not the cause and the difference is something else again.")
