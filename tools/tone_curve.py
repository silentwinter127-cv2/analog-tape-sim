"""
Set the tone curve against the pole's excess instead of inheriting it, and check the three inert terms.

**Why this is the step.** `brightness` and `contrast` carry the inversion -- `brightness = 1.0` puts the excess at
**+0.0207** against the source's **+0.0363**, restoring the sign, while `contrast = 1.0` drives it to **-0.1496**.
Those are the two levers, they interact, and neither appears in the reproduction's display dictionary at all: both sit
at the **base preset's defaults**, which is how a tone curve that inverts the pole's contrast came along unexamined.

**The search is two-dimensional and small**, since brightness and contrast are each one number in 0..1 and the target is
a single figure. It is run as a grid and the best pair is reported with its distance from the target, **rather than
tuning one and assuming the other**.

**And the three inert terms are checked for reachability**, separately from the sweep, because a parameter that does
nothing across its whole range is a fault of a different kind -- the same one found earlier in `player.line_delay`,
`tape.row_chroma_*` and `player.chroma_vert_blend`. **Two tests:** does it change the picture at all (a global
statistic), and does it change the pole specifically. A term can be inert for the pole and still act elsewhere.
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


def run(disp=None):
    layers = {"camera": dict(LIGHT),
              "format": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0, "chroma_gain": 1.0,
                         "color_noise": 0.0, "cross_color": 0.0, "cross_luma": 0.0,
                         "edge_boost": 0.0},
              "player": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0, "color_noise": 0.0,
                         "cross_color": 0.0, "cross_luma": 0.0},
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
print("   the source:  EXCESS %+.4f  (pole %+.4f, surround %+.4f)" % (REF["excess"], REF["pole"], REF["surround"]))
print()
print("   the tone-curve grid -- excess, and the distance from the source's %+.4f" % REF["excess"])
grid = [0.0, 0.25, 0.5, 0.75, 1.0]
print("   %-10s %s" % ("brightness", "  ".join("%-19s" % ("contrast %.2f" % c) for c in grid)))
best = None
for b in grid:
    cells = []
    for c in grid:
        q = pole(run({"brightness": b, "contrast": c}))
        if q is None:
            cells.append("%-19s" % "not visible")
            continue
        cells.append("%-+8.4f (%+.4f)" % (q["excess"], q["excess"] - REF["excess"]))
        if best is None or abs(q["excess"] - REF["excess"]) < abs(best[2]["excess"] - REF["excess"]):
            best = (b, c, q)
    print("   %-10.2f %s" % (b, "  ".join(cells)))
print()
print("   the closest pair: brightness %.2f, contrast %.2f -> excess %+.4f (source %+.4f, off by %+.4f)"
      % (best[0], best[1], best[2]["excess"], REF["excess"], best[2]["excess"] - REF["excess"]))

print()
print("   and the three inert terms, tested on a GLOBAL statistic as well as on the pole -- a term can be inert")
print("   for the pole and still act elsewhere on the picture:")
print("   %-16s %-12s %-14s %-14s %s"
      % ("term", "value", "frame mean", "pole excess", "changes the picture?"))
base_img = run({"brightness": best[0], "contrast": best[1]})
base_mean = float((base_img @ COEF).mean())
for k in ("crt_age", "convergence", "mask_depth"):
    for v in (0.0, 1.0):
        img = run({"brightness": best[0], "contrast": best[1], k: v})
        q = pole(img)
        mean = float((img @ COEF).mean())
        same = np.allclose(img, base_img, atol=1e-6)
        print("   %-16s %-12.2f %-14.5f %-14s %s"
              % (k, v, mean, ("%+.4f" % q["excess"]) if q else "n/a",
                 "NO (identical frame)" if same else "yes (mean %+.5f)" % (mean - base_mean)))
