"""
Test the mechanisms that act on chroma regardless of brightness -- the only candidates the sweep did not cover.

**Why these.** Ten mechanisms were varied over their ranges in the last sweep and **the pole's excess never changed
sign**: the source has the pole **cooler** than its surround (+0.0363) and every render has it **warmer** (negative at
all settings). A fault that no light mechanism can touch must come from something that acts on colour **independently
of how bright the pixel is**. The model has three such groups, and none was varied:

    the chroma bandwidth limits     format.chroma_bandwidth, player.chroma_bandwidth
    the crosstalk terms             format.cross_color, format.cross_luma, player.*
    the display's saturation        display.saturation

**And the arithmetic says one of them is a prime suspect.** The display applies `sat = 0.5 + 1.15 * saturation`, which
at the reproduction's **0.62** is **1.21** -- a **scaling** of every chroma difference. A scaling cannot change a sign,
so saturation alone cannot invert the pole. But the **bandwidth limits** can: the pole is a 6-px-wide structure, and a
chroma lowpass narrow enough to smear 6 px will mix the pole's colour with the wall's, which *can* move it past zero.

**So each group is neutralised in turn with the light mechanisms held at the reproduction's values**, and the pole's
excess -- measured at the columns the control validated -- is read. The target is the source's **+0.0363**; a group
that is responsible will show a positive excess when it is removed, and one that is not will leave the sign where it is.
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


class Still:
    def __init__(self):
        self.width, self.height, self.fps, self.frame_count = W, H, 25.0, 3

    def frame(self, index):
        return PIC


BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])
RASTER_OFF = {"scanlines": 0.0, "mask_depth": 0.0, "interlace_flicker": 0.0, "scan_pitch": 0.0,
              "spot_sigma": 0.0, "defocus": 0.0, "vignette": 0.0, "reflection": 0.0,
              "bloom": 0.0, "geometry_warp": 0.0, "overscan": 0.0, "curvature": 0.0,
              "breathing": 0.0, "crt_age": 0.0, "convergence": 0.0, "crt_moisture": 0.0,
              "crt_corrosion": 0.0}
LIGHT = dict(halation=0.25, ccd_smear=0.20, flare_defocus=0.35, lens_flare=0.30,
             spike_spread=0.40, overflow_drain=0.60, chroma_aberration=0.6,
             longitudinal_ca=0.4, axis_offset=0.5, halation_source_colour=0.0,
             shadow_depth=0.0, source_glow=0.04, exposure=0.0, color_cast=0.0,
             lens_fog=0.0, lens_wear=0.0, bloom_amount=0.0, dead_sites=0.0)


def run(camera=None, fmt=None, plr=None, disp=None, sat=0.62):
    layers = {"camera": dict(LIGHT, **(camera or {})),
              "format": dict({"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0, "chroma_gain": 1.0,
                              "color_noise": 0.0, "cross_color": 0.0, "cross_luma": 0.0,
                              "edge_boost": 0.0}, **(fmt or {})),
              "player": dict({"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0, "color_noise": 0.0,
                              "cross_color": 0.0, "cross_luma": 0.0}, **(plr or {})),
              "codec": {"dv_enabled": 1.0}, "tape": {"head_switching": 0.0},
              "display": dict(RASTER_OFF, enabled=1.0, gamma=1.0,
                              saturation=(0.62 if disp is None else sat))}
    if disp:
        layers["display"].update(disp)
    p = merge_values(BASE, layers)
    r = Renderer(Still(), W, H, 25.0, p, seed=20240101)
    return np.clip(np.asarray(r.render_frame(0), np.float32), 0, 1)


def pole(img, left=LEFT, right=RIGHT, off=OFF, min_contrast=0.02):
    lum = img @ COEF
    Hh, Ww = img.shape[:2]
    k = max(5, Hh // 40)
    box = np.lib.stride_tricks.sliding_window_view(lum, (k, k)).mean(axis=(2, 3))
    cy, _ = np.unravel_index(int(np.argmax(box)), box.shape)
    cy += k // 2
    rows = np.arange(min(Hh - 2, cy + 8), Hh)
    c0, c1 = max(0, left - off), max(1, left - 3)
    d0, d1 = min(Ww - 1, right + 3), min(Ww, right + off)
    pl = lum[rows][:, left:right + 1].mean(axis=1)
    sl = np.concatenate([lum[rows][:, c0:c1], lum[rows][:, d0:d1]], axis=1).mean(axis=1)
    keep = pl > sl + min_contrast
    if keep.sum() < 10:
        return None, int(keep.sum()), len(rows)
    sel = rows[keep]
    ph = float(np.median([float((img[q, left:right + 1, 2] - img[q, left:right + 1, 0]).mean())
                          for q in sel]))
    sh = float(np.median([float(np.concatenate([
        img[q, c0:c1, 2] - img[q, c0:c1, 0],
        img[q, d0:d1, 2] - img[q, d0:d1, 0]]).mean()) for q in sel]))
    return dict(pole=ph, surround=sh, excess=ph - sh, n=int(keep.sum())), int(keep.sum()), len(rows)


REF, ns, ntot = pole(PIC)
print("   the source:  pole %+.4f, surround %+.4f, EXCESS %+.4f  (%d of %d rows)"
      % (REF["pole"], REF["surround"], REF["excess"], ns, ntot))
print()
print("   %-46s %-11s %-11s %-11s %s"
      % ("setting", "pole", "surround", "EXCESS", "sign"))


def line(label, **kw):
    q, n_ok, n_tot = pole(run(**kw))
    if q is None:
        print("   %-46s %-35s %d of %d (not visible)" % (label, "", n_ok, n_tot))
        return
    print("   %-46s %-+11.4f %-+11.4f %-+11.4f %s"
          % (label, q["pole"], q["surround"], q["excess"],
             "cooler (as the source)" if q["excess"] > 0 else "warmer (inverted)"))


line("the reproduction as it stands")
print()
line("chroma bandwidth at the maximum (no limit)",
     fmt={"chroma_bandwidth": 6.0}, plr={"chroma_bandwidth": 6.0})
line("chroma bandwidth at the minimum",
     fmt={"chroma_bandwidth": 0.4}, plr={"chroma_bandwidth": 0.4})
print()
line("crosstalk at its maximum", fmt={"cross_color": 1.0, "cross_luma": 1.0},
     plr={"cross_color": 1.0, "cross_luma": 1.0})
line("crosstalk at zero (as it stands)")
print()
line("display saturation 0.0", sat=0.0)
line("display saturation 1.0", sat=1.0)
print()
line("display disabled entirely", disp={"enabled": 0.0})
print()
line("the codec switched off (composite instead of component)", camera={"dv_enabled": None})
print()
print("   and to establish whether the fault is even in the chain rather than in the source:")
line("every light mechanism at zero")
line("every light mechanism at zero, display disabled",
     camera={k: 0.0 for k in ("halation", "ccd_smear", "flare_defocus", "lens_flare",
                              "spike_spread", "overflow_drain", "chroma_aberration",
                              "longitudinal_ca", "axis_offset", "source_glow")},
     disp={"enabled": 0.0})
