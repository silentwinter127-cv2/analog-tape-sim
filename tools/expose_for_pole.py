"""
Match the render's exposure to the source's, because the pole is blown to white and a white pixel has no colour.

**The cause, and it is not a mechanism.** The pole's band in the render is **10.8% fully white** -- R, G and B all at
exactly 1.0 -- and the frame overall **3.4%**, against the source's **0.00001%**. **A pixel whose three channels are all
saturated has a channel difference of exactly zero by arithmetic**, which is why the pole's B-R read `+0.0000` in every
sweep: bandwidth, crosstalk, saturation, the tone curve, `chroma_gain`, and the roll-off clamp all left it unchanged
because **there was no colour left to move**.

**And the roll-off clamp is excluded as the cause** by raising `dynamic_range` to 16, which puts the white point at
**2.0** so the clamp can bind only above twice white: the excess stays inverted (**-0.0513**) and the band is still
**9.4%** fully white. So the blowout is not the clamp pinning values that were merely above 1.0 -- **it is the picture
genuinely being driven past white before the clamp sees it.**

**So the lever is exposure and the added light, and this is the same class of error as the `source_glow` magnitude
earlier:** the mechanisms were set at values that suit a dim source, and this source's pole is already near white, so
adding light pushes it through. **The fix is to expose for the subject** -- the pole must sit below saturation in the
render as it does in the source -- **and the measurement that governs it is the fully-white fraction**, which the source
puts at essentially zero.

**The target is therefore explicit**: the render's fully-white fraction should be of the order of the source's
(0.00001), or at least small enough that the pole's band retains colour. Exposure is swept against it, and the pole's
excess is read beside it, since the whole point is to recover the colour that the blowout removed.
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
             shadow_depth=0.0, source_glow=0.04, color_cast=0.0,
             lens_fog=0.0, lens_wear=0.0, bloom_amount=0.0, dead_sites=0.0)
RASTER_OFF = {"scanlines": 0.0, "mask_depth": 0.0, "interlace_flicker": 0.0, "scan_pitch": 0.0,
              "spot_sigma": 0.0, "defocus": 0.0, "vignette": 0.0, "reflection": 0.0,
              "bloom": 0.0, "geometry_warp": 0.0, "overscan": 0.0, "curvature": 0.0,
              "breathing": 0.0, "crt_age": 0.0, "convergence": 0.0, "crt_moisture": 0.0,
              "crt_corrosion": 0.0}


def run(exposure=0.0, cam=None, sat=0.62):
    layers = {"camera": dict(LIGHT, exposure=exposure, **(cam or {})),
              "format": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0, "chroma_gain": 1.0,
                         "color_noise": 0.0, "cross_color": 0.0, "cross_luma": 0.0,
                         "edge_boost": 0.0},
              "player": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0, "color_noise": 0.0,
                         "cross_color": 0.0, "cross_luma": 0.0},
              "codec": {"dv_enabled": 1.0}, "tape": {"head_switching": 0.0},
              "display": dict(RASTER_OFF, enabled=1.0, gamma=1.0, saturation=sat)}
    p = merge_values(BASE, layers)
    r = Renderer(Still(), W, H, 25.0, p, seed=20240101)
    return np.clip(np.asarray(r.render_frame(0), np.float32), 0, 1)


def stats(img):
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
    ex = None
    if keep.sum() >= 10:
        sel = rows[keep]
        ph = float(np.median([float((img[q, LEFT:RIGHT + 1, 2] - img[q, LEFT:RIGHT + 1, 0]).mean())
                              for q in sel]))
        sh = float(np.median([float(np.concatenate([
            img[q, c0:c1, 2] - img[q, c0:c1, 0],
            img[q, d0:d1, 2] - img[q, d0:d1, 0]]).mean()) for q in sel]))
        ex = ph - sh
    return dict(ex=ex, band_white=float(np.all(img[rows][:, LEFT:RIGHT + 1] >= 0.9999, axis=-1).mean()),
                frame_white=float(np.all(img >= 0.9999, axis=-1).mean()),
                mean=float(lum.mean()), p999=float(np.quantile(lum, 0.9999)))


s = stats(PIC)
print("   the source:  mean %.4f, 99.99th pct luma %.4f, fully white %.5f, EXCESS %+.4f"
      % (s["mean"], s["p999"], s["frame_white"], 0.0363))
print()
print("   %-12s %-11s %-11s %-15s %-15s %s"
      % ("exposure", "mean", "p99.99", "band fully white", "frame fully white", "EXCESS"))
best = None
for e in (0.0, -0.3, -0.6, -0.9, -1.2, -1.6, -2.0, -2.5):
    q = stats(run(exposure=e))
    print("   %-12.2f %-11.4f %-11.4f %-15.5f %-15.5f %s"
          % (e, q["mean"], q["p999"], q["band_white"], q["frame_white"],
             ("%+.4f" % q["ex"]) if q["ex"] is not None else "not visible"))
    if q["ex"] is not None and (best is None or abs(q["ex"] - 0.0363) < abs(best[1] - 0.0363)):
        best = (e, q["ex"], q)
print()
if best:
    print("   the exposure nearest the source's excess: %.2f -> EXCESS %+.4f, band fully white %.5f"
          % (best[0], best[1], best[2]["band_white"]))
    print("   (the source's band is essentially never fully white, so the band's figure is the guard)")
