"""
Locate the pole independently on every frame, which is the fix for the instrument's remaining fault.

**The fault, stated exactly.** The band was located **once, on the source**, and then those columns were used to measure
every render. But the mechanisms change the picture -- `source_glow` alone lifts the surroundings around a source, and
the others bloom and smear -- so a fixed band and a fixed surround offset are **not measuring the same object** the
source's measurement did. That is why the excess and the pole's hue disagreed about which value was "the same": the two
numbers were being read from different geometry.

**The fix is to locate the pole on each frame by the same rule**, so the comparison is always pole-against-its-own-
surround. That also removes the need to assume the pole has not moved, which was never checked.

**And the rule is applied identically**, so a frame where the mechanisms have buried the pole will **fail to find one**
rather than silently measuring a wall -- which is the behaviour that makes the instrument trustworthy. Frames where no
pole is found are reported as such.
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
W, H = 452, 382


def find_pole(img, min_rows=40):
    """
    The pole, located on this frame: the column band whose brightness exceeds its horizontal neighbours
    **consistently down the frame**.

    A wall cannot do this -- its excess fluctuates around zero -- so requiring the excess to hold over many rows is
    what makes the search an object detector rather than a brightest-column pick. Returns `None` when no band holds
    up, so a frame with no visible pole is reported instead of being measured as something else.
    """
    lum = img @ COEF
    k = max(5, img.shape[0] // 40)
    box = np.lib.stride_tricks.sliding_window_view(lum, (k, k)).mean(axis=(2, 3))
    cy, cx = np.unravel_index(int(np.argmax(box)), box.shape)
    cy += k // 2
    cx += k // 2
    rows = slice(min(H - 2, cy + 8), H)
    band = lum[rows]
    if band.shape[0] < min_rows:
        return None
    wide = np.apply_along_axis(lambda r: np.convolve(r, np.ones(41) / 41, mode="same"), 1, band)
    excess = band - wide
    score = excess.mean(axis=0)
    lo, hi = max(0, cx - W // 3), min(W, cx + W // 3)
    if hi - lo < 5:
        lo, hi = 0, W
    j = lo + int(np.argmax(score[lo:hi]))
    # the band must actually stand out: the column's excess over its neighbours must be positive and persistent
    if score[j] <= 0.01:
        return None
    left = right = j
    while left - 1 >= 0 and score[left - 1] > 0.3 * score[j]:
        left -= 1
    while right + 1 < W and score[right + 1] > 0.3 * score[j]:
        right += 1
    sel = rows
    col_ok = np.ones(band.shape[0], dtype=bool)
    off = max(12, (right - left) * 4)
    c0, c1 = max(0, left - off), max(1, left - 3)
    d0, d1 = min(W - 1, right + 3), min(W, right + off)
    pole_lum = band[:, left:right + 1].mean(axis=1)
    sur_lum = np.concatenate([band[:, c0:c1], band[:, d0:d1]], axis=1).mean(axis=1)
    col_ok = pole_lum > sur_lum + 0.02
    if col_ok.sum() < min_rows:
        return None
    return dict(cy=cy, cx=cx, left=int(left), right=int(right), width=int(right - left + 1),
                c0=int(c0), c1=int(c1), d0=int(d0), d1=int(d1), keep=col_ok, rows=slice(rows.start, rows.stop))


def measure(img):
    """The pole's hue, its surround's hue, and the excess between them, all located on this frame."""
    r = find_pole(img)
    if r is None:
        return None
    rows = np.arange(r["rows"].start, r["rows"].stop)
    sel = rows[r["keep"]]
    if len(sel) < 20:
        return None
    lum = img @ COEF
    pole_hue = float(np.median([float((img[q, r["left"]:r["right"] + 1, 2]
                                      - img[q, r["left"]:r["right"] + 1, 0]).mean()) for q in sel]))
    sur_hue = float(np.median([float(np.concatenate([
        img[q, r["c0"]:r["c1"], 2] - img[q, r["c0"]:r["c1"], 0],
        img[q, r["d0"]:r["d1"], 2] - img[q, r["d0"]:r["d1"], 0]]).mean()) for q in sel]))
    con = float(np.median([float(lum[q, r["left"]:r["right"] + 1].mean()
                                 - np.concatenate([lum[q, r["c0"]:r["c1"]],
                                                   lum[q, r["d0"]:r["d1"]]]).mean()) for q in sel]))
    return dict(band=(r["left"], r["right"]), width=r["width"], n=int(len(sel)),
                pole=pole_hue, surround=sur_hue, excess=pole_hue - sur_hue, contrast=con)


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
CHOSEN = dict(halation=0.25, ccd_smear=0.20, flare_defocus=0.35, lens_flare=0.30,
              spike_spread=0.40, overflow_drain=0.60, chroma_aberration=0.6,
              longitudinal_ca=0.4, axis_offset=0.5, halation_source_colour=0.0,
              shadow_depth=0.0, exposure=0.0, color_cast=0.0, lens_fog=0.0, lens_wear=0.0,
              bloom_amount=0.0, dead_sites=0.0)


def run(camera, sat=0.62):
    layers = {"camera": dict(camera), "format": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0},
              "player": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0},
              "codec": {"dv_enabled": 1.0}, "tape": {"head_switching": 0.0},
              "display": dict(RASTER_OFF, enabled=1.0, gamma=1.0, saturation=sat)}
    p = merge_values(BASE, layers)
    r = Renderer(Still(), W, H, 25.0, p, seed=20240101)
    return np.clip(np.asarray(r.render_frame(0), np.float32), 0, 1)


REF = measure(PIC)
print("   THE REFERENCE SCENE, located on its own frame:")
print("      band %d..%d (%d px), %d rows |  pole %+.4f, surround %+.4f, EXCESS %+.4f, contrast %.4f"
      % (REF["band"][0], REF["band"][1], REF["width"], REF["n"],
         REF["pole"], REF["surround"], REF["excess"], REF["contrast"]))
print()
print("   %-14s %-14s %-11s %-11s %-11s %-11s %s"
      % ("source_glow", "band found", "pole", "surround", "EXCESS", "contrast", "vs the reference"))
for g in (0.0, 0.02, 0.04, 0.06, 0.10, 0.15):
    q = measure(run(dict(CHOSEN, source_glow=g)))
    if q is None:
        print("   %-14.2f %-14s" % (g, "NONE -- no pole visible"))
        continue
    print("   %-14.2f %-14s %-+11.4f %-+11.4f %-+11.4f %-11.4f %+.4f"
          % (g, "%d..%d (%dpx)" % (q["band"][0], q["band"][1], q["width"]),
             q["pole"], q["surround"], q["excess"], q["contrast"], q["excess"] - REF["excess"]))
print()
print("   and the point of locating every frame: the band must be found in the same place, or the two frames are")
print("   not being compared on the same object.")
