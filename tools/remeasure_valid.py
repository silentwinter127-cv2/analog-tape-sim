"""
Re-measure with the detector that passed the positive control, and see whether the withdrawn figures come back.

**What the control settled.** The detector, unconstrained, is fooled by a bright vertical edge elsewhere on the wall --
it picks the edge (band 330..335) in preference to the pole (band 200..205), which is exactly what it did on the real
renders when it reported column 308. Constrained to a narrow window around the pole it finds the pole in all three
cases, **and it still returns "not found" on a wall with no pole**, so it does not invent one. Those are the two
properties that make it usable, and both are now established by a case with a known answer rather than by argument.

**The window is set from the source**, where the pole is unambiguous: columns **270..275**, so the search is bounded to
**230..320**. That is not an assumption that the pole cannot move; it is a statement that the measurement is about the
pole in that region, and the **found band is reported every time** so a run that drifts out of the window is visible
rather than silent.

**And the figures from the last-but-one turn are re-taken**, because they were produced by the fooled detector: the
three-way sweep of `source_glow`, `halation_source_colour` and `shadow_depth` against the pole's excess. Whether the
withdrawn reversal -- "the line was over-distinct" -- survives is exactly what this measures.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from atsim.core.params import merge_values                                   # noqa: E402
from atsim.core.profiles import FACTORY_PRESETS, SCHEMA                      # noqa: E402
from atsim.render.renderer import Renderer                                   # noqa: E402
from pole_control import find_pole, measure                                  # noqa: E402

OUT = os.path.join(ROOT, "tools", "ntscrs_ref")
COEF = np.array([0.299, 0.587, 0.114], np.float32)
W, H = 452, 382

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
WIN = (230, 320)

print("   the source, by the constrained detector:")
r = find_pole(PIC, search=WIN)
q = measure(PIC, search=WIN)
print("      band %d..%d (%d px), %d rows | pole %+.4f, surround %+.4f, EXCESS %+.4f, contrast %.4f"
      % (q["band"][0], q["band"][1], q["width"], q["n"], q["pole"], q["surround"],
         q["excess"], q["contrast"]))
print("      (the unconstrained detector reported band %d..%d on this same image)"
      % (find_pole(PIC)["left"], find_pole(PIC)["right"]))


def run(camera, sat=0.62):
    layers = {"camera": dict(camera), "format": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0},
              "player": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0},
              "codec": {"dv_enabled": 1.0}, "tape": {"head_switching": 0.0},
              "display": dict(RASTER_OFF, enabled=1.0, gamma=1.0, saturation=sat)}
    p = merge_values(BASE, layers)
    r = Renderer(Still(), W, H, 25.0, p, seed=20240101)
    return np.clip(np.asarray(r.render_frame(0), np.float32), 0, 1)


REF = q
def fixed_measure(img, left=270, right=275, off=24, min_contrast=0.02):
    """
    Measure the pole at **fixed columns taken from the source**, and report whether it is still visible there.

    The location must not be re-decided per frame: asking a detector to find the pole in a picture the mechanisms have
    changed makes the measurement follow the thing being measured.  So the columns and the surround offset come from
    the source once, and each frame is asked a narrower question -- **is the band still brighter than its surround, and
    what colour is it**.  A frame where it is not is reported as such rather than measured somewhere else.
    """
    lum = img @ COEF
    Hh = img.shape[0]
    # below the core, as before: the brightest patch's row
    k = max(5, Hh // 40)
    box = np.lib.stride_tricks.sliding_window_view(lum, (k, k)).mean(axis=(2, 3))
    cy, _ = np.unravel_index(int(np.argmax(box)), box.shape)
    cy += k // 2
    start = min(Hh - 2, cy + 8)
    rows = np.arange(start, Hh)
    c0, c1 = max(0, left - off), max(1, left - 3)
    d0, d1 = min(img.shape[1] - 1, right + 3), min(img.shape[1], right + off)
    pole_lum = lum[rows][:, left:right + 1].mean(axis=1)
    sur_lum = np.concatenate([lum[rows][:, c0:c1], lum[rows][:, d0:d1]], axis=1).mean(axis=1)
    keep = pole_lum > sur_lum + min_contrast
    if keep.sum() < 10:
        return None, int(keep.sum()), len(rows)
    sel = rows[keep]
    pole = float(np.median([float((img[q, left:right + 1, 2] - img[q, left:right + 1, 0]).mean())
                            for q in sel]))
    sur = float(np.median([float(np.concatenate([
        img[q, c0:c1, 2] - img[q, c0:c1, 0],
        img[q, d0:d1, 2] - img[q, d0:d1, 0]]).mean()) for q in sel]))
    con = float(np.median(pole_lum[keep] - sur_lum[keep]))
    return dict(pole=pole, surround=sur, excess=pole - sur, contrast=con, n=int(keep.sum())), \
        int(keep.sum()), len(rows)


print()
print("   RE-MEASURED, with the geometry FIXED from the source (columns 270..275, surround +/-24)")
print("   the detector's only job is the visibility check: does the band still stand above its surround?")

print("   %-40s %-11s %-11s %-11s %-11s %s"
      % ("setting", "pole", "surround", "EXCESS", "contrast", "visible rows"))
q_src, ns, tot = fixed_measure(PIC)
print("   %-40s %-+11.4f %-+11.4f %-+11.4f %-11.4f %d of %d  <- THE SOURCE"
      % ("", q_src["pole"], q_src["surround"], q_src["excess"], q_src["contrast"], ns, tot))
print()

def fline(label, cam):
    qq, n2, t2 = fixed_measure(run(cam))
    if qq is None:
        print("   %-40s NOT VISIBLE (%d of %d rows) -- the pole is buried here" % (label, n2, t2))
        return None
    print("   %-40s %-+11.4f %-+11.4f %-+11.4f %-11.4f %d of %d"
          % (label, qq["pole"], qq["surround"], qq["excess"], qq["contrast"], n2, t2))
    return qq

fline("the reproduction, source_glow 0.04", dict(CHOSEN, source_glow=0.04))
print()
for label, key, values in (
    ("source_glow", "source_glow", (0.0, 0.01, 0.02, 0.04, 0.15)),
    ("halation", "halation", (0.0, 1.0)),
    ("halation_source_colour", "halation_source_colour", (1.0,)),
    ("shadow_depth", "shadow_depth", (0.25,)),
):
    for v in values:
        fline("%s = %.2f" % (label, v), dict(dict(CHOSEN, source_glow=0.04), **{key: v}))
    print()
print("   the source's excess is %+.4f; the pole is 'buried' when the band no longer clears its surround." % q_src["excess"])

print()
print("   RE-MEASURED WITH THE DETECTOR THAT PASSED THE CONTROL")
print("   %-40s %-14s %-11s %-11s %-11s %s"
      % ("setting", "band", "pole", "surround", "EXCESS", "contrast"))
def line(label, cam):
    qq = measure(run(cam), search=WIN)
    if qq is None:
        print("   %-40s NOT FOUND" % label)
        return None
    print("   %-40s %-14s %-+11.4f %-+11.4f %-+11.4f %.4f"
          % (label, "%d..%d" % (qq["band"][0], qq["band"][1]), qq["pole"], qq["surround"],
             qq["excess"], qq["contrast"]))
    return qq

print("   %-40s %-14s %-+11.4f %-+11.4f %-+11.4f %.4f"
      % ("THE SOURCE", "%d..%d" % (REF["band"][0], REF["band"][1]), REF["pole"],
         REF["surround"], REF["excess"], REF["contrast"]))
print()
q0 = line("the reproduction (source_glow 0.04)", dict(CHOSEN, source_glow=0.04))
print()
for label, key, values in (
    ("source_glow", "source_glow", (0.0, 0.02, 0.15, 0.30)),
    ("halation_source_colour", "halation_source_colour", (0.5, 1.0)),
    ("shadow_depth", "shadow_depth", (0.25, 0.5)),
    ("halation", "halation", (0.0, 1.0)),
):
    for v in values:
        line("%s = %.2f" % (label, v), dict(dict(CHOSEN, source_glow=0.04), **{key: v}))
    print()
print("   the reference's excess is %+.4f; a setting is closer to it the nearer its excess is to that." % REF["excess"])
