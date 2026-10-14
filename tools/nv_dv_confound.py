"""
Un-confound "DV on vs off": the first bench changed the bandwidth at the same time.

**The measurement that needs cleaning.**  `tools/nv_dv_test.py` reports, on the camcorder's own frame, a dark-band hue
spread of **0.156 with DV on** and **0.797 with DV off** -- a factor of five.  It would be easy to write that down as
"the DV codec destroys the fringe", and it would be wrong: that bench also switched `format.chroma_bandwidth` and
`player.chroma_bandwidth` from 0.75 MHz to 2.0 MHz at the same time, because the composite branch and the component
branch *should* differ in bandwidth.  **Two things changed together, so the difference belongs to neither of them.**

This file runs the four combinations -- {DV on, DV off} x {narrow, wide} -- so the two effects are separated:

    chroma bandwidth 0.75 MHz   the colour-under record's own sidebands (VHS / Video8 / Hi8)
    chroma bandwidth 2.00 MHz   a component path with no subcarrier to fit around

**Why it matters for the round's own report.**  If the five-fold difference is the *bandwidth*, then the model's
behaviour is exactly right and the number says nothing about DV; if it is the *codec*, the DV branch is introducing a
fringe that a component recorder should not.  REVIEW.md section 4.4 already records that the link order is only
PARTIAL precisely because chroma damage sits on both sides of the recorder, so which of the two it is decides whether
there is anything to fix.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from atsim.core.params import merge_values                                    # noqa: E402
from atsim.core.profiles import FACTORY_PRESETS, SCHEMA                       # noqa: E402
from atsim.render.renderer import Renderer                                    # noqa: E402
from nv_targets import crop_osd, measure                                      # noqa: E402

OUT = os.path.join(ROOT, "tools", "nightshot")
BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])
QUIET = dict(halation=0.0, ccd_smear=0.0, flare_defocus=0.0, lens_flare=0.0, spike_spread=0.0,
             overflow_drain=0.0, source_glow=0.0, shadow_depth=0.0, sensor_noise=0.0, noise_grain=0.0,
             fixed_pattern=0.0, motion_blur=0.0, sharpness=1.0)
MAX_EDGE = 1440


class Still:
    fps, frame_count = 25.0, 1

    def __init__(self, a):
        self.a = a
        self.height, self.width = a.shape[:2]

    def frame(self, i):
        return self.a


def load(path: str, frac: float) -> np.ndarray:
    a = np.asarray(Image.open(path).convert("RGB"), np.float32) / 255.0
    a = crop_osd(a, frac) if frac else a
    h, w = a.shape[:2]
    s = MAX_EDGE / float(max(h, w))
    if s < 1.0:
        a = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize(
            (max(2, int(round(w * s))), max(2, int(round(h * s)))), Image.LANCZOS), np.float32) / 255.0
    return a


def run(src: np.ndarray, dv: bool, bw: float) -> np.ndarray:
    layers = {"camera": dict(QUIET, nv_mode=1.0),
              "format": {"chroma_bandwidth": bw, "luma_bandwidth": 5.0 if dv else 3.0},
              "player": {"chroma_bandwidth": bw, "luma_bandwidth": 5.0 if dv else 3.0},
              "codec": {"dv_enabled": 1.0 if dv else 0.0}, "display": {"enabled": 0.0}}
    h, w = src.shape[:2]
    r = Renderer(Still(src), w, h, 25.0, merge_values(BASE, layers), seed=20240101)
    return np.clip(np.asarray(r.render_frame(0), np.float32), 0.0, 1.0)


def main() -> int:
    src = load(os.path.join(OUT, "ref_osd_1440.png"), 0.20)
    b_in = np.asarray(measure(src)["spreads"], np.float64)
    print("=" * 122)
    print("  IS THE FIVE-FOLD DIFFERENCE THE CODEC OR THE BANDWIDTH?")
    print("=" * 122)
    print()
    print("   the input's own spread (no chain):   %s" % "  ".join("%.3f" % v for v in b_in))
    print("   the real NIGHTSHOT 1 target:         0.193  0.147  0.216  0.302")
    print()
    print("   %-10s %-14s %s" % ("codec", "chroma bw", "the chain's own spread:  dark     mid      bright   brightest"))
    print("   " + "-" * 118)
    grid = {}
    for dv in (True, False):
        for bw in (0.75, 2.0):
            d = np.asarray(measure(run(src, dv, bw))["spreads"], np.float64) - b_in
            grid[(dv, bw)] = d
            print("   %-10s %-14s %s" % ("DV on" if dv else "DV off", "%.2f MHz" % bw,
                                         "  ".join("%+.3f" % x for x in d)))
    print()
    # the two main effects, each with the other held fixed
    print("   --- the effect of the CODEC, at each bandwidth ---")
    for bw in (0.75, 2.0):
        d = grid[(True, bw)] - grid[(False, bw)]
        print("       at %.2f MHz:  DV on minus DV off = %s" % (bw, "  ".join("%+.3f" % x for x in d)))
    print("   --- the effect of the BANDWIDTH, for each codec ---")
    for dv in (True, False):
        d = grid[(dv, 2.0)] - grid[(dv, 0.75)]
        print("       %-7s:  2.00 minus 0.75 MHz  = %s" % ("DV on" if dv else "DV off",
                                                             "  ".join("%+.3f" % x for x in d)))
    print()
    print("   **whichever of the two blocks above carries the large numbers is the one the five-fold figure belongs to.**")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
