"""
What the NIGHT-VISION display chain does to a neutral grey, with every scenario-chroma path removed.

**The physical claim being tested.**  NightShot removes the IR-cut filter and then **routes the sensor's output as one
luminance onto a green display** -- the recorded picture is monochrome, and its colour comes from the *display*, not
from the scene.  A consequence that can be measured exactly: **the display's channel ratios must not depend on the
input at all.**  Feed it neutral grey, feed it a blue LED, feed it a saturated red -- the R:G:B ratio of the output
must be the same, because the scene's visible colour was discarded when the signal was treated as luminance.

**And that is the property to check, not to assume.**  `monochrome_green` carries a term
`out = mono + (rgb - lum) * r` which adds the *input's own chroma* back onto the projection.  If that term is doing
anything, the output ratio moves with the input -- which is the signature of a colour grade rather than a display, and
it is what makes the case photograph come out with its LEDs' blue in it.

The probe is deliberately a **flat field**, so the numbers are channel ratios with no scene structure in them, and it is
run on several inputs at once so the dependence is visible as a spread rather than argued.
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from atsim.codecs.nightvision import NightVisionSpec, monochrome_green    # noqa: E402

LUMA = np.array([0.299, 0.587, 0.114], np.float32)

#: Flat fields: a neutral grey, the case photograph's own blue-violet primary, its cyan and its magenta, plus a
#: saturated red and a warm grey.  If the display is a display, every one of these must come out with the SAME ratio.
FIELDS = {
    "neutral grey 0.40": (0.40, 0.40, 0.40),
    "warm grey (0.45,0.40,0.36)": (0.45, 0.40, 0.36),
    "the case's own primary": (0.267, 0.259, 0.396),
    "blue LED 0.40": (0.10, 0.12, 0.40),
    "cyan LED 0.40": (0.08, 0.40, 0.42),
    "magenta LED 0.40": (0.40, 0.08, 0.40),
    "saturated red 0.40": (0.40, 0.05, 0.05),
    "saturated green 0.40": (0.05, 0.40, 0.05),
}


def ratios(rgb: np.ndarray) -> tuple:
    r, g, b = (float(rgb[..., i].mean()) for i in range(3))
    return r, g, b, r / max(1e-9, g), b / max(1e-9, g)


def run(spec: NightVisionSpec, label: str) -> None:
    print("   --- %s ---" % label)
    print("      %-30s %-24s %-8s %-8s %s" % ("input flat field", "output R:G:B", "R/G", "B/G", ""))
    out_rows = []
    for tag, c in FIELDS.items():
        a = np.zeros((24, 24, 3), np.float32)
        a[...] = c
        o = np.asarray(monochrome_green(a, spec), np.float32)
        r, g, b, rg, bg = ratios(o)
        out_rows.append((rg, bg))
        print("      %-30s %-24s %-8.4f %-8.4f" % (tag, "%.4f:%.4f:%.4f" % (r, g, b), rg, bg))
    rgs = np.array([x[0] for x in out_rows])
    bgs = np.array([x[1] for x in out_rows])
    print("      **spread across the inputs**: R/G %.4f..%.4f (%.3fx)   B/G %.4f..%.4f (%.3fx)"
          % (rgs.min(), rgs.max(), rgs.max() / max(1e-9, rgs.min()),
             bgs.min(), bgs.max(), bgs.max() / max(1e-9, bgs.min())))
    print("      a DISPLAY must give one ratio; a colour grade gives a spread.")
    print()


print("=" * 118)
print("  IS THE GREEN VIEW A DISPLAY OR A GRADE?  -- the channel ratio must not depend on the input")
print("=" * 118)
print()
print("   the real NIGHTSHOT 1 frame's own ratio, for reference:  R/G 0.5434   B/G 0.8525")
print()

run(NightVisionSpec(mode=1.0, green_mix=1.0, chroma_residue=0.0),
    "chroma_residue = 0.00  (the scene's colour discarded)")
run(NightVisionSpec(mode=1.0, green_mix=1.0, chroma_residue=0.15),
    "chroma_residue = 0.15  (the shipped default)")
run(NightVisionSpec(mode=1.0, green_mix=1.0, chroma_residue=0.45),
    "chroma_residue = 0.45")
