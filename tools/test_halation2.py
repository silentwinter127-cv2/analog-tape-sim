"""
Test the colour inheritance with sources that actually cross the highlight threshold.

**The earlier test's fault, now diagnosed.** The highlight mask is `hot = clip((lum - 0.72) / 0.28, 0, None)`, so a
source must exceed **luma 0.72** to contribute anything. The warm patch used had luma **0.6914** -- below it -- so the
mechanism correctly did nothing there and the mean was dominated by the blue line. **A mechanism that never fires
cannot be judged**, and the fix is a source that is actually bright, which is what a blown lamp is:

    warm (1.00, 0.62, 0.25)          luma 0.6914   below the threshold
    warm-bright (1.00, 0.85, 0.55)   luma 0.8607   above it  <- the one to use
    white (1.00, 1.00, 1.00)         luma 1.0000   above it
    blue-white (0.75, 0.92, 1.00)    luma 0.8783   above it

**The two properties being established**, both by measurement:

  * at `halation_source_colour = 0` the result is **bit-identical** to the old body, so nothing else in the model
    moves and the change is opt-in;
  * at 1.0 a **blue-white** source's halo stays blue-white and a **warm** source's halo stays warm, so the halo stops
    dividing away a small chroma difference -- which is the property that was losing the pole's blue.
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from atsim.core.signal import lowpass                                      # noqa: E402

LUMA = np.array([0.299, 0.587, 0.114], np.float32)


def halation(rgb, a, source_colour):
    """`CameraStage._halation`, both forms, so old and new can be compared on one input."""
    if a < 1e-4:
        return rgb
    lum = rgb @ LUMA
    hot = np.clip((lum - 0.72) / 0.28, 0.0, None)
    core = lowpass(hot, 2.5 + 3.5 * a)
    veil = lowpass(hot, 12.0 + 20.0 * a)
    peak = float(core.max())
    if peak > 1e-6:
        core = core / peak
    coverage = float(np.clip(hot.mean() * 14.0, 0.0, 1.0))
    halo = np.clip(0.80 * core + 0.32 * coverage * veil, 0.0, 1.6)
    lens = np.array([1.05, 0.88, 1.05], dtype=np.float32)
    if source_colour <= 0.0:
        tint = lens
    else:
        mass = float(hot.sum())
        if mass > 1e-6:
            mean_rgb = (rgb * hot[..., None]).reshape(-1, 3).sum(axis=0) / mass
            g = float(mean_rgb[1])
            src = np.clip(mean_rgb / g, 0.55, 1.75).astype(np.float32) if g > 1e-4 \
                else np.ones(3, np.float32)
        else:
            src = np.ones(3, np.float32)
        tint = (1.0 - source_colour) * lens + source_colour * src
    return (rgb + (a * 0.52) * halo[..., None] * tint[None, None, :]).astype(np.float32)


def halo_tint(rgb, a, sc):
    """The colour actually added, as a mean over the region the halo reaches."""
    out = halation(rgb.copy(), a, sc)
    added = out - rgb
    return float(added[..., 2].mean() - added[..., 0].mean())


H, W = 300, 200
print("   the colour the halo actually adds (B-R), by source colour and by setting")
print()
print("   %-34s %-12s %-14s %-14s %s"
      % ("the blown source", "its own B-R", "halation 0.0", "halation 1.0", "does it follow?"))
CASES = [
    ("blue-white (0.75, 0.92, 1.00)", (0.75, 0.92, 1.00)),
    ("white (1.00, 1.00, 1.00)", (1.00, 1.00, 1.00)),
    ("warm-bright (1.00, 0.85, 0.55)", (1.00, 0.85, 0.55)),
    ("sodium (1.00, 0.72, 0.30)", (1.00, 0.72, 0.30)),
]
for label, rgb in CASES:
    a = np.zeros((H, W, 3), np.float32)
    a[..., :] = 0.03
    a[60:140, 60:140] = np.array(rgb, np.float32)
    own = float(np.array(rgb, np.float32)[2] - np.array(rgb, np.float32)[0])
    t0 = halo_tint(a, 0.25, 0.0)
    t1 = halo_tint(a, 0.25, 1.0)
    follows = "yes" if (own == 0.0 or (own > 0) == (t1 > t0)) else "NO"
    print("   %-34s %-+12.4f %-+14.5f %-+14.5f %s" % (label, own, t0, t1, follows))

print()
print("   and the safety property, on each of those sources:")
for label, rgb in CASES:
    a = np.zeros((H, W, 3), np.float32)
    a[..., :] = 0.03
    a[60:140, 60:140] = np.array(rgb, np.float32)
    # the old body, written out independently
    hot = np.clip((a @ LUMA - 0.72) / 0.28, 0.0, None)
    core = lowpass(hot, 2.5 + 3.5 * 0.25)
    core = core / max(1e-6, float(core.max()))
    veil = lowpass(hot, 12.0 + 20.0 * 0.25)
    cov = float(np.clip(hot.mean() * 14.0, 0.0, 1.0))
    halo = np.clip(0.80 * core + 0.32 * cov * veil, 0.0, 1.6)
    old = (a + (0.25 * 0.52) * halo[..., None]
           * np.array([1.05, 0.88, 1.05], np.float32)[None, None, :]).astype(np.float32)
    print("      %-34s new(0.0) identical to the old body: %s"
          % (label, np.array_equal(halation(a.copy(), 0.25, 0.0), old)))

print()
print("   and the pole case specifically -- a blue-white thin line under the halo:")
a = np.zeros((H, W, 3), np.float32)
a[..., :] = 0.03
a[20:280, 98:102] = np.array([0.75, 0.92, 1.00], np.float32)
seg = a[120:280, 98:102].reshape(-1, 3).mean(axis=0)
print("      the line's own hue B-R %+.4f" % float(seg[2] - seg[0]))
for sc in (0.0, 0.5, 1.0):
    out = halation(a.copy(), 0.25, sc)
    s = out[120:280, 98:102].reshape(-1, 3).mean(axis=0)
    print("      source_colour %.1f -> the line B-R %+.4f, added luma %+.5f"
          % (sc, float(s[2] - s[0]), float((out @ LUMA).mean() - (a @ LUMA).mean())))
print()
print("   a thin line is dominated by its own pixels in the mask, so its tint barely moves -- the mechanism is")
print("   aimed at the **surround** of a bright source, which is where a small difference gets divided away.")
