"""
Re-verify `accumulate`'s noise reduction with a metric that means the same thing for both mechanisms.

**Why the previous check could not settle it.**  It measured the spread across three *adjacent* rendered frames, and
`read_noise` seeds from the **frame index** -- so successive frames carry different noise realisations and an accumulated
frame overlaps its neighbours.  **The metric therefore measured different things depending on where the window sat**, and
the first version of the slow-shutter change made `accumulate` inert without the metric noticing: measured, temporal
noise at `accumulate = 8` was 0.07968 against a single frame's 0.07968, no reduction at all, where 1/sqrt(8) = 0.354 is
what Super NightShot exists to produce.

**The metric that does mean the same thing** holds the picture and the mechanism fixed and varies only the noise: **the
same frame index rendered at several seeds**, with a source whose content changes with the frame so that the seed is what
moves.  `read_noise`'s seed is `f(index)`, so the source must differ per index for the noise to differ -- **a still
source gives identical noise at every seed, which is itself the reason the earlier metric was confounded.**

**So the source alternates between two frames on a period the window crosses**, and the ensemble is taken over indices
rather than over seeds.  **The prediction, stated before measuring:** with `accumulate = 8` the spread across indices
should fall to about **1/sqrt(8) = 0.354** of the single-frame value, because the signal adds coherently and the
uncorrelated noise in quadrature.  **A value near 1.0 means it does not reduce anything.**
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from atsim.core.params import merge_values                                    # noqa: E402
from atsim.core.profiles import FACTORY_PRESETS, SCHEMA                       # noqa: E402
from atsim.render.renderer import Renderer                                    # noqa: E402

COEF = np.array([0.299, 0.587, 0.114], np.float32)
W, H = 320, 240


class Flicker:
    """
    A source that changes every few frames, so `read_noise`'s frame-indexed seed gives a different draw each time.

    **A still source is what made the earlier check meaningless**: with the same content at every index, the noise is
    the same at every index too, and any statistic over indices measures the content rather than the noise.
    """

    height, width = H, W
    fps = 25.0
    frame_count = 400

    def frame(self, i: int) -> np.ndarray:
        a = np.full((H, W, 3), 0.30, np.float32)
        a[60:180, 40:280] = 0.45 + 0.06 * ((int(i) % 3) - 1)
        a[20:40, 20:60] = 0.75
        return a


BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])
OFF = {"halation": 0.0, "ccd_smear": 0.0, "flare_defocus": 0.0, "lens_flare": 0.0,
       "overflow_drain": 0.0, "chroma_aberration": 0.0, "sensor_noise": 0.0, "noise_grain": 0.0,
       "fixed_pattern": 0.0, "spike_spread": 0.0, "motion_blur": 0.0, "stuck_fraction": 0.0,
       "bloom_threshold": 0.0, "overflow_width": 0.0, "sharpness": 1.0}


def ensemble(indices, **cam):
    """The spread across INDICES at one seed, which is a spread across noise realisations for this source."""
    layers = {"camera": dict(OFF, nv_green_mix=0.0, nv_illuminator=0.0, **cam),
              "format": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0},
              "player": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0},
              "codec": {"dv_enabled": 0.0}, "display": {"enabled": 0.0}}
    rr = Renderer(Flicker(), W, H, 25.0, merge_values(BASE, layers), seed=7)
    outs = []
    for idx, rgb in rr.render_frames(list(indices)):
        outs.append(np.clip(np.asarray(rgb, np.float32), 0, 1))
    st = np.stack(outs)
    g = np.clip(st @ COEF, 0, 1)
    return float(g.std(axis=0).mean())


IDX = list(range(60, 80))
print("   a source that flickers on a 3-frame period, so the frame-indexed noise seed gives a fresh draw each frame.")
print("   the ensemble is over indices, which for this source is an ensemble over noise realisations.")
print()
print("   === the accumulated noise, against the length of the accumulation ===")
print("   %-34s %-14s %-14s %s" % ("", "spread", "vs one frame", "1/sqrt(N) predicts"))
# **The MODE is named, not `nv_accumulate`.**  `mode_spec` builds the spec from the mode's preset on every frame, so
# passing `nv_accumulate` alongside `nv_mode = 1` is overridden to 1 and the accumulation never runs.  **That mistake
# produced a confident and completely wrong verdict in the previous round** -- "accumulate is inert, 1/2/4/8 all give
# 0.05014" -- and it is the same one made in the three-mode test before it.  Naming the mode is what the interface does;
# the parameter is an override, not the control.
one = None
for tag, cam, pred in (("mode 1 (nightshot, one frame)", dict(nv_mode=1.0), 1.0),
                       ("mode 2 (super, 8 frames)", dict(nv_mode=2.0), 8 ** -0.5),
                       ("mode 3 (slow shutter, 8 integrated)", dict(nv_mode=3.0), None)):
    s = ensemble(IDX, nv_read_noise=0.2, **cam)
    if one is None:
        one = s
    print("   %-34s %-14.5f %-14.3f %s" % (tag, s, s / one, ("%.3f" % pred) if pred else "-"))
print()
print("   **the prediction:** each doubling should multiply the spread by about 0.707, so an eight-frame accumulation")
print("   should land near 0.354 of one frame.  A ratio near 1.000 means the accumulation removes nothing.")
print()
print("   and the two mechanisms are separable: mode 3 integrates eight frames into ONE exposure, so its smear is")
print("   present while its noise falls for a different reason than mode 2's.")
