"""
Does the slow shutter actually smear a moving subject, and is it a different mechanism from accumulating frames?

**`note.txt`'s fourth mechanism, and the distinction it draws:**

    "根据曝光时间对运动物体进行时间积分，而不是只复制几张半透明图层。快门越慢，运动物体在一次曝光期间移动得越远，
     残影就越明显。 这是 Super NightShot 与普通 NightShot 区别的重要来源之一。"

**The test is a subject that moves a known distance per frame**, which is the only way to see a trail: a still picture
cannot distinguish an integration from an average.  **The moving object is a bright bar crossing a dark field**, so its
trail is measurable as the **spread of its own smeared profile** -- a longer exposure drags it further.

**And the two mechanisms must be shown to differ, because they compute the same mean.**  `accumulate` sums separate
exposures (cleaner, no trail) and `shutter_frames` holds one open (trail).  **If the two produce identical pictures the
distinction is not implemented**, however plausible the parameters look -- which is the same test every mechanism in this
work has had to pass.

**The quantities:**

  * **the trail's length** -- how far the bar's brightness extends along its direction of travel, which should grow with
    `shutter_frames` and *not* with `accumulate`;
  * **the temporal noise** -- which should fall with `accumulate` and fall faster with a long integration;
  * and **the mean level**, which both should leave alone.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from atsim.core.params import merge_values                                    # noqa: E402
from atsim.core.profiles import FACTORY_PRESETS, SCHEMA                       # noqa: E402
from atsim.render.renderer import Renderer                                    # noqa: E402

COEF = np.array([0.299, 0.587, 0.114], np.float32)
W, H = 400, 300
SPEED = 26            # px per frame, so an 8-frame exposure drags the bar 208 px


class Mover:
    """A bright bar crossing a dark field at a known rate, so any trail is measurable."""

    height, width = H, W
    fps = 25.0
    frame_count = 400

    def frame(self, i: int) -> np.ndarray:
        a = np.full((H, W, 3), 0.06, np.float32)
        x = 40 + int(SPEED) * int(i)
        x0, x1 = x % (W + 60) - 30, (x % (W + 60)) + 6
        if x1 > 0 and x0 < W:
            a[:, max(0, x0):min(W, x1)] = 0.85
        return a


def trail_length(a: np.ndarray) -> float:
    """How far the bright bar's energy extends along its direction of travel, in pixels."""
    g = np.clip(np.asarray(a, np.float32) @ COEF, 0, 1)
    row = g[H // 2]
    prof = row - np.median(row)
    prof = np.clip(prof, 0, None)
    if prof.sum() < 1e-6:
        return 0.0
    # a rectangle of width L has energy L; a smeared bar is longer than the bar itself
    return float(prof.sum() / max(1e-9, prof.max()))


BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])
OFF = {"halation": 0.0, "ccd_smear": 0.0, "flare_defocus": 0.0, "lens_flare": 0.0,
       "overflow_drain": 0.0, "chroma_aberration": 0.0, "sensor_noise": 0.0, "noise_grain": 0.0,
       "fixed_pattern": 0.0, "spike_spread": 0.0, "motion_blur": 0.0, "stuck_fraction": 0.0,
       "bloom_threshold": 0.0, "overflow_width": 0.0, "sharpness": 1.0}


def run(frames_out, **cam):
    layers = {"camera": dict(OFF, nv_mode=1.0, nv_green_mix=0.0, nv_illuminator=0.0, **cam),
              "format": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0},
              "player": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0},
              "codec": {"dv_enabled": 0.0}, "display": {"enabled": 0.0}}
    rr = Renderer(Mover(), W, H, 25.0, merge_values(BASE, layers), seed=5)
    out = []
    for idx, rgb in rr.render_frames(frames_out):
        out.append(np.clip(np.asarray(rgb, np.float32), 0, 1))
    return out

IDX = [30, 31, 32]
print("   a bright bar crossing a dark field at %d px per frame; the exposure is measured at frames 30-32." % SPEED)
print()
print("   === the trail, against the length of the exposure ===")
print("   %-34s %-14s %-14s %s" % ("", "trail (px)", "luma", "temporal noise"))
base = None
for tag, cam in (("one frame (no integration)", dict(nv_shutter_frames=1.0)),
                 ("shutter 4 frames", dict(nv_shutter_frames=4.0)),
                 ("shutter 8 frames", dict(nv_shutter_frames=8.0)),
                 ("shutter 16 frames", dict(nv_shutter_frames=16.0)),
                 ("accumulate 8 (Super NightShot)", dict(nv_accumulate=8.0)),
                 ("accumulate 8 + shutter 8", dict(nv_accumulate=8.0, nv_shutter_frames=8.0))):
    stack = run(IDX, sensor_noise=0.0, noise_grain=0.0, **cam)
    last = stack[-1]
    tl = trail_length(last)
    lm = float(np.clip(last @ COEF, 0, 1).mean())
    tsd = float(np.stack(stack).std(axis=0).mean())
    if base is None:
        base = tl
    print("   %-34s %-14.1f %-14.4f %.5f" % (tag, tl, lm, tsd))
print()
print("   the bar itself is 36 px wide, so the un-integrated trail is its own width.")
print()
print("   === and with the mode's own read noise, which is what accumulation exists to reduce ===")
print("   %-34s %-14s %s" % ("", "temporal noise", "vs one frame"))
one = None
for tag, cam in (("one frame", dict(nv_shutter_frames=1.0)),
                 ("accumulate 8", dict(nv_accumulate=8.0)),
                 ("shutter 8", dict(nv_shutter_frames=8.0))):
    stack = run(IDX, sensor_noise=0.7, noise_grain=0.7, **cam)
    tsd = float(np.stack(stack).std(axis=0).mean())
    if one is None:
        one = tsd
    print("   %-34s %-14.5f %.3f" % (tag, tsd, tsd / max(1e-9, one)))
print()
print("   **`accumulate` should approach 1/sqrt(8) = 0.354, and a long exposure should be lower still** -- one exposure")
print("   collects the whole interval's light, so its noise does not add across the window the way separate")
print("   exposures' does.  A value near 1.0 would mean neither mechanism is reducing anything.")
