"""
Test the auto-exposure loop as the camera stage's fixed compression, since it is a closed-loop normaliser.

**What the ramp proved.**  A **black-to-white horizontal ramp** -- the simplest signal there is, with no detail to lose
and no noise to gain -- comes out of the camera stage at **0.785 of its own contrast**, and out of the player at
**0.588**.  **A split field gives the same two ratios (0.787 and 0.607)**, so the operation is a fixed scaling of the
tonal range rather than anything content-dependent; and the player's bandwidths, swept at 3, 16 and 40 MHz, are
**bit-identical**.

**So the question is what normalises a picture unconditionally**, and the camera stage's own default table names the
candidate: **`ae_target = 0.46` with `ae_limit_stops = 2.0`**.  An automatic-exposure loop is by construction a
**closed-loop normaliser** -- it drives the picture toward a target level -- and **any device that holds a level constant
across a variety of inputs compresses their contrast to a similar value**, which is exactly what a ramp and a split field
coming out at the same ratio looks like.

**`ae_tau_s = 1.1`** is also in the table, and a loop with a 1.1 s time constant cannot settle within a single frame --
so if the compression is the loop, **the first frame is being pushed a long way toward the target**, and the measured
ratio is its first step.

**So the loop is switched off** by taking its limit to zero (and, separately, by setting the target so no correction is
asked for), and the contrast after the camera is measured against the ramp's own.  **If the 0.785 comes back to 1.0, the
loop owns the camera's half; and the same test then applies to the player.**
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from atsim.core.params import merge_values                                    # noqa: E402
from atsim.core.profiles import FACTORY_PRESETS, SCHEMA                       # noqa: E402
from atsim.render import renderer as R                                        # noqa: E402

COEF = np.array([0.299, 0.587, 0.114], np.float32)
W, H = 320, 240
ramp = np.zeros((H, W, 3), np.float32)
ramp[:] = np.linspace(0.0, 1.0, W, dtype=np.float32)[None, :, None]
RAMP_STD = float((ramp @ COEF).std())
print("   the ramp's own contrast: %.4f" % RAMP_STD)


class One:
    height, width = H, W
    fps = 25.0
    frame_count = 1

    def __init__(self, a):
        self.a = a

    def frame(self, i):
        return self.a


BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])
NEUTRAL = {"halation": 0.0, "ccd_smear": 0.0, "flare_defocus": 0.0, "lens_flare": 0.0,
           "overflow_drain": 0.0, "chroma_aberration": 0.0, "sensor_noise": 0.0, "noise_grain": 0.0,
           "fixed_pattern": 0.0, "spike_spread": 0.0, "motion_blur": 0.0, "stuck_fraction": 0.0,
           "dynamic_range": 14.0, "sharpness": 1.0, "bloom_threshold": 0.0, "overflow_width": 0.0,
           "nv_mode": 0.0}


def after(stage_name, img, **cam):
    layers = {"camera": dict(NEUTRAL, **cam),
              "format": {"chroma_bandwidth": 16.0, "luma_bandwidth": 24.0},
              "player": {"chroma_bandwidth": 16.0, "luma_bandwidth": 24.0},
              "codec": {"dv_enabled": 0.0}, "display": {"enabled": 0.0}}
    rr = R.Renderer(One(img), W, H, 25.0, merge_values(BASE, layers), seed=1)
    if rr.chain is None:
        rr._build_chain()
    got = {}
    for st in rr.chain.stages:
        fn = st.apply

        def make(stage, f):
            def wrapped(buf):
                out = f(buf)
                got[stage.name] = np.clip(np.asarray(getattr(out, "rgb", out), np.float32), 0, 1)
                return out
            return wrapped
        st.apply = make(st, fn)
    rr.render_frame(0)
    return got.get(stage_name)


print()
print("   === the auto-exposure loop, switched off ===")
print("   %-44s %-12s %-12s %s" % ("setting", "after camera", "of the ramp", "vs the default"))
base = float((after("camera", ramp) @ COEF).std())
print("   %-44s %-12.4f %-12.3f %s" % ("the camera's defaults", base, base / RAMP_STD, "-"))
for tag, over in (("ae_limit_stops = 0 (the loop cannot move)", {"ae_limit_stops": 0.0}),
                  ("ae_target = the ramp's own mean 0.5", {"ae_target": 0.5}),
                  ("ae_target = 0.5 and limit 0", {"ae_target": 0.5, "ae_limit_stops": 0.0}),
                  ("ae_tau_s very large (the loop is slow)", {"ae_tau_s": 1000.0})):
    a = after("camera", ramp, **over)
    c = float((a @ COEF).std())
    print("   %-44s %-12.4f %-12.3f %+.4f" % (tag, c, c / RAMP_STD, c - base))

print()
print("   === and the player's half, which no parameter in it explains ===")
pl = after("player", ramp)
print("      the player's defaults      -> %.3f of the ramp" % (float((pl @ COEF).std()) / RAMP_STD))
for tag, over in (("+ ae_limit_stops = 0", {"ae_limit_stops": 0.0}),
                  ("+ ae_target = 0.5", {"ae_target": 0.5})):
    got = after("player", ramp, **over)
    print("      %-26s -> %.3f of the ramp" % (tag, float((got @ COEF).std()) / RAMP_STD))
print()
print("   the camera's defaults read %.3f of the ramp and the player's %.3f, so the two stages together take a"
      % (base / RAMP_STD, float((pl @ COEF).std()) / RAMP_STD))
print("   full-range signal down to about %.2f -- which is the compression that makes every render look flat."
      % (float((pl @ COEF).std()) / RAMP_STD))
