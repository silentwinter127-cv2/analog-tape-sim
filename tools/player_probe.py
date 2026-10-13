"""
Measure what the player stage does to a picture on its own, since no single term in it accounts for the 30%.

**Where this stands.**  Neutralising **every** player term in turn moves the contrast after the stage by at most
**0.0002** -- `transport_jitter`, `cross_color`, `tbc_strength`, `ringing`, `preemphasis`, `output_gain`, all of them --
while the stage as a whole takes it from **0.1903 to 0.1337**, a loss of **0.0566**.  **So the loss is in something the
stage does unconditionally**, not in a defect term that can be switched off, and the same was true of the camera stage's
contribution.

**So the question changes from "which parameter" to "what fixed operation"**, and the way to answer it is to hand the
stage a picture whose contrast is known and see what comes back -- with the chain reduced to the player alone so nothing
else can be responsible.

**And the input is chosen to make the answer unambiguous**: a vertical ramp, whose contrast is exactly determined by its
own slope, so any reduction is measurable against a closed-form expectation rather than against another render.  If the
player compresses a linear ramp's range, the operation is a tone curve; if it leaves the range and changes the
distribution, it is a filter.
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
from atsim.render import renderer as R                                        # noqa: E402

COEF = np.array([0.299, 0.587, 0.114], np.float32)
W, H = 320, 240

#: a horizontal ramp from black to white: contrast 0.2887 by construction for a 0..1 ramp
ramp = np.zeros((H, W, 3), np.float32)
ramp[:] = np.linspace(0.0, 1.0, W, dtype=np.float32)[None, :, None]
RAMP_STD = float((ramp @ COEF).std())
print("   a black-to-white horizontal ramp at %dx%d: contrast %.4f (by construction)" % (W, H, RAMP_STD))

#: and a two-level checker, whose contrast is also known
chk = np.zeros((H, W, 3), np.float32)
chk[:, : W // 2] = 0.25
chk[:, W // 2:] = 0.75
CHK_STD = float((chk @ COEF).std())
print("   a 0.25/0.75 split field:            contrast %.4f" % CHK_STD)


class One:
    height, width = H, W
    fps = 25.0
    frame_count = 1

    def __init__(self, a):
        self.a = a

    def frame(self, i):
        return self.a


BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])


def through(stage_name, img, **over):
    """The picture after one named stage, with every other stage made neutral."""
    neutral = {"camera": {"nv_mode": 0.0, "halation": 0.0, "ccd_smear": 0.0, "flare_defocus": 0.0,
                          "lens_flare": 0.0, "overflow_drain": 0.0, "chroma_aberration": 0.0,
                          "sensor_noise": 0.0, "noise_grain": 0.0, "fixed_pattern": 0.0,
                          "spike_spread": 0.0, "motion_blur": 0.0, "stuck_fraction": 0.0,
                          "dynamic_range": 14.0, "sharpness": 1.0, "bloom_threshold": 0.0,
                          "overflow_width": 0.0},
               "format": {"chroma_bandwidth": 16.0, "luma_bandwidth": 24.0},
               "player": {},
               "codec": {"dv_enabled": 0.0},
               "display": {"enabled": 0.0}}
    if stage_name in ("format", "player"):
        neutral[stage_name] = dict(neutral[stage_name], **over) if over else neutral[stage_name]
    rr = R.Renderer(One(img), W, H, 25.0, merge_values(BASE, neutral), seed=1)
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
    return got


print()
print("   === the stage-by-stage contrast of a source whose answer is known ===")
for tag, img, ref in (("the ramp", ramp, RAMP_STD), ("the split field", chk, CHK_STD)):
    got = through("player", img)
    print("   %s (its own contrast %.4f)" % (tag, ref))
    prev = ref
    for name in ("camera", "format", "codec", "connector", "tape", "player", "display"):
        if name not in got:
            continue
        c = float((got[name] @ COEF).std())
        mark = "   <-- the step" if abs(c - prev) > 0.005 * ref else ""
        print("      after %-10s %.4f  (%+.4f, %.3f of the input)%s" % (name, c, c - prev, c / ref, mark))
        prev = c

print()
print("   === and the player's bandwidth terms, which the sweep proved are not it ===")
for tag, img, ref in (("the ramp", ramp, RAMP_STD),):
    for cbw, lbw in ((3.0, 3.0), (16.0, 24.0), (40.0, 60.0)):
        got = through("player", img, chroma_bandwidth=cbw, luma_bandwidth=lbw)
        c = float((got["player"] @ COEF).std())
        print("      %s at %.0f / %.0f MHz -> contrast %.4f (%.3f of the input)" % (tag, cbw, lbw, c, c / ref))
print()
print("   if the ramp's own contrast survives the player but the photograph's does not, the stage is not compressing")
print("   levels -- it is removing detail, and the photograph's 'contrast' is largely its detail rather than its range.")
