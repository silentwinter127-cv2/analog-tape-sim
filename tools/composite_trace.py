"""
Measure the COMPOSITE signal's contrast, because that is what the player reads and the trace never looked at it.

**The instrument's blind spot, found by reading the stage's first line.**  `player.apply` begins

    comp = buf.composite
    if comp is None:
        return buf

**so the stage acts on the *composite* signal -- which the format stage publishes -- and not on the RGB buffer.**  The
stage trace recorded `.rgb` after every stage and therefore reported **"format contributes 0.0000"**, because the format
stage's work goes into `composite` while `.rgb` travels past it untouched.  **So the trace was measuring a quantity the
player does not use**, and every conclusion drawn from "format contributes nothing" was about the wrong buffer.

**And the round trip is not the cause**: the YIQ conversion pair is exactly invertible here -- a ramp and a split field
both come back at **1.000 of their own contrast**, and `y` spans the full [0, 1].

**So the quantity to measure is the composite's own luminance contrast after each stage**, which means recording
`buf.composite` as well as `buf.rgb` in the trace.  That is the fix to the instrument, and it is what the next figure
comes from.
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
chk = np.zeros((H, W, 3), np.float32)
chk[:, : W // 2] = 0.25
chk[:, W // 2:] = 0.75

NEUTRAL = {"halation": 0.0, "ccd_smear": 0.0, "flare_defocus": 0.0, "lens_flare": 0.0,
           "overflow_drain": 0.0, "chroma_aberration": 0.0, "sensor_noise": 0.0, "noise_grain": 0.0,
           "fixed_pattern": 0.0, "spike_spread": 0.0, "motion_blur": 0.0, "stuck_fraction": 0.0,
           "sharpness": 1.0, "bloom_threshold": 0.0, "overflow_width": 0.0, "nv_mode": 0.0}
BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])


class One:
    height, width = H, W
    fps = 25.0
    frame_count = 1

    def __init__(self, a):
        self.a = a

    def frame(self, i):
        return self.a


def contrast_of_signal(sig):
    """The luminance contrast of a signal frame, whatever its own representation is."""
    if sig is None:
        return None
    try:
        rgb = np.clip(np.asarray(sig.to_rgb(), np.float32), 0, 1)
    except Exception:                                                          # noqa: BLE001
        return None
    return float((rgb @ COEF).std())


def trace(img, tag, ref):
    layers = {"camera": dict(NEUTRAL),
              "format": {"chroma_bandwidth": 16.0, "luma_bandwidth": 24.0},
              "player": {"chroma_bandwidth": 16.0, "luma_bandwidth": 24.0},
              "codec": {"dv_enabled": 0.0}, "display": {"enabled": 0.0}}
    rr = R.Renderer(One(img), W, H, 25.0, merge_values(BASE, layers), seed=1)
    if rr.chain is None:
        rr._build_chain()
    log = []
    for st in rr.chain.stages:
        fn = st.apply

        def make(stage, f):
            def wrapped(buf):
                out = f(buf)
                rgb = np.clip(np.asarray(getattr(out, "rgb", out), np.float32), 0, 1)
                comp = getattr(out, "composite", None)
                sig = out.meta.get("signal") if hasattr(out, "meta") else None
                log.append((stage.name, float((rgb @ COEF).std()),
                            contrast_of_signal(comp), contrast_of_signal(sig)))
                return out
            return wrapped
        st.apply = make(st, fn)
    rr.render_frame(0)
    print()
    print("   === %s (its own RGB contrast %.4f) ===" % (tag, ref))
    print("   %-12s %-12s %-14s %-14s %s" % ("after", "rgb std", "composite std", "signal std", "of the input"))
    for name, rgbstd, compstd, sigstd in log:
        print("   %-12s %-12.4f %-14s %-14s %.3f"
              % (name, rgbstd,
                 "n/a" if compstd is None else "%.4f" % compstd,
                 "n/a" if sigstd is None else "%.4f" % sigstd,
                 rgbstd / ref))
    return log


trace(ramp, "the black-to-white ramp", float((ramp @ COEF).std()))
trace(chk, "the 0.25/0.75 split field", float((chk @ COEF).std()))
print()
print("   `composite` and `signal` are what the format and player stages actually pass along, so a change that shows")
print("   in those columns and not in `rgb std` is work the trace had been blind to.")
