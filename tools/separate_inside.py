"""
Capture the real `_separate` call's intermediates, since reconstructing its input by hand gave different numbers.

**Why the hand-built version is not trustworthy.**  Rebuilding the composite from the ramp and running the modulator
round trip by hand gave **1.000, 1.000, 1.000, 1.165** -- lossless and then slightly *expanded* -- while the stage's own
`_separate` takes the real signal from **1.083 to 0.802**.  And the hand-built run's last variant collapsed to
**0.0003**, which is not a compression but a wipe-out.  **So the composite the real stage receives is not the one built
from a YIQ round trip of the source**, and every figure from that reconstruction describes a different experiment.

**So nothing is rebuilt here.**  The stage's own `yc_separate`, `demod_chroma` and `remod_chroma` are wrapped, and the
real call's arguments and results are recorded -- which shows what the real composite actually looks like and which of
the four steps inside `_separate` takes the 26%.  **That is the same seam-by-seam method that found the loss in the first
place, applied one level deeper.**

**And the first thing to record is the composite's own statistics**, because a signal whose range is not [0, 1] -- or
whose mean is not where its nominal level is -- would explain a loss that no filter could produce.
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
import atsim.core.signal as SG                                                # noqa: E402
from atsim.stages.player import PlayerStage                                   # noqa: E402

COEF = np.array([0.299, 0.587, 0.114], np.float32)
W, H = 320, 240
ramp = np.zeros((H, W, 3), np.float32)
ramp[:] = np.linspace(0.0, 1.0, W, dtype=np.float32)[None, :, None]
RAMP = float((ramp @ COEF).std())

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


def stat(tag, arr):
    a = np.asarray(arr, np.float32)
    if a.ndim == 3:
        lum = np.clip(a @ COEF, 0, 1)
    else:
        lum = np.clip(a, 0, 1)
    print("   %-40s std %-9.4f (%.3f) mean %-9.4f min %-9.4f max %-9.4f"
          % (tag, float(lum.std()), float(lum.std()) / RAMP, float(lum.mean()),
             float(lum.min()), float(lum.max())))


#: wrap the three signal helpers so the stage's own calls are recorded
orig = {n: getattr(SG, n) for n in ("yc_separate", "demod_chroma", "remod_chroma")}
seen = {"comp": 0, "ys": 0}


def yc_separate(comp, *a, **kw):
    if seen["comp"] == 0 and seen["ys"] == 0:
        print("   --- the composite the stage actually receives, and its split ---")
        stat("the composite", comp)
        stat("  its kind/comb_depth/etc", comp)   # the same array, for the header
        print("      args: kind=%r comb_depth=%r" % (kw.get("kind"), kw.get("comb_depth")))
    y, c = orig["yc_separate"](comp, *a, **kw)
    seen["ys"] += 1
    stat("  after yc_separate #%d" % seen["ys"], y)
    return y, c


def demod_chroma(c, *a, **kw):
    i, q = orig["demod_chroma"](c, *a, **kw)
    stat("  after demod_chroma (the i channel)", i)
    return i, q


def remod_chroma(i, q, *a, **kw):
    r = orig["remod_chroma"](i, q, *a, **kw)
    stat("  after remod_chroma", r)
    return r


SG.yc_separate = yc_separate
SG.demod_chroma = demod_chroma
SG.remod_chroma = remod_chroma

#: the stage imports these names directly, so patch them where it looks them up
import atsim.stages.player as PL                                              # noqa: E402
for n in ("yc_separate", "demod_chroma", "remod_chroma"):
    if hasattr(PL, n):
        setattr(PL, n, {"yc_separate": yc_separate, "demod_chroma": demod_chroma,
                        "remod_chroma": remod_chroma}[n])

#: and the core signal module is where `_separate` imports them from inside the call
import atsim.core.signal as SG2                                               # noqa: E402
SG2.yc_separate = yc_separate
SG2.demod_chroma = demod_chroma
SG2.remod_chroma = remod_chroma

layers = {"camera": dict(NEUTRAL),
          "format": {"chroma_bandwidth": 16.0, "luma_bandwidth": 24.0},
          "player": {"chroma_bandwidth": 16.0, "luma_bandwidth": 24.0},
          "codec": {"dv_enabled": 0.0}, "display": {"enabled": 0.0}}

print("   === the ramp, with the stage's own signal calls recorded ===")
print("   the ramp's own std: %.4f" % RAMP)
print()
rr = R.Renderer(One(ramp), W, H, 25.0, merge_values(BASE, layers), seed=1)
rr.render_frame(0)
print()
print("   the numbers to compare with: the signal ARRIVES at 1.083 of the ramp and `_separate` RETURNS 0.802.")
print("   the step in the list above where the ratio falls to 0.80 is the operation.")
