"""
Measure what the real `_separate` returns now, since the tail script simulates the code rather than calling it.

**Why the previous reading could not settle it.**  The statement-by-statement script **re-implements** the function's
tail with its own local variables, so it still computes the residue from the post-`cross_color` band -- i.e. it simulates
the *old* code and reports 0.777 whatever the file says.  **A script that reproduces a function instead of calling it
cannot verify a change to that function**, which is the same fault as rebuilding the composite by hand one round earlier.

**So the real function is called and its return measured**, with the fix in place, and the same figure is printed beside
what the stage produced before the fix (0.802) and what the input carried (1.083).
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


# capture the composite the real stage receives
captured = {}
o_yc = SG.yc_separate


def yc(comp, *a, **kw):
    captured.setdefault("comp", np.array(comp, copy=True))
    return o_yc(comp, *a, **kw)


SG.yc_separate = yc
import atsim.stages.player as PL                                              # noqa: E402
PL.yc_separate = yc

#: the stage's own return value, and the value of `cross_luma` it used
measured = {}
o_sep = PlayerStage._separate


def sep(self, comp, buf):
    y, i, q = o_sep(self, comp, buf)
    measured["ret"] = float(np.clip(y, 0, 1).std())
    measured["luma"] = float(self.cross_luma)
    return y, i, q


PlayerStage._separate = sep

layers = {"camera": dict(NEUTRAL),
          "format": {"chroma_bandwidth": 16.0, "luma_bandwidth": 24.0},
          "player": {"chroma_bandwidth": 16.0, "luma_bandwidth": 24.0},
          "codec": {"dv_enabled": 0.0}, "display": {"enabled": 0.0}}
rr = R.Renderer(One(ramp), W, H, 25.0, merge_values(BASE, layers), seed=1)
out = np.clip(np.asarray(rr.render_frame(0), np.float32), 0, 1)

print("   === the REAL function's return, with the fix in place ===")
print("      the composite arriving            %.4f  (%.3f of the ramp)"
      % (float(np.clip(captured["comp"], 0, 1).std()), float(np.clip(captured["comp"], 0, 1).std()) / RAMP))
print("      `_separate` returns               %.4f  (%.3f of the ramp)   [it was 0.802 before the fix]"
      % (measured["ret"], measured["ret"] / RAMP))
print("      cross_luma in force               %.2f" % measured["luma"])
print("      the chain's output                %.4f  (%.3f of the ramp)   [it was 0.800 before the fix]"
      % (float((out @ COEF).std()), float((out @ COEF).std()) / RAMP))
print()
d_sep = measured["ret"] / RAMP - 0.802
d_chain = float((out @ COEF).std()) / RAMP - 0.800
if abs(d_sep) < 0.01:
    print("   `_separate` is UNCHANGED, so the fix did not alter what the function computes -- the residue it now reads")
    print("   is close enough to the old one that the term's contribution is the same.  The 37% the statement-by-statement")
    print("   script showed is therefore a property of ITS reconstruction, not of the real call.")
elif d_chain > 0.05:
    print("   the chain recovered by %+.3f, so the fix works and the compression was that line." % d_chain)
else:
    print("   `_separate` moved by %+.4f but the chain by %+.4f -- so the change is real and is being cancelled" % (d_sep, d_chain))
    print("   downstream, which means the compression the chain shows comes from somewhere else as well.")
PlayerStage._separate = o_sep
SG.yc_separate = o_yc
