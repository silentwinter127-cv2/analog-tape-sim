"""
Run the ramp with the in-function probe on, so every statement's own spread is measured in the real call.

**Why this and not another reconstruction.**  Two rebuilds have already been falsified: a hand-built composite read
1.000/1.000/1.000/1.165 where the real split reads 1.078/1.078/.../1.233, and a statement-by-statement reproduction of
`_separate`'s tail read **0.777** while the real call returns **0.802** -- unchanged even after the line it blamed was
modified.  **`_sep_probe` is inside the real function, so the numbers it prints are the ones the code actually
computes**, and it is switched on by an environment variable rather than by editing.

**And the comparison that decides the question** is between the second split's own spread and the value the function
returns, both now read from the same execution.
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["ATSIM_SEP_PROBE"] = "1"

from atsim.core.params import merge_values                                    # noqa: E402
from atsim.core.profiles import FACTORY_PRESETS, SCHEMA                       # noqa: E402
from atsim.render import renderer as R                                        # noqa: E402
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


returned = {}
o_sep = PlayerStage._separate


def sep(self, comp, buf):
    y, i, q = o_sep(self, comp, buf)
    returned["y"] = float(np.asarray(y, np.float32).std())
    return y, i, q


PlayerStage._separate = sep

layers = {"camera": dict(NEUTRAL),
          "format": {"chroma_bandwidth": 16.0, "luma_bandwidth": 24.0},
          "player": {"chroma_bandwidth": 16.0, "luma_bandwidth": 24.0},
          "codec": {"dv_enabled": 0.0}, "display": {"enabled": 0.0}}
print("   the ramp's own std %.4f" % RAMP)
print()
rr = R.Renderer(One(ramp), W, H, 25.0, merge_values(BASE, layers), seed=1)
out = np.clip(np.asarray(rr.render_frame(0), np.float32), 0, 1)
PlayerStage._separate = o_sep
print()
print("   `_separate` returned std %.5f  (%.3f of the ramp)" % (returned.get("y", float("nan")),
                                                               returned.get("y", float("nan")) / RAMP))
print("   the chain's output      %.5f  (%.3f of the ramp)" % (float((out @ COEF).std()),
                                                               float((out @ COEF).std()) / RAMP))
print()
print("   every line above comes from a probe inside the real function, so the statement whose spread differs from the")
print("   returned value is the one that does it -- and there is no reconstruction between them.")
