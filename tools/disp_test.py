"""
Measure `disp`, the timebase displacement `_separate` applies by wrapping, since a large one inverts a correlation.

**Why this is the next thing to look at.**  The stage trace is unambiguous about *where*: every stage's output correlates
at **+0.999 or +1.000** with its input except **`player`, at -0.352** -- and the four corrections tried against
`split_composite`'s chroma division all failed, two of them repairing the polarity while destroying the contrast, so that
division is not the cause.  **Inside `apply` the remaining candidates are `_chroma_phase_error` and the two
`horizontal_shift` calls on `i` and `q`, and inside `_separate` the `disp` those shifts use.**

**And `horizontal_shift` is the natural suspect** because it **wraps**: a displacement of a large fraction of the frame
moves content from one edge to the other, and a wrapped picture correlates *negatively* with the original at frame scale
even though nothing has been negated.  **A phase error or a timebase error should be a few pixels or a few lines; a
displacement near half the width would look exactly like an inversion.**

**So `disp` is measured directly**, with its own statistics and its size relative to the frame, and the same for the
`_timebase_offset`'s parameters -- `tracking_error`, `line_jitter`, `micro_jitter`, `vertical_instability`, `walk_amplitude`
-- which is what feeds it.
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
from atsim.stages.player import PlayerStage                                   # noqa: E402

COEF = np.array([0.299, 0.587, 0.114], np.float32)
OBJ = (r"C:\Users\Administrator\.dsh\attachments\v1\objects")
PHOTO = os.path.join(OBJ, "69", "699c23315df037b01fa831a273364b41f99c28733fb1dc928b22d1e6c281818d")
W, H = 533, 400

src = np.asarray(Image.open(PHOTO).convert("RGB"), np.float32) / 255.0
src = np.asarray(Image.fromarray((np.clip(src, 0, 1) * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS),
                 np.float32) / 255.0


class One:
    height, width = H, W
    fps = 25.0
    frame_count = 1

    def frame(self, i):
        return src


BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])
print("   the parameters that feed the timebase offset:")
for k in ("tracking_error", "line_jitter", "micro_jitter", "vertical_instability", "walk_amplitude",
          "tape_stability", "tbc_enabled", "tbc_strength", "tbc_residual", "transport_jitter"):
    print("      %-24s %s" % (k, BASE["player"].get(k)))

#: capture the displacement the real call computes
got = {}
o_tb = PlayerStage._timebase_offset


def tb(self, buf, h):
    d = o_tb(self, buf, h)
    d = np.asarray(d, np.float32)
    got["disp"] = d
    return d


PlayerStage._timebase_offset = tb
o_sep = PlayerStage._separate


def sep(self, comp, buf):
    got["bw"] = int(comp.shape[1])
    return o_sep(self, comp, buf)


PlayerStage._separate = sep

layers = {"camera": {"nv_mode": 0.0, "halation": 0.0, "ccd_smear": 0.0, "flare_defocus": 0.0,
                     "lens_flare": 0.0, "overflow_drain": 0.0, "chroma_aberration": 0.0,
                     "sensor_noise": 0.0, "noise_grain": 0.0, "fixed_pattern": 0.0,
                     "spike_spread": 0.0, "motion_blur": 0.0, "stuck_fraction": 0.0,
                     "bloom_threshold": 0.0, "overflow_width": 0.0, "sharpness": 1.0},
          "format": {"chroma_bandwidth": 3.0, "luma_bandwidth": 6.0},
          "player": {"chroma_bandwidth": 3.0, "luma_bandwidth": 6.0},
          "codec": {"dv_enabled": 0.0}, "display": {"enabled": 0.0}}
rr = R.Renderer(One(), W, H, 25.0, merge_values(BASE, layers), seed=1)
rr.render_frame(0)
PlayerStage._timebase_offset = o_tb
PlayerStage._separate = o_sep

d = got.get("disp")
print()
print("   === the displacement the stage applies ===")
if d is None:
    print("      `_timebase_offset` was not called on this render")
else:
    print("      length %d (the picture is %d wide)" % (d.size, got.get("bw", W)))
    print("      min %.3f  max %.3f  mean %.3f  std %.3f  |max| %.3f"
          % (float(d.min()), float(d.max()), float(d.mean()), float(d.std()), float(np.abs(d).max())))
    print("      as a fraction of the width: mean %.4f, largest %.4f"
          % (float(np.abs(d).mean()) / max(1, got.get("bw", W)),
             float(np.abs(d).max()) / max(1, got.get("bw", W))))
    print()
    if float(np.abs(d).max()) > 0.25 * got.get("bw", W):
        print("      -> the displacement is a QUARTER OF THE FRAME OR MORE.  `horizontal_shift` wraps, so a shift")
        print("         that large moves content across the frame and inverts the correlation -- which is the fault")
        print("         being chased.  A timebase error should be a few pixels.")
    else:
        print("      -> the displacement is small, so wrapping is NOT the cause of the inversion and the suspect is")
        print("         `_chroma_phase_error` or the YIQ reconstruction instead.")
