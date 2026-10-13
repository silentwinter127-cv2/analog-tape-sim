"""
Measure the composite round trip on a flat colour, step by step, because no parameter accounts for the drift.

Every decoder setting was swept and almost none of them moved the output:
`chroma_bandwidth` from 0.4 to 3.0 changed nothing, nor did the vertical averaging, the luma bandwidth, the comb
depth, the crosstalk terms or the chroma noise. Only `separation_kind` and `output_gain` moved it, and both do so by
grossly wrong amounts rather than by a little.

So the fault is not a setting: **the conversion itself does not preserve a flat colour**, and a flat field is the
one input a colour transform is required to pass through unchanged. The suspect is the encode/decode pair, and the
way to find it is to measure the colour at each step of the round trip rather than at the end.

Each step is exercised on the same flat field:

  * `rgb -> Y/I/Q` and back, which is the matrix pair and must be an identity;
  * `Y/I/Q -> composite -> Y/I/Q`, which is the subcarrier modulation and must be an identity for a constant;
  * and the flat field's own colour, so the target is stated rather than assumed.
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from atsim.core.signal import SignalFrame, composite_encode, lowpass        # noqa: E402

H, W = 90, 120
COLOUR = np.array([0.7519, 0.3592, 0.0184], np.float32)      # the source's own mean, a sodium-lit orange
flat = np.zeros((H, W, 3), np.float32)
flat[:] = COLOUR

# the standard the model uses for this preset
from atsim.core.params import merge_values                                   # noqa: E402
from atsim.core.profiles import FACTORY_PRESETS, SCHEMA                      # noqa: E402
from atsim.render.renderer import Renderer                                   # noqa: E402


class Still:
    def __init__(self):
        self.width, self.height, self.fps, self.frame_count = W, H, 25.0, 3

    def frame(self, index):
        return flat


BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])
r = Renderer(Still(), W, H, 25.0, BASE, seed=5)
STD = r.std
print("   the standard: %s" % getattr(STD, "name", STD))
print("   the flat field: %s" % np.round(COLOUR, 4))


def means(a):
    return np.asarray(a, np.float32).reshape(-1, 3).mean(axis=0)


print()
print("   step                               channel means              blue vs source")
print("   %-34s %-26s %+.4f" % ("the flat field", np.round(means(flat), 4), 0.0))

# --- 1. rgb -> Y/I/Q and back
sig = SignalFrame.from_rgb(flat, STD)
back = np.clip(sig.to_rgb(), 0, 1)
m = means(back)
print("   %-34s %-26s %+.4f" % ("rgb -> Y/I/Q -> rgb", np.round(m, 4), float(m[2] - COLOUR[2])))

# --- 2. the composite itself: modulate and demodulate a constant
comp = composite_encode(sig.y, sig.i, sig.q, STD)
print("   %-34s min %.4f max %.4f" % ("the composite", float(comp.min()), float(comp.max())))
rec = SignalFrame(*( [sig.y, sig.i, sig.q] ))
m2 = means(np.clip(rec.to_rgb(), 0, 1))
print("   %-34s %-26s %+.4f" % ("Y/I/Q unchanged (a control)", np.round(m2, 4), float(m2[2] - COLOUR[2])))

# --- 3. what the player actually does: separate the composite, then rebuild
print()
print("   and the player's own separation of that composite, at its defaults:")
from atsim.stages.player import PlayerStage                                 # noqa: E402
from atsim.core.pipeline import FrameBuffer                                  # noqa: E402

for kind in (0.0, 1.0, 2.0, 3.0):
    p = merge_values(BASE, {"player": {"separation_kind": kind}, "display": {"enabled": 0.0},
                            "format": {"chroma_bandwidth": 3.0, "luma_bandwidth": 6.0}})
    rr = Renderer(Still(), W, H, 25.0, p, seed=5)
    rr._build_chain()
    # feed the composite directly, which is what the player reads
    sig2 = SignalFrame.from_rgb(flat, STD)
    cmp2 = composite_encode(sig2.y, sig2.i, sig2.q, STD)
    buf = FrameBuffer(flat.copy(), cmp2, 1, 0.04, STD, {"field": 0})
    out = rr.stage["player"].apply(buf)
    rgb = out.rgb if out.rgb is not None else out.signal.to_rgb()
    m3 = means(np.clip(rgb, 0, 1))
    print("   %-34s %-26s %+.4f" % ("separation_kind=%.0f" % kind, np.round(m3, 4),
                                    float(m3[2] - COLOUR[2])))

print()
print("   the source's own means: %s" % np.round(COLOUR, 4))
