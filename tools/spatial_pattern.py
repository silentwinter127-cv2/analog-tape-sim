"""
A known-answer signal with spatial chroma structure, chosen so a per-row displacement cannot confound it.

**The lesson from the last three attempts.** A ramp was unusable because `horizontal_shift` changes a row's mean by
`d * slope`, so the answer moved with the operation. A constant has no spatial structure, so it exercises none of the
path that is actually suspected. What is needed is a pattern that has structure **and** is invariant under the
displacements the chain applies.

**An even, symmetric stripe pattern is exactly that.** With a whole number of periods across the frame and the
pattern centred, every row's mean is the same value regardless of how far the row is shifted horizontally -- shifting
a periodic pattern changes *where* the stripes are, not *how much* of the row each level covers. So:

    I is a vertical square wave, constant down every column
    -> a correct recovery has the same row mean on every row, **and still does after a shift**
    -> any row-to-row variation is error, with nothing else it could be

`Q` stays zero and luma stays constant, so nothing else in the picture can supply the variation. This is the first
test in the series whose answer survives every operation in the chain.
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from atsim.core.params import merge_values                                   # noqa: E402
from atsim.core.pipeline import FrameBuffer                                  # noqa: E402
from atsim.core.profiles import FACTORY_PRESETS, SCHEMA                      # noqa: E402
from atsim.core.signal import (SignalFrame, composite_encode,                # noqa: E402
                               horizontal_shift)
from atsim.render.renderer import Renderer                                   # noqa: E402

H, W = 240, 320
BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])


class Dummy:
    def __init__(self):
        self.width, self.height, self.fps, self.frame_count = W, H, 25.0, 3

    def frame(self, index):
        return np.full((H, W, 3), 0.5, np.float32)


LAYERS = {
    "camera": {"exposure": 0.0, "color_cast": 0.0, "halation": 0.0, "ccd_smear": 0.0,
               "chroma_aberration": 0.0, "lens_flare": 0.0, "spike_spread": 0.0,
               "flare_defocus": 0.0, "longitudinal_ca": 0.0, "axis_offset": 0.0,
               "overflow_drain": 0.0, "lens_fog": 0.0, "lens_wear": 0.0,
               "bloom_amount": 0.0, "dead_sites": 0.0},
    "format": {"color_noise": 0.0, "cross_color": 0.0, "cross_luma": 0.0, "edge_boost": 0.0,
               "chroma_bandwidth": 3.0, "luma_bandwidth": 6.0},
    "player": {"color_noise": 0.0, "cross_color": 0.0, "cross_luma": 0.0,
               "chroma_bandwidth": 3.0, "luma_bandwidth": 6.0, "tbc_enabled": 0.0},
    "codec": {"dv_enabled": 0.0},
    "tape": {},
    "display": {"enabled": 0.0},
}

p = merge_values(BASE, LAYERS)
r = Renderer(Dummy(), W, H, 25.0, p, seed=5)
r._build_chain()
std = r.std
st = r.stage["player"]
y = np.full((H, W), 0.5, np.float32)

# an even number of periods, so a shift cannot change a row's mean
PERIODS = 8
xs = np.arange(W)
square = np.where(((xs // (W // (2 * PERIODS))) % 2) == 0, 1.0, -1.0).astype(np.float32)
print("   the pattern: %d periods across %d px, so a row's mean is set by the duty cycle, not the phase"
      % (PERIODS, W))
print("   a horizontal shift moves the stripes without changing the mean: %s"
      % np.allclose(horizontal_shift(square[None, :], np.array([1.7], np.float32))[0].mean(),
                    square.mean(), atol=1e-6))
print()

LEVEL = 0.30
i_true = (square * LEVEL)[None, :].repeat(H, axis=0).astype(np.float32)
q_true = np.zeros((H, W), np.float32)
print("   the designed signal: I a +-%.2f square wave, constant down every column; Q zero" % LEVEL)
print("   so a correct recovery has the same row mean on every row: %.6f" % float(i_true.mean(axis=1).mean()))


def ripple(a):
    x = np.asarray(a, np.float32)
    return float(np.abs(np.diff(x.mean(axis=1))).mean())


comp = np.asarray(composite_encode(y, i_true, q_true, std), np.float32)
buf = FrameBuffer(np.zeros((H, W, 3), np.float32), comp, 1, 0.04, std, {"field": 0})
sep = st._separate(comp, buf)
i_got = np.asarray(sep[1], np.float32)
q_got = np.asarray(sep[2], np.float32)

print()
print("   %-40s %-16s %-16s" % ("measurement", "I", "Q"))
print("   %-40s %-16.6f %-16.6f" % ("row-to-row ripple of the row means", ripple(i_got), ripple(q_got)))
print("   %-40s %-16.6f %-16.6f" % ("row-to-row ripple of the DESIGNED row means",
                                    ripple(i_true), ripple(q_true)))
print("   %-40s %-16.6f %-16.6f" % ("mean absolute column error",
                                    float(np.abs(i_got - i_true).mean()),
                                    float(np.abs(q_got - q_true).mean())))
print("   %-40s %-16.6f %-16.6f" % ("peak-to-peak of the recovered row means",
                                    float(np.ptp(i_got.mean(axis=1))),
                                    float(np.ptp(q_got.mean(axis=1)))))
print()
print("   the designed pattern's peak-to-peak is %.6f, so the recovered modulation is %.2f%% of it"
      % (float(np.ptp(i_true[0])), 100.0 * float(np.ptp(i_got[0])) / float(np.ptp(i_true[0]))))
print("   (a square wave through a 0.75 MHz chroma band should come back reduced and rounded -- the question is")
print("    whether it comes back ROW-VARYING, which nothing in the design allows)")
