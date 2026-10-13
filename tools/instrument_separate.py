"""
Instrument `_separate` step by step on the known-answer signal, to find where the ripple appears.

**What is now established.** With a constant I the recovery returns a row-to-row ripple of **0.000649** and a mean of
0.299717 against a designed 0.30 -- so the recovery is accurate and that figure is its **error floor**. On the
photograph the recovered I ripples **0.006990**, which is **10.8x that floor**. So the photograph's figure is real
error, not signal, and five candidate causes have been refuted: the chroma band filter, the per-line phase error, the
luma frequency response, `composite_encode` and the time-base displacement.

**What the refutations leave.** Replacing `_separate` with a pass-through removed the banding, and that was withdrawn
as evidence because it also skips `remod_chroma` and the reconstruction. So those are the untested steps inside, and
`_separate` is short enough to instrument line by line:

    y_band, chroma_band = yc_separate(comp, ..., chroma_bandwidth_hz=window)
    i, q                = demod_chroma(chroma_band / gain, ...)
    i, q                = horizontal_shift(i, disp), ...
    comp_shifted        = y_band + remod_chroma(i, q, ...)
    y_band, chroma_band = yc_separate(comp_shifted, ..., comb_depth, chroma_bw, vert_blend)
    chroma_band         = chroma_band / gain
    ... cross-colour and cross-luma terms

Each is replayed by hand in that order, with the known-answer signal, and the ripple is printed after every step. The
step where it jumps is the answer, and the known-answer signal makes "jumps" mean something because the correct
value is **zero** at every step.
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
                               demod_chroma, horizontal_shift, remod_chroma, yc_separate)
from atsim.render.renderer import Renderer                                   # noqa: E402

H, W = 240, 320
BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])


class Dummy:
    def __init__(self):
        self.width, self.height, self.fps, self.frame_count = W, H, 25.0, 3

    def frame(self, index):
        return np.full((H, W, 3), 0.5, np.float32)


LAYERS = {
    "camera": {"exposure": 0.0, "color_cast": 0.0},
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
gain = 0.63

# the known-answer signal: I is a ramp, so a correct recovery is row-constant and its ripple is zero
y = np.full((H, W), 0.5, np.float32)
i_true = np.tile(np.linspace(-0.4, 0.4, W, dtype=np.float32)[None, :], (H, 1))
q_true = np.zeros((H, W), np.float32)
comp = np.asarray(composite_encode(y, i_true, q_true, std), np.float32)
buf = FrameBuffer(np.zeros((H, W, 3), np.float32), comp, 1, 0.04, std, {"field": 0})


def ripple(a):
    x = np.asarray(a, np.float32)
    return float(np.abs(np.diff(x.mean(axis=1))).mean())


print("   the known-answer signal: constant Y, I a ramp, Q zero -> a correct recovery ripples 0.000000")
print()
print("   %-46s %-16s %-16s" % ("step", "I ripple", "Q ripple"))

# --- the steps of _separate, replayed in order
disp = st._timebase_offset(buf, comp.shape[0])
window = float(np.clip(st.chroma_bw * 1.15, 0.35e6, 0.9e6))
y_band, chroma_band = yc_separate(comp, std, kind=st.separation_kind, comb_depth=0.0,
                                  chroma_bandwidth_hz=window)
i, q = demod_chroma(chroma_band / gain, std, i_bw_hz=1.5e6, q_bw_hz=1.5e6)
print("   %-46s %-16.6f %-16.6f" % ("after the first yc_separate + demod", ripple(i), ripple(q)))

i2 = horizontal_shift(i, disp)
q2 = horizontal_shift(q, disp)
print("   %-46s %-16.6f %-16.6f" % ("after horizontal_shift(disp)", ripple(i2), ripple(q2)))

comp_shifted = (horizontal_shift(y_band, disp) + remod_chroma(i2, q2, std, chroma_gain=gain)
                ).astype(np.float32)
y_b2, chroma_b2 = yc_separate(comp_shifted, std, kind=st.separation_kind,
                              comb_depth=st.comb_depth,
                              chroma_bandwidth_hz=st.chroma_bw,
                              vert_blend=st.chroma_vert_blend)
print("   %-46s %-16.6f %-16.6f" % ("after remod + the second yc_separate", ripple(i2), ripple(q2)))

i3, q3 = demod_chroma(chroma_b2 / gain, std, i_bw_hz=1.5e6, q_bw_hz=1.5e6)
print("   %-46s %-16.6f %-16.6f" % ("after demodulating the second split", ripple(i3), ripple(q3)))

print()
print("   and what the stage's own `_separate` returns, for comparison:")
sep = st._separate(comp, buf)
print("   %-46s %-16.6f %-16.6f" % ("the real _separate", ripple(sep[1]), ripple(sep[2])))

print()
print("   and the same replay with the SECOND split skipped, to isolate its contribution:")
comp_only = (horizontal_shift(y_band, disp) + remod_chroma(i2, q2, std, chroma_gain=gain)
             ).astype(np.float32)
print("   %-46s %-16.6f %-16.6f" % ("first split + remod only (no second split)",
                                    ripple(i), ripple(q)))
print()
print("   the second split's own output against the first's, so its contribution is visible:")
print("      first split  chroma_band ripple %.6f" % ripple(chroma_band))
print("      second split chroma_band ripple %.6f" % ripple(chroma_b2))
print("      difference %.6f" % (ripple(chroma_b2) - ripple(chroma_band)))
