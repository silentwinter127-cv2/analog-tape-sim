"""
Give the chroma recovery a signal whose answer is known exactly: I is a chosen ramp, Q is zero.

**Why this is the step that was missing.** Five hypotheses were refuted in turn, and the reason none of them could
settle the question is that every measurement compared the recovery against *another measurement* -- the encoder's own
ripple, the source's banding, the picture after the chain -- rather than against a value that is correct **by
construction**. With a designed I there is no ambiguity: the recovery either returns it or it does not, and any
per-row variation is error rather than signal.

**The signal is built directly**, not rendered from a picture, so the decode chain is the only thing between the
input and the output:

    Y      a constant plane, so nothing in the luma can masquerade as chroma
    I      a horizontal ramp, **constant down each column**, so a correct recovery is row-constant by construction
    Q      zero

A recovery that returns I row-varying on this input is wrong, and the amount tells how wrong -- with no reference
figure to guess at and no picture whose own content could supply the variation.

Then the same measurement on the photograph, where the answer *is* the recovery's own error, because the source has
no row-to-row chroma structure to begin with beyond what the subcarrier's phase per line requires.
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
from atsim.core.signal import SignalFrame, composite_encode                  # noqa: E402
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

# --- the designed signal: I constant down every column, Q zero
y = np.full((H, W), 0.5, np.float32)
i_true = np.tile(np.linspace(-0.4, 0.4, W, dtype=np.float32)[None, :], (H, 1))
q_true = np.zeros((H, W), np.float32)
print("   the designed signal: Y constant 0.5, I a horizontal ramp from -0.4 to +0.4, Q zero")
print("   so a correct recovery is row-constant BY CONSTRUCTION, and every per-row variation is error")
print()

comp = np.asarray(composite_encode(y, i_true, q_true, std), np.float32)
buf = FrameBuffer(np.zeros((H, W, 3), np.float32), comp, 1, 0.04, std, {"field": 0})
sep = st._separate(comp, buf)
i_got = np.asarray(sep[1], np.float32)
q_got = np.asarray(sep[2], np.float32)


def row_error(a, ref):
    """The per-row means against the reference's per-row means: zero for a correct, row-constant recovery."""
    return np.abs(a.mean(axis=1) - np.asarray(ref, np.float32).mean(axis=1))


print("   %-44s %-16s %-16s" % ("measurement", "I", "Q"))
print("   %-44s %-16.6f %-16.6f" % ("mean |column error| of the recovered plane",
                                    float(np.abs(i_got - i_true).mean()),
                                    float(np.abs(q_got - q_true).mean())))
print("   %-44s %-16.6f %-16.6f" % ("row-to-row ripple of the recovered row means",
                                    float(np.abs(np.diff(i_got.mean(axis=1))).mean()),
                                    float(np.abs(np.diff(q_got.mean(axis=1))).mean())))
print("   %-44s %-16.6f %-16.6f" % ("row-to-row ripple of the DESIGNED row means",
                                    float(np.abs(np.diff(i_true.mean(axis=1))).mean()),
                                    float(np.abs(np.diff(q_true.mean(axis=1))).mean())))
print()
print("   the designed signal's row means are constant, so the second row above is pure recovery error.")
print("   max per-row error in I: %.6f   (the designed range is 0.8, so this is %.3f%% of it)"
      % (float(row_error(i_got, i_true).max()),
         100.0 * float(row_error(i_got, i_true).max()) / 0.8))

print()
print("   and the same chain on a signal whose I is CONSTANT everywhere, which is the flattest case there is:")
i_flat = np.full((H, W), 0.30, np.float32)
comp2 = np.asarray(composite_encode(y, i_flat, q_true, std), np.float32)
buf2 = FrameBuffer(np.zeros((H, W, 3), np.float32), comp2, 1, 0.04, std, {"field": 0})
sep2 = st._separate(comp2, buf2)
i2 = np.asarray(sep2[1], np.float32)
print("      recovered I mean %.6f (designed 0.30), row-to-row ripple %.6f, max column error %.6f"
      % (float(i2.mean()), float(np.abs(np.diff(i2.mean(axis=1))).mean()),
         float(np.abs(i2 - 0.30).max())))
print("      a constant I must come back constant: the ripple above is the floor this recovery has.")
