"""
Revert the reorder, because limiting the composite destroys the subcarrier, and test the other candidate.

**The reorder failed, and instructively.** Applying `lowpass_hz` to the composite dropped its mean from 0.4378 to
**0.1920** and its std to 0.1352, and the player then recovered **I = 0.0000** -- the colour vanished and the frame
went grey. The reason is aliasing: at this working resolution the pixel rate is below the 3.58 MHz subcarrier, so the
composite's subcarrier sits at a folded frequency and a lowpass evaluated in Hz removes it as though it were
high-frequency detail. **The encoder's original order is the one that works.**

**So the fault is not the order, and the earlier reasoning that it was needs correcting.** The encoder limits the
baseband chroma before modulating, which preserves the subcarrier; the player then applies *the same limit again* to
the I and Q it has demodulated. Both limits are "correct" for what they model, and together they cost the chroma
twice:

    the encoder's baseband limit alone      composite retains 68.9% of ideal
    and the player's limit again            recovered I falls to 48% of the encoder's value

**The candidate now tested is that the player's limit is the redundant one.** In a real deck the chroma bandwidth is
a property of the recording, so it is applied once, on the way in; the player's job is to demodulate what the tape
actually carries. If that is right, removing the player's band limit should bring the recovered I back to the
encoder's own value without touching the format stage at all.

The test is the same one that found the fault: what the player recovers from the format stage's real composite,
against what the encoder was given.
"""
from __future__ import annotations

import ast
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FMT = os.path.join(ROOT, "atsim", "stages", "format.py")
BAK = os.path.join(ROOT, "tools", "_fmt_mv.py")

shutil.copyfile(BAK, FMT)
print("   reverted format.py to the working order")
src = open(FMT, encoding="utf-8").read()
assert "comp = lowpass_hz(comp" not in src, "the composite lowpass survived the revert"
assert "i = lowpass_hz(i, self.chroma_bw, rate, choice)" in src, "the baseband lowpass was not restored"
print("   confirmed: the baseband lowpass is back and the composite one is gone")

# --- now test whether the player's own limit is the redundant one, without editing anything yet
import sys
sys.path.insert(0, ROOT)
import numpy as np
from atsim.core.params import merge_values
from atsim.core.pipeline import FrameBuffer
from atsim.core.profiles import FACTORY_PRESETS, SCHEMA
from atsim.core.signal import SignalFrame, composite_encode
from atsim.render.renderer import Renderer

H, W = 90, 120
COLOUR = np.array([0.7519, 0.3592, 0.0184], np.float32)
BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])


class Still:
    def __init__(self):
        self.width, self.height, self.fps, self.frame_count = W, H, 25.0, 3

    def frame(self, index):
        a = np.zeros((H, W, 3), np.float32)
        a[:] = COLOUR
        return a


FLAT = np.asarray(Still().frame(0), np.float32)


def recovered(**player):
    p = merge_values(BASE, {"player": dict(player), "display": {"enabled": 0.0},
                            "camera": {"exposure": 0.0}, "codec": {"dv_enabled": 0.0}})
    r = Renderer(Still(), W, H, 25.0, p, seed=5)
    r._build_chain()
    st = r.stage["format"]
    enc = st.apply(FrameBuffer(FLAT.copy(), None, 1, 0.04, r.std, {"field": 0}))
    comp = enc.composite
    sig = SignalFrame.from_rgb(FLAT, r.std)
    want_i = float(np.asarray(sig.i, np.float32).mean())
    buf = FrameBuffer(FLAT.copy(), comp, 1, 0.04, r.std, {"field": 0})
    sep = r.stage["player"]._separate(comp, buf)
    got_i = float(np.asarray(sep[1], np.float32).mean())
    rgb = np.clip(np.asarray(SignalFrame(sep[0], sep[1], sep[2]).to_rgb(), np.float32), 0, 1)
    return want_i, got_i, rgb.reshape(-1, 3).mean(axis=0)


print()
print("   what the player recovers from the format stage's real composite")
print("   %-34s %-12s %-12s %-10s %s" % ("player settings", "encoder I", "recovered I", "ratio", "rgb"))
for label, kw in (
    ("as it is", {}),
    ("chroma_bandwidth = 6 MHz", {"chroma_bandwidth": 6.0}),
    ("separation_kind = 1", {"separation_kind": 1.0}),
):
    wi, gi, rgb = recovered(**kw)
    print("   %-34s %-12.4f %-12.4f %-10.1f%% %s"
          % (label, wi, gi, 100.0 * gi / max(1e-9, wi), np.round(rgb, 4)))
print()
print("   the source: %s" % np.round(COLOUR, 4))
