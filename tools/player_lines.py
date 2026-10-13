"""
Print y, i, q and the output rgb at each line of the player's `apply`, on a flat field.

The separation was measured in isolation and is accurate -- encoder `I=+0.3435, Q=-0.0230`, decoder
`I=+0.3426, Q=-0.0229`, and the rebuilt rgb `[0.7512 0.3594 0.0202]` against a source of
`[0.7519 0.3592 0.0184]`, which is 0.25%. Grey and white round trip exactly. So `_separate` is not the fault, and my
previous conclusion was wrong.

Yet the whole chain on the same flat field gives blue **0.0691 against 0.0184**, a factor of 3.8. So the damage is
somewhere between the separation and the returned rgb, and the lines in between are few: the chroma phase error, the
luma frequency response, the head-wear noise, the output gain, and the rebuild.

Each is therefore applied by hand, in order, on the same flat-field signal, printing the channel means after every
step. That attributes the change to a line rather than to a stage.
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


def rgb_of(y, i, q):
    return np.clip(np.asarray(SignalFrame(y, i, q).to_rgb(), np.float32), 0, 1).reshape(-1, 3).mean(axis=0)


print("   the source flat field: %s" % np.round(COLOUR, 4))
print()
for kind in (0.0, 1.0):
    p = merge_values(BASE, {"player": {"separation_kind": kind},
                            "display": {"enabled": 0.0},
                            "format": {"chroma_bandwidth": 3.0, "luma_bandwidth": 6.0},
                            "camera": {"exposure": 0.0}})
    r = Renderer(Still(), W, H, 25.0, p, seed=5)
    r._build_chain()
    st = r.stage["player"]
    # build the composite exactly as the format stage does, so the test starts where the player starts
    comp = None
    fmt = r.stage["format"]
    buf = FrameBuffer(FLAT.copy(), None, 1, 0.04, r.std, {"field": 0})
    try:
        enc = fmt.apply(FrameBuffer(FLAT.copy(), None, 1, 0.04, r.std, {"field": 0}))
        comp = enc.composite
    except Exception as exc:                                                # noqa: BLE001
        print("   format.apply raised %s" % exc)
    if comp is None:
        sig = SignalFrame.from_rgb(FLAT, r.std)
        comp = composite_encode(sig.y, sig.i, sig.q, r.std)
    buf = FrameBuffer(FLAT.copy(), comp, 1, 0.04, r.std, {"field": 0})

    print("   --- separation_kind %.0f ---" % kind)
    sep = st._separate(comp, buf)
    y, i, q = sep[0], sep[1], sep[2]
    print("      after _separate          rgb %s" % np.round(rgb_of(y, i, q), 4))
    i2, q2 = st._chroma_phase_error(i, q, buf)
    print("      after _chroma_phase_error rgb %s" % np.round(rgb_of(y, i2, q2), 4))
    y2 = st._frequency_response(y)
    print("      after _frequency_response rgb %s" % np.round(rgb_of(y2, i2, q2), 4))
    y3 = y2 * (1.0 + 0.25 * float(getattr(st, "output_gain", 0.0))) + 0.08 * float(getattr(st, "output_bias", 0.0))
    print("      after output gain         rgb %s" % np.round(rgb_of(y3, i2, q2), 4))
    out = st.apply(buf)
    rgb = out.rgb if out.rgb is not None else out.signal.to_rgb()
    print("      the stage's own apply     rgb %s   <- vs the source %s"
          % (np.round(np.asarray(rgb, np.float32).reshape(-1, 3).mean(axis=0), 4), np.round(COLOUR, 4)))
    print()
