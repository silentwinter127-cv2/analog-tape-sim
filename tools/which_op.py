"""
Which of the two chroma operations costs the composite its amplitude, on a flat field.

`_limit_chroma` does two things before the subcarrier is ever generated: it lowpasses I and Q at `chroma_bandwidth`
and it averages them vertically by `chroma_vertical_averaging`. Then `composite_encode` modulates. Measured, the
format stage's composite has std **0.1057** against an ideal **0.1534**, so 31% of the chroma amplitude is gone -- and
the player, doing the same lowpass again, brings the recovered I down to **48%** of what the encoder was given.

**A lowpass does not change a constant, and a flat field's I and Q are constant**, so the horizontal limit is a
surprising suspect. The vertical averaging is the one that touches a constant field in a way that could matter if it
is applied along the wrong axis or with the wrong kernel. Both are therefore measured separately, on the same flat
field, by reading the composite's own standard deviation with each in turn.

That quantity -- the composite's std -- is the one that shows the loss at all: I/Q and a flat field's colour both
hide it, which is why every earlier sweep found nothing.
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
CLEAN = {
    "camera": {"exposure": 0.0, "color_cast": 0.0},
    "codec": {"dv_enabled": 0.0},
    "tape": {},
    "display": {"enabled": 0.0},
}


def probe(**fmt):
    layers = {k: dict(v) for k, v in CLEAN.items()}
    layers["format"] = dict(fmt)
    p = merge_values(BASE, layers)
    r = Renderer(Still(), W, H, 25.0, p, seed=5)
    r._build_chain()
    st = r.stage["format"]
    sig = SignalFrame.from_rgb(FLAT, r.std)
    ideal = composite_encode(sig.y, sig.i, sig.q, r.std)
    enc = st.apply(FrameBuffer(FLAT.copy(), None, 1, 0.04, r.std, {"field": 0}))
    real = enc.composite
    return (float(np.asarray(ideal, np.float32).std()),
            float(np.asarray(real, np.float32).std()) if real is not None else float("nan"))


print("   the composite's own standard deviation: this is the quantity that shows the loss")
print("   (I/Q and a flat field's colour both hide it, which is why every earlier sweep found nothing)")
print()
print("   %-42s %-14s %-14s %s" % ("format settings", "ideal std", "real std", "retained"))
for label, fmt in (
    ("as it is (0.75 MHz, vavg 0.85)", {}),
    ("chroma_vavg = 0", {"chroma_vertical_averaging": 0.0}),
    ("chroma_bandwidth = 6 MHz", {"chroma_bandwidth": 6.0}),
    ("both off", {"chroma_vertical_averaging": 0.0, "chroma_bandwidth": 6.0}),
    ("DV on (the composite path is skipped)", {"chroma_bandwidth": 0.75}),
):
    if "DV" in label:
        layers_dv = True
        p = merge_values(BASE, {"format": dict(fmt), "codec": {"dv_enabled": 1.0},
                                "display": {"enabled": 0.0}, "camera": {"exposure": 0.0}})
        r = Renderer(Still(), W, H, 25.0, p, seed=5)
        r._build_chain()
        st = r.stage["format"]
        enc = st.apply(FrameBuffer(FLAT.copy(), None, 1, 0.04, r.std, {"field": 0}))
        real = enc.composite
        sig = SignalFrame.from_rgb(FLAT, r.std)
        ideal = composite_encode(sig.y, sig.i, sig.q, r.std)
        i0, r0 = float(np.asarray(ideal, np.float32).std()), float(np.asarray(real, np.float32).std())
    else:
        i0, r0 = probe(**fmt)
    print("   %-42s %-14.4f %-14.4f %.1f%%" % (label, i0, r0, 100.0 * r0 / max(1e-9, i0)))
