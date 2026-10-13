"""
Measure the encoder's Y/I/Q against the decoder's, which is where the flat-field error must be.

The round trip is now narrowed to one place. The matrix pair `rgb -> Y/I/Q -> rgb` is exact to 1e-4, and the player's
separation of the composite is off by **+0.0637 in blue** on a flat sodium-lit colour. A flat field is the one input
a chroma transform is *required* to pass through unchanged, so the error is in the separation itself.

Two measurements pin it down without guessing at any coefficient:

  * **I and Q, in against out.**  For a flat field the encoder produces constant I and Q; the decoder reads constant
    I and Q back. Comparing them says whether the error is a **gain mismatch** -- a scale factor -- or an additive
    offset, and those have different causes and different fixes.
  * **and the same for a set of flat colours**, because a gain error shows up as a ratio that is constant across
    colours while an offset does not.

The separation methods are reported separately, since they disagree with each other by two orders of magnitude and
that disagreement is itself part of the finding.
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
BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])

COLOURS = {
    "sodium orange": [0.7519, 0.3592, 0.0184],
    "neutral grey": [0.45, 0.45, 0.45],
    "saturated red": [0.80, 0.10, 0.10],
    "saturated blue": [0.10, 0.10, 0.80],
    "saturated green": [0.10, 0.80, 0.10],
    "white": [0.90, 0.90, 0.90],
}


class Still:
    def __init__(self, colour):
        self.width, self.height, self.fps, self.frame_count = W, H, 25.0, 3
        self._c = np.asarray(colour, np.float32)

    def frame(self, index):
        a = np.zeros((H, W, 3), np.float32)
        a[:] = self._c
        return a


def player_out(colour, kind, params=None):
    p = merge_values(BASE, {"player": dict({"separation_kind": kind}, **(params or {})),
                            "display": {"enabled": 0.0},
                            "format": {"chroma_bandwidth": 3.0, "luma_bandwidth": 6.0},
                            "camera": {"exposure": 0.0}})
    r = Renderer(Still(colour), W, H, 25.0, p, seed=5)
    r._build_chain()
    std = r.std
    flat = np.asarray(Still(colour).frame(0), np.float32)
    sig = SignalFrame.from_rgb(flat, std)
    comp = composite_encode(sig.y, sig.i, sig.q, std)
    # what the encoder put in
    src_i = float(np.asarray(sig.i, np.float32).mean())
    src_q = float(np.asarray(sig.q, np.float32).mean())
    buf = FrameBuffer(flat.copy(), comp, 1, 0.04, std, {"field": 0})
    st = r.stage["player"]
    sep = st._separate(comp, buf)
    # `_separate` returns three values: it yields its own luma as well, which is why unpacking two failed
    got_i, got_q = sep[1], sep[2]
    rgb = np.clip(np.asarray(SignalFrame(float(np.asarray(sig.y, np.float32).mean()),
                                         got_i, got_q).to_rgb(), np.float32), 0, 1)
    return (src_i, src_q, float(np.asarray(got_i, np.float32).mean()),
            float(np.asarray(got_q, np.float32).mean()), rgb.reshape(-1, 3).mean(axis=0))


print("   %-16s %-26s %-26s %s" % ("colour", "encoder I,Q", "decoder I,Q", "recovered rgb"))
print("   %-16s %-26s %-26s %s" % ("", "", "", "(the source)"))
for name, c in COLOURS.items():
    si, sq, gi, gq, rgb = player_out(c, 0.0)
    print("   %-16s %-26s %-26s %s"
          % (name, "%+.4f %+.4f" % (si, sq), "%+.4f %+.4f" % (gi, gq),
             "%s  (src %s)" % (np.round(rgb, 4), np.round(np.asarray(c, np.float32), 4))))

print()
print("   the ratio the decoder applies, per colour (decoder / encoder), kind 0:")
for name, c in COLOURS.items():
    si, sq, gi, gq, _ = player_out(c, 0.0)
    ri = gi / si if abs(si) > 1e-6 else float("nan")
    rq = gq / sq if abs(sq) > 1e-6 else float("nan")
    print("      %-16s I x%.4f   Q x%.4f" % (name, ri, rq))

print()
print("   and the methods compared on the neutral grey, where a correct transform changes nothing:")
for kind in (0.0, 1.0, 2.0, 3.0):
    _, _, _, _, rgb = player_out([0.45, 0.45, 0.45], kind)
    print("      separation_kind=%.0f -> %s  (blue error %+.4f)"
          % (kind, np.round(rgb, 4), float(rgb[2] - 0.45)))
