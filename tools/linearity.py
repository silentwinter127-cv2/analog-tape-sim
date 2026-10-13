"""
Is the player's separation linear in chroma amplitude? That single question separates the two remaining causes.

The encoder's composite retains 68.9% of an ideal one's amplitude, and the player then recovers 47.7% of the I the
encoder was handed. Two things could explain the second number:

  * **linearity.** If the separation is linear in the composite's chroma amplitude, then feeding it a composite whose
    chroma has been scaled by `s` returns I scaled by `s`, and 47.7% is simply the consequence of the encoder's own
    loss -- 0.689 times something, or the same loss applied twice. The fix would then be at the encoder, and the
    player is innocent.
  * **non-linearity.** If the separation *loses more* as the amplitude falls -- a normalisation against a reference,
    a phase estimate that degrades, a threshold that bites -- then 47.7% is a loss of its own and the fix is in the
    demodulation.

The test builds composites from one set of I/Q with the chroma scaled by a known factor, feeds each to the same
`_separate`, and reports the recovered I as a fraction of what was put in. **A constant fraction means linear; a
fraction that falls with the scale means the separation is losing the chroma itself.**
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


def setup(**player):
    p = merge_values(BASE, {"player": dict(player), "display": {"enabled": 0.0},
                            "camera": {"exposure": 0.0}, "codec": {"dv_enabled": 0.0}})
    r = Renderer(Still(), W, H, 25.0, p, seed=5)
    r._build_chain()
    return r


r = setup()
std = r.std
st = r.stage["player"]
sig = SignalFrame.from_rgb(FLAT, std)
I0 = float(np.asarray(sig.i, np.float32).mean())
Q0 = float(np.asarray(sig.q, np.float32).mean())
print("   the I/Q the encoder is handed: I %+.5f  Q %+.5f" % (I0, Q0))
print("   the subcarrier is %.2f MHz; at this width the pixel rate is %.2f MHz"
      % (std.subcarrier_hz / 1e6, std.pixel_rate_hz / 1e6))
print()

print("   %-10s %-14s %-14s %-12s %s"
      % ("chroma x", "composite std", "recovered I", "fraction", "linear?"))

fractions = []
for s in (1.0, 0.8, 0.689, 0.5, 0.35, 0.2, 0.1):
    comp = composite_encode(sig.y, sig.i * s, sig.q * s, std)
    ideal_std = float(np.asarray(comp, np.float32).std())
    buf = FrameBuffer(FLAT.copy(), comp, 1, 0.04, std, {"field": 0})
    sep = st._separate(comp, buf)
    gi = float(np.asarray(sep[1], np.float32).mean())
    frac = gi / max(1e-9, I0 * s)
    fractions.append((s, frac))
    print("   %-10.3f %-14.5f %-14.5f %-12.3f" % (s, ideal_std, gi, frac))

print()
lo = [f for s, f in fractions if s <= 0.35]
hi = [f for s, f in fractions if s >= 0.8]
print("   the fraction at high amplitude: %.3f" % (sum(hi) / len(hi)))
print("   the fraction at low amplitude:  %.3f" % (sum(lo) / len(lo)))
print("   the spread is %.3f, so the separation is %s"
      % (max(f for _, f in fractions) - min(f for _, f in fractions),
         "LINEAR in chroma amplitude" if max(f for _, f in fractions) - min(f for _, f in fractions) < 0.05
         else "NON-LINEAR: it loses more as the amplitude falls"))

print()
print("   and the same for the player's own recovered picture, at each scale:")
for s in (1.0, 0.689, 0.5):
    comp = composite_encode(sig.y, sig.i * s, sig.q * s, std)
    buf = FrameBuffer(FLAT.copy(), comp, 1, 0.04, std, {"field": 0})
    out = st.apply(buf)
    rgb = np.asarray(out.rgb if out.rgb is not None else out.signal.to_rgb(), np.float32)
    print("      chroma x%.3f -> rgb %s   (the source %s)"
          % (s, np.round(rgb.reshape(-1, 3).mean(axis=0), 4), np.round(COLOUR, 4)))
