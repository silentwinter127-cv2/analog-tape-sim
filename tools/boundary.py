"""
Compare the format stage's real composite against an ideal one, and see what the player recovers from each.

The fault is at the boundary: `_separate` recovers a flat colour to within 0.25% from a composite built with
`composite_encode`, and returns blue 0.0950 from the composite the **format stage actually produces**, against a
source of 0.0184. So the two composites are not the same signal, and the question is what the format stage adds or
loses.

Three things are measured on the same flat field, in the same process, so nothing but the composite differs:

  * the **I and Q the format stage built its composite from**, and the I and Q it produces after its own encoder;
  * an **ideal composite** built from the same Y/I/Q with `composite_encode`, as the control;
  * and what the player recovers from **each**, so the difference between the two composites is expressed as the
    colour it produces.

If the format stage's own I and Q are already wrong, the fault is in the encoder; if they are right and only the
composite differs, it is in the modulation or in what the player assumes about it.
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
    "camera": {"exposure": 0.0, "color_cast": 0.0, "halation": 0.0, "ccd_smear": 0.0,
               "chroma_aberration": 0.0, "lens_flare": 0.0, "spike_spread": 0.0,
               "flare_defocus": 0.0, "longitudinal_ca": 0.0, "axis_offset": 0.0},
    "codec": {"dv_enabled": 0.0},
    "tape": {},
    "display": {"enabled": 0.0},
}


def build(**fmt):
    layers = {k: dict(v) for k, v in CLEAN.items()}
    layers["format"] = dict(fmt)
    p = merge_values(BASE, layers)
    r = Renderer(Still(), W, H, 25.0, p, seed=5)
    r._build_chain()
    return r


def stats(a):
    a = np.asarray(a, np.float32)
    return float(a.mean()), float(a.std())


print("   the flat field: %s" % np.round(COLOUR, 4))
print()
for label, fmt in (
    ("format at its defaults", {}),
    ("format with chroma_gain 1.0 (explicit)", {"chroma_gain": 1.0}),
    ("format with no chroma lowpass", {"chroma_bandwidth": 6.0}),
):
    r = build(**fmt)
    fmt_stage = r.stage["format"]
    player = r.stage["player"]

    # what the encoder is given
    ideal = SignalFrame.from_rgb(FLAT, r.std)
    ideal_i = float(np.asarray(ideal.i, np.float32).mean())
    ideal_q = float(np.asarray(ideal.q, np.float32).mean())

    # run the format stage and take its composite
    enc = fmt_stage.apply(FrameBuffer(FLAT.copy(), None, 1, 0.04, r.std, {"field": 0}))
    comp_real = enc.composite
    comp_ideal = composite_encode(ideal.y, ideal.i, ideal.q, r.std)

    print("   --- %s ---" % label)
    print("      I/Q the encoder was given      I %+.4f  Q %+.4f" % (ideal_i, ideal_q))
    if comp_real is not None and comp_ideal is not None:
        same = (comp_real.shape == comp_ideal.shape and
                np.allclose(comp_real, comp_ideal, atol=1e-6))
        print("      its composite against the ideal: identical %s" % same)
        if not same and comp_real.shape == comp_ideal.shape:
            d = np.abs(np.asarray(comp_real, np.float32) - np.asarray(comp_ideal, np.float32))
            print("         max |diff| %.5f, mean %.5f" % (float(d.max()), float(d.mean())))
            print("         real: mean %.4f std %.4f | ideal: mean %.4f std %.4f"
                  % (stats(comp_real)[0], stats(comp_real)[1], stats(comp_ideal)[0], stats(comp_ideal)[1]))
    # what the player recovers from each
    for tag, comp in (("the real composite", comp_real), ("the ideal composite", comp_ideal)):
        if comp is None:
            continue
        buf = FrameBuffer(FLAT.copy(), comp, 1, 0.04, r.std, {"field": 0})
        sep = player._separate(comp, buf)
        gi = float(np.asarray(sep[1], np.float32).mean())
        gq = float(np.asarray(sep[2], np.float32).mean())
        y = float(np.asarray(sep[0], np.float32).mean())
        rgb = np.clip(np.asarray(SignalFrame(y, sep[1], sep[2]).to_rgb(), np.float32), 0, 1)
        print("      from %-20s I %+.4f Q %+.4f  -> rgb %s"
              % (tag, gi, gq, np.round(rgb.reshape(-1, 3).mean(axis=0), 4)))
    print()
