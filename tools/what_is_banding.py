"""
Establish what the banding actually is, and check the input before blaming the chain.

**Where this stands, honestly.** Six hypotheses have been examined: five refuted and one confounded by a statistic
that could not separate a legitimate per-row effect from an error. The chroma recovery is now measured as **accurate**
-- a constant I comes back within 0.1% -- and three of my own conclusions have been withdrawn. So before proposing a
seventh hypothesis, two things have to be settled that should have been settled first:

  * **is the source itself banded?** It is a night JPEG, and JPEG blocks and quantisation show up as exactly this kind
    of horizontal streaking in smooth regions. If the input already has it, then part of what has been chased through
    the chain was never the chain's;
  * **and is the banding in luma, in chroma, or both?** Everything measured so far used R-B, which mixes the two.
    A row-to-row variation of **luma** is a brightness band; of **chroma** it is a colour band; and they have
    different causes and different fixes.

Both are measured on the source and on the neutral chain, with the statistic split into luma and chroma so the two
cannot be confused again.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from atsim.core.params import merge_values                                   # noqa: E402
from atsim.core.pipeline import FrameBuffer                                  # noqa: E402
from atsim.core.profiles import FACTORY_PRESETS, SCHEMA                      # noqa: E402
from atsim.render.renderer import Renderer                                   # noqa: E402

OUT = os.path.join(ROOT, "tools", "ntscrs_ref")
COEF = np.array([0.299, 0.587, 0.114], np.float32)
H, W = 240, 320
SRC = np.asarray(Image.open(os.path.join(OUT, "clean_source.jpg")).convert("RGB"),
                 np.float32) / 255.0
PIC = np.asarray(Image.fromarray((SRC * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR),
                 np.float32) / 255.0


class Still:
    def __init__(self):
        self.width, self.height, self.fps, self.frame_count = W, H, 25.0, 3

    def frame(self, index):
        return PIC


BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])
NEUTRAL = {
    "camera": {"exposure": 0.0, "color_cast": 0.0, "halation": 0.0, "ccd_smear": 0.0,
               "chroma_aberration": 0.0, "lens_flare": 0.0, "spike_spread": 0.0,
               "flare_defocus": 0.0, "longitudinal_ca": 0.0, "axis_offset": 0.0,
               "overflow_drain": 0.0, "lens_fog": 0.0, "lens_wear": 0.0,
               "bloom_amount": 0.0, "dead_sites": 0.0},
    "format": {"color_noise": 0.0, "cross_color": 0.0, "cross_luma": 0.0, "edge_boost": 0.0},
    "player": {"color_noise": 0.0, "cross_color": 0.0, "cross_luma": 0.0},
    "codec": {"dv_enabled": 0.0},
    "tape": {},
    "display": {"enabled": 0.0},
}


def split_banding(img):
    """Row-to-row variation of the row means, separately for luma and for each colour-difference."""
    x = np.asarray(img, np.float32)
    lum = x @ COEF
    rb = x[..., 0] - x[..., 2]
    def rip(a):
        return float(np.abs(np.diff(a.mean(axis=1))).mean())
    return rip(lum), rip(rb)


print("   the source, as it arrives (a night JPEG):")
sl, sr = split_banding(PIC)
print("      luma row-to-row %.6f, R-B row-to-row %.6f" % (sl, sr))

# the same measurement on the full-resolution file, so the resize cannot be the cause
srcl, srcr = split_banding(SRC)
print("      at full resolution (%dx%d): luma %.6f, R-B %.6f"
      % (SRC.shape[1], SRC.shape[0], srcl, srcr))

# and on an artificially smooth field, as a floor for what the statistic reads when there is nothing there
flat = np.zeros((H, W, 3), np.float32)
flat[:] = PIC.reshape(-1, 3).mean(axis=0)
fl, fr = split_banding(flat)
print("      a perfectly flat field of the same colour: luma %.6f, R-B %.6f" % (fl, fr))

print()
print("   and through the neutral chain, display off:")
p = merge_values(BASE, NEUTRAL)
r = Renderer(Still(), W, H, 25.0, p, seed=5)
out = np.clip(r.render_frame(0), 0, 1)
ol, orr = split_banding(out)
print("      luma row-to-row %.6f, R-B row-to-row %.6f" % (ol, orr))

print()
print("   so, against the flat-field floor, the chain multiplies:")
print("      luma  %.3fx  and  R-B  %.3fx" % (ol / max(1e-9, fl), orr / max(1e-9, fr)))
print("   and the source itself already carries, against the same floor:")
print("      luma  %.3fx  and  R-B  %.3fx" % (sl / max(1e-9, fl), sr / max(1e-9, fr)))

print()
print("   and the same split statistic with the picture smoothed, to see what is JPEG and what is scene:")
from atsim.core.signal import lowpass                                      # noqa: E402
sm = lowpass(PIC, 2.0)
sml, smr = split_banding(sm)
print("      the source after a sigma-2 blur: luma %.6f, R-B %.6f" % (sml, smr))
