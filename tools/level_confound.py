"""
Separate the two things `_night_vision` does to the measure: it changes the LEVEL, and it adds signal-scaled noise.

**What is established.**  The camera takes the row-structure measure from **0.12737 to 0.16272**, and **every tone curve,
lens term and the sharpen after the night-vision call leaves it untouched**.  The whole addition is inside
`_night_vision` (+0.03844).  **And the night-vision terms applied alone to the source move it by at most 0.002** -- so no
term does it in isolation.

**The confound that has to be ruled out first.**  The measure is the mean absolute adjacent-row difference **divided by
the picture's own spread**, so **anything that changes the spread changes the measure without changing any structure**.
And `_night_vision` raises the luma from **0.1973 to 0.2303**.  **If a plain level change moves the measure by a similar
amount, then the "residual" is a property of the instrument and not of the chain** -- which would be the tenth statistic
in this work to have misled, and the one that would close the question rather than open it.

**So a level change is applied to the source with no night-vision terms at all**, using the same `_expose` the pipeline
itself uses, and the measure is taken across a range of levels.  **The prediction is stated so it can fail**: if the
measure is roughly flat in level, the confound is not the explanation and the addition is real; if it moves by about
0.035 over the range the mode covers, the question is closed.

**And the second half of the confound is tested the same way**: the shot-noise term alone, at the level the mode runs it,
to see how much of the addition it accounts for.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from atsim.core.params import merge_values                                    # noqa: E402
from atsim.core.profiles import FACTORY_PRESETS, SCHEMA                       # noqa: E402
from atsim.render.renderer import Renderer                                    # noqa: E402
from atsim.codecs import nightvision as NV                                    # noqa: E402

COEF = np.array([0.299, 0.587, 0.114], np.float32)
OBJ = (r"C:\Users\Administrator\.dsh\attachments\v1\objects")
PHOTO = os.path.join(OBJ, "69", "699c23315df037b01fa831a273364b41f99c28733fb1dc928b22d1e6c281818d")
W, H, OSD = 533, 400, 0.22
sl = slice(int(H * OSD), H)

_a = np.asarray(Image.open(PHOTO).convert("RGB"), np.float32) / 255.0
SRC = np.asarray(Image.fromarray((np.clip(_a, 0, 1) * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS),
                 np.float32) / 255.0


def rowsig(a):
    g = np.clip(np.asarray(a, np.float32)[sl] @ COEF, 0, 1)
    return float(np.abs(np.diff(g, axis=0)).mean()) / max(1e-9, float(g.std()))


def luma(a):
    return float(np.clip(np.asarray(a, np.float32)[sl] @ COEF, 0, 1).mean())


print("   the source: rowsig %.5f  luma %.4f" % (rowsig(SRC), luma(SRC)))
print("   the mode raises the luma to about 0.2303, so that is the level change to reproduce.")
print()
print("   === a PURE LEVEL change, with no night-vision terms at all ===")
print("   %-12s %-12s %-12s %s" % ("gain", "luma", "rowsig", "change"))
base = rowsig(SRC)
for gain in (0.8, 0.9, 1.0, 1.1, 1.17, 1.3, 1.5, 1.8):
    a = np.clip(SRC * gain, 0, 1)
    print("   %-12.2f %-12.4f %-12.5f %+.5f" % (gain, luma(a), rowsig(a), rowsig(a) - base))
print()
print("   the mode's own level change is 0.1973 -> 0.2303, a gain of about 1.17.")
print()
print("   === and the shot-noise term alone, at the level the mode runs it ===")
spec_base = NV.NightVisionSpec(mode=1.0, illuminator=0.0, ir_response=0.0, ir_albedo_mix=0.0,
                               green_mix=0.0, chroma_residue=0.0, chroma_noise=0.0, read_noise=0.0,
                               shot_noise=0.0)
print("   %-24s %-12s %-12s %s" % ("", "luma", "rowsig", "change"))
for sn in (0.0, 0.10, 0.20, 0.35):
    spec = NV.NightVisionSpec(**{**spec_base.__dict__, "shot_noise": sn})
    a = np.clip(np.asarray(NV.read_noise(SRC, spec, 5), np.float32), 0, 1)
    print("   %-24s %-12.4f %-12.5f %+.5f" % ("shot_noise %.2f" % sn, luma(a), rowsig(a), rowsig(a) - base))
print()
print("   === and the two together, which is what the mode actually does ===")
print("   %-24s %-12s %-12s %s" % ("", "luma", "rowsig", "change"))
for sn in (0.0, 0.10):
    spec = NV.NightVisionSpec(**{**spec_base.__dict__, "shot_noise": sn})
    a = np.clip(np.asarray(NV.read_noise(SRC, spec, 5), np.float32) * 1.17, 0, 1)
    print("   %-24s %-12.4f %-12.5f %+.5f" % ("gain 1.17, shot %.2f" % sn, luma(a), rowsig(a), rowsig(a) - base))
print()
print("   **the prediction:** if the measure is roughly FLAT in level, the level change is not the explanation and the")
print("   camera's addition is real structure.  If it moves by about 0.035 over this range, the 'residual' is the")
print("   instrument and the question is closed.")
