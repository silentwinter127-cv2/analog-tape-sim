"""
Calibrate the illuminator's scale to the one measurement that bounds it, instead of leaving its units undefined.

**Where the chain stands after the field was rebuilt.**  The field is now correct in shape -- mean **1.0000**, a smooth
peak (**0.0009%** of pixels at the maximum), and **corr(top) = -0.000** so its vertical tilt is set by `beam_y` rather
than baked in.  But normalising it to a **mean of 1.0** means the lamp adds an amount equal to the picture's own
brightness, and measured that is **+0.5072 at every setting from 0.02 to 0.25** -- **13.5x** the **0.0376** the mode is
allowed.

**So the parameter's scale was never defined**, and that is the fault the last several rounds kept meeting in different
clothes.  **The bound is measurable**, which is what makes this a calibration rather than a preference:

    the real NIGHTSHOT 0 is 0.2349 against its own source's 0.1973
    so everything the mode adds -- lamp, IR response and display together -- is 0.0376 of luma
    and the lamp is one part of that

**The response and tint terms average 0.6038 and 1.0060** on this picture, so the product the field is multiplied by is
about **0.607**.  For the lamp to contribute a sensible share of the 0.0376 -- the lamp is the dominant term in a
night-vision shot, so most of it -- the whole scale must come down by roughly **a factor of ten**.

**So a single explicit reference constant carries the scale**, derived from the measurement rather than searched, and
`illuminator` reads as the fraction of that reference the lamp is set to.
"""
from __future__ import annotations

import ast
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ---------------------------------------------------------------- 1. the scale constant in the field
P = os.path.join(ROOT, "atsim", "codecs", "nightvision.py")
s = open(P, encoding="utf-8").read()
OLD = "    field = field / max(1e-6, float(field.mean()))"
NEW = """    # **Normalised by the mean, then reduced by a reference constant derived from the measurement.**  A mean of 1.0
    # alone would mean the lamp adds an amount equal to the picture's whole brightness: measured, `+0.5072` at every
    # setting from 0.02 to 0.25, which is **13.5x** the 0.0376 the mode is allowed.  The bound is measurable -- the real
    # NIGHTSHOT 0 is 0.2349 against its own source's 0.1973, so **everything the mode adds, lamp and IR response and
    # display together, is 0.0376 of luma** -- and the response and tint terms average 0.6038 and 1.0060 on this
    # picture, so the product the field is multiplied by is about 0.607.  **The lamp is the dominant term in a
    # night-vision shot, so `illuminator = 1.0` is defined as "the lamp accounts for the whole measured difference"**,
    # which puts the reference at 0.0376 / 0.607 = 0.062.
    field = field / max(1e-6, float(field.mean()))
    field = field * 0.062"""
if OLD not in s:
    raise SystemExit("the mean normalisation was not found")
s = s.replace(OLD, NEW, 1)
ast.parse(s)
open(P, "w", encoding="utf-8").write(s)
print("   the field now carries a reference scale of 0.062, derived from the measurement")

from atsim.codecs import nightvision as nv                                    # noqa: E402
from atsim.core.params import merge_values                                    # noqa: E402
from atsim.core.profiles import FACTORY_PRESETS, SCHEMA                       # noqa: E402
from atsim.render.renderer import Renderer                                    # noqa: E402

COEF = np.array([0.299, 0.587, 0.114], np.float32)
OBJ = (r"C:\Users\Administrator\.dsh\attachments\v1\objects")
P0 = os.path.join(OBJ, "69", "699c23315df037b01fa831a273364b41f99c28733fb1dc928b22d1e6c281818d")
W, H, OSD = 533, 400, 0.22
sl = slice(int(H * OSD), H)
src = np.asarray(Image.open(P0).convert("RGB"), np.float32) / 255.0
src = np.asarray(Image.fromarray((np.clip(src, 0, 1) * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS),
                 np.float32) / 255.0
L0 = float((src[sl] @ COEF).mean())
ALLOWED = 0.2349 - L0

print()
print("   === the term alone (the whole mode may add %.4f) ===" % ALLOWED)
print("   %-16s %-14s %s" % ("illuminator", "added luma", "verdict"))
for ill in (0.0, 0.15, 0.5, 1.0):
    spec = nv.NightVisionSpec(illuminator=ill, ir_response=1.0)
    out = np.clip(np.asarray(nv.add_illuminator(src, spec), np.float32), 0, 1)
    gained = float((out[sl] @ COEF).mean()) - L0
    print("   %-16.2f %+-14.4f %s" % (ill, gained,
                                      "OK" if gained <= ALLOWED * 1.25 else "%.1fx" % (gained / max(1e-9, ALLOWED))))


class One:
    height, width = H, W
    fps = 25.0
    frame_count = 1

    def frame(self, i):
        return src


BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])
print()
print("   === the chain: smooth, monotone, and near the target 0.2349 ===")
vals = []
for ill in (0.0, 0.15, 0.3, 0.5, 0.8, 1.0):
    layers = {"camera": {"nv_mode": 1.0, "nv_green_mix": 1.0, "nv_illuminator": ill},
              "format": {"chroma_bandwidth": 3.0, "luma_bandwidth": 6.0},
              "player": {"chroma_bandwidth": 3.0, "luma_bandwidth": 6.0}}
    rr = Renderer(One(), W, H, 25.0, merge_values(BASE, layers), seed=1)
    a = np.clip(np.asarray(rr.render_frame(0), np.float32), 0, 1)
    lm = float((a[sl] @ COEF).mean())
    vals.append(lm)
    print("      nv_illuminator = %.2f  ->  luma %.4f  %s"
          % (ill, lm, "<- within 10%% of the target" if abs(lm - 0.2349) < 0.0235 else ""))
d = np.diff(vals)
print("      monotone? %s     flat anywhere? %s"
      % (bool(np.all(d > 1e-5)), bool(np.any(np.abs(d) < 1e-4))))
