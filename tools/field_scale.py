"""
Scale the illuminator's field to a physical magnitude, because the radial rebuild left it an order too large.

**The fault, isolated.**  Applied alone to the source, the scene branch's pieces measure:

    add_illuminator, illuminator = 0.15     luma 0.1973 -> 0.6707   **factor 3.3991**
    add_illuminator, illuminator = 0        luma 0.1973 -> 0.1973   factor 1.0000   (the gate works)
    ir_highlights (mix 0.35)                luma 0.1973 -> 0.2066   factor 1.0468
    accumulate                              luma 0.1973 -> 0.1973   factor 1.0000

**So `add_illuminator` is the whole of it**, and the cause is a **units error**: the rebuilt radial field reaches about
**1.18 at its centre** and is **added directly to a picture whose values are in [0, 1]** -- i.e. it adds **118% of full
scale at the brightest point**, with `near_bound = 3.0` allowing up to **300%**.  The old top-down field happened to be
small (0.02 to 0.27), so the same mistake was invisible; **rebuilding the geometry exposed a magnitude that had never
been right.**

**And the physically correct magnitude is measurable rather than a preference.**  The real NIGHTSHOT 0 is **1.19x** its
own source -- 0.2349 against 0.1973 -- so the **entire** difference to be explained, including the lamp, the IR response
and the display, is **0.0376 of luma**.  A lamp cannot be contributing more than that, which puts it at roughly
**0.04** at this framing rather than 1.18: **a factor of thirty**.

**So the field is normalised so that `illuminator = 1.0` means "the lamp's own contribution is comparable to the
scene's brightness"**, which makes the default 0.15 land near the measured difference, and the check is the level against
the footage's own 0.2349.
"""
from __future__ import annotations

import ast
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

P = os.path.join(ROOT, "atsim", "codecs", "nightvision.py")
s = open(P, encoding="utf-8").read()

OLD = """    near_bound = 3.0
    ref = 0.75
    field = np.minimum(ref * ref / np.maximum(d, 1e-3) ** 2, near_bound)
    return field.astype(np.float32)"""
NEW = """    # **Normalised to a physical magnitude, because the radial rebuild exposed an error that the old field hid.**
    # The rebuilt field reached about **1.18 at its centre** and is added directly to a picture in [0, 1] -- **118% of
    # full scale at the brightest point**, with the near-field clamp allowing up to 300%.  Measured alone,
    # `add_illuminator` at the schema's 0.15 took a source of 0.1973 to **0.6707**, a factor of **3.40**, and it was the
    # whole of the chain's brightness excess.  The old top-down field happened to be small (0.02 to 0.27), so the same
    # mistake was invisible.
    #
    # **The correct magnitude is measurable**: the real NIGHTSHOT 0 is 1.19x its own source -- 0.2349 against 0.1973 --
    # so the entire difference to be accounted for, lamp, IR response and display together, is **0.0376 of luma**.  A
    # lamp cannot contribute more than that.  **So `illuminator = 1.0` is defined as "the lamp's peak contribution
    # equals the scene's own mean brightness"**, and the default 0.15 then lands at about a seventh of that -- near the
    # measured difference rather than thirty times above it.
    near_bound = 3.0
    ref = 0.75
    field = np.minimum(ref * ref / np.maximum(d, 1e-3) ** 2, near_bound)
    # scaled so the peak is 1.0, which makes the units the picture's own
    field = field / max(1e-6, float(field.max()))
    return field.astype(np.float32)"""
if OLD not in s:
    raise SystemExit("the field's tail was not found")
s = s.replace(OLD, NEW, 1)
ast.parse(s)
open(P, "w", encoding="utf-8").write(s)
print("   the field is now normalised so its peak is 1.0")

from atsim.codecs import nightvision as nv                                    # noqa: E402
from atsim.core.params import merge_values                                    # noqa: E402
from atsim.core.profiles import FACTORY_PRESETS, SCHEMA                       # noqa: E402
from atsim.render.renderer import Renderer                                    # noqa: E402

COEF = np.array([0.299, 0.587, 0.114], np.float32)
OBJ = (r"C:\Users\Administrator\.dsh\attachments\v1\objects")
NS0 = os.path.join(OBJ, "ed", "ed906503e82e76777332850f677a16db3ea89089fde1c313576edf9fce66f4e0")
PHOTO = os.path.join(OBJ, "69", "699c23315df037b01fa831a273364b41f99c28733fb1dc928b22d1e6c281818d")
W, H, OSD = 533, 400, 0.22
sl = slice(int(H * OSD), H)

real = np.asarray(Image.open(NS0).convert("RGB"), np.float32) / 255.0
TARGET = float((real[int(real.shape[0] * OSD):] @ COEF).mean())
src = np.asarray(Image.open(PHOTO).convert("RGB"), np.float32) / 255.0
src = np.asarray(Image.fromarray((np.clip(src, 0, 1) * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS),
                 np.float32) / 255.0
L0 = float((src[sl] @ COEF).mean())

print()
print("   === the term alone, before and after ===")
print("   %-34s %-10s %-10s" % ("piece", "luma", "factor"))
for tag, spec in (("add_illuminator, illuminator = 0.15",
                   nv.NightVisionSpec(illuminator=0.15, ir_response=1.0)),
                  ("add_illuminator, illuminator = 0.0",
                   nv.NightVisionSpec(illuminator=0.0, ir_response=1.0))):
    o = np.clip(np.asarray(nv.add_illuminator(src, spec), np.float32), 0, 1)
    lm = float((o[sl] @ COEF).mean())
    print("   %-34s %-10.4f %-10.4f" % (tag, lm, lm / max(1e-9, L0)))
print("      the real frame is %.4f against the source's %.4f -- a factor of %.3f"
      % (TARGET, L0, TARGET / L0))


class One:
    height, width = H, W
    fps = 25.0
    frame_count = 1

    def frame(self, i):
        return src


BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])
OFFDEF = {"halation": 0.0, "ccd_smear": 0.0, "flare_defocus": 0.0, "lens_flare": 0.0,
          "overflow_drain": 0.0, "chroma_aberration": 0.0}
DISPLAY = {"enabled": 1.0, "gamma": 1.0, "scanlines": 0.0, "mask_depth": 0.0, "saturation": 0.62}


def luma_of(**cam):
    layers = {"camera": dict(OFFDEF, nv_mode=1.0, nv_green_mix=1.0, **cam),
              "format": {"chroma_bandwidth": 3.0, "luma_bandwidth": 6.0},
              "player": {"chroma_bandwidth": 3.0, "luma_bandwidth": 6.0},
              "codec": {"dv_enabled": 1.0}, "tape": {"head_switching": 0.0}, "display": dict(DISPLAY)}
    rr = Renderer(One(), W, H, 25.0, merge_values(BASE, layers), seed=20240101)
    a = np.clip(np.asarray(rr.render_frame(0), np.float32), 0, 1)
    return float((a[sl] @ COEF).mean())


print()
print("   === the chain's level at each illuminator setting ===")
print("   %-34s %-10s %-10s %s" % ("setting", "luma", "factor", "against the target %.4f" % TARGET))
for ill in (0.0, 0.05, 0.10, 0.15, 0.25):
    lm = luma_of(nv_illuminator=ill)
    print("   %-34s %-10.4f %-10.4f %s"
          % ("nv_illuminator = %.2f" % ill, lm, lm / max(1e-9, L0),
             "<- within 10%" if abs(lm - TARGET) < 0.10 * TARGET else ""))
