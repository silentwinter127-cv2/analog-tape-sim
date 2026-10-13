"""
Find where the 89% brightness difference comes from by isolating each stage's own contribution.

**The fact to explain, measured:**  the source photograph's own luma is **0.2452** and the real NIGHTSHOT 0's is
**0.2349** -- nearly the same, which is what a monochrome view should do -- while the model's output reads **0.4443**,
a factor of **1.89**.  And the display is **not** the cause: its luma gain is **exactly 1.00000** on every input, and its
sum gain lies between **0.865 and 1.093**, i.e. a **0--13%** effect against the 89% that needs explaining.

**So the excess is elsewhere and the method has to be to attribute it rather than to guess and re-measure.**  The chain
is applied to the photograph one component at a time, and after each the mean luminance is recorded -- so the 89% is
divided among the contributors instead of being chased one edit at a time.  **That is the step the previous rounds
skipped**, and it is why a plausible mechanism (the display's normalisation) was adopted without checking its sign
against the effect it was meant to explain.

**The components, and what each is expected to do:**

  * **the illuminator** -- adds infrared light, which is *additive* and therefore **raises** the level by construction;
  * **the IR highlights and the mode's exposure bias** -- both raise it;
  * the **display's mono projection** -- measured to be neutral to within 13%;
  * and the **format/player/display stages** -- bandwidth and gamma, which move the level in ways that have to be
    measured rather than assumed.

**The output is an attribution table**, and the largest term is the answer regardless of what it turns out to be.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from atsim.codecs import nightvision as nv                                    # noqa: E402
from atsim.core.params import merge_values                                    # noqa: E402
from atsim.core.profiles import FACTORY_PRESETS, SCHEMA                       # noqa: E402
from atsim.render.renderer import Renderer                                    # noqa: E402

COEF = np.array([0.299, 0.587, 0.114], np.float32)
OBJ = (r"C:\Users\Administrator\.dsh\attachments\v1\objects")
NS0 = os.path.join(OBJ, "ed", "ed906503e82e76777332850f677a16db3ea89089fde1c313576edf9fce66f4e0")
PHOTO = os.path.join(OBJ, "69", "699c23315df037b01fa831a273364b41f99c28733fb1dc928b22d1e6c281818d")
OSD = 0.22

real = np.asarray(Image.open(NS0).convert("RGB"), np.float32) / 255.0
TARGET = float((real[int(real.shape[0] * OSD):] @ COEF).mean())

W, H = 533, 400
src = np.asarray(Image.open(PHOTO).convert("RGB"), np.float32) / 255.0
src = np.asarray(Image.fromarray((np.clip(src, 0, 1) * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS),
                 np.float32) / 255.0
SOURCE = float((src[int(H * OSD):] @ COEF).mean())

print("   === the two anchors ===")
print("      the source photograph's own luma   %.4f" % SOURCE)
print("      the real NIGHTSHOT 0's luma        %.4f   (the model reads 0.4443)" % TARGET)
print("      so the chain owes at most %.4f of change, and delivers 0.4443 -- an excess of %.4f"
      % (TARGET - SOURCE, 0.4443 - TARGET))


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


def luma_of(camera, fmt=None, disp=None):
    layers = {"camera": dict(OFFDEF, **camera),
              "format": fmt or {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0},
              "player": fmt or {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0},
              "codec": {"dv_enabled": 1.0}, "tape": {"head_switching": 0.0},
              "display": dict(DISPLAY, **(disp or {}))}
    p = merge_values(BASE, layers)
    rr = Renderer(One(), W, H, 25.0, p, seed=20240101)
    a = np.clip(np.asarray(rr.render_frame(0), np.float32), 0, 1)
    return float((a[int(H * OSD):] @ COEF).mean())


print()
print("   === attribution: each component added to the one before it ===")
print("   %-44s %-9s %-9s %s" % ("configuration", "luma", "change", "of the source"))

# 1. the mode off entirely -- the reference for what the chain does without NightShot
steps = [("the mode off (baseline chain)", dict(nv_mode=0.0))]
# 2. the mode on with everything at its minimum
steps.append(("+ the mode on, nothing else", dict(nv_mode=1.0, nv_green_mix=1.0,
                                                  nv_illuminator=0.0, nv_exposure_steps=0.0)))
# 3. + the illuminator
steps.append(("+ the infrared illuminator", dict(nv_mode=1.0, nv_green_mix=1.0,
                                                 nv_illuminator=0.15, nv_exposure_steps=0.0)))
# 4. + the exposure bias
steps.append(("+ the mode's exposure bias", dict(nv_mode=1.0, nv_green_mix=1.0,
                                                 nv_illuminator=0.15, nv_exposure_steps=1.0)))
# 5. the display alone, with the illuminator off, to isolate it
steps.append(("the display alone (no lamp, no bias)", dict(nv_mode=1.0, nv_green_mix=1.0,
                                                           nv_illuminator=0.0, nv_exposure_steps=0.0)))
prev = SOURCE
print("   %-44s %-9.4f %-9s %.3f" % ("the source photograph alone", SOURCE, "-", 1.0))
for name, cam in steps:
    lm = luma_of(cam)
    print("   %-44s %-9.4f %-+9.4f %.3f" % (name, lm, lm - prev, lm / max(1e-9, SOURCE)))
    prev = lm

print()
print("   === and the display's own contribution, isolated from the chain ===")
lum = np.clip(src @ COEF, 0, 1)
mono = nv.monochrome_green(src, nv.NightVisionSpec(green_mix=1.0, illuminator=0.0))
sc = slice(int(H * OSD), H)
print("      the source, below the display band        %.4f" % float(lum[sc].mean()))
print("      after the mono projection alone           %.4f   (luma gain %.5f, sum gain %.5f)"
      % (float((mono[sc] @ COEF).mean()),
         float((mono[sc] @ COEF).mean() / max(1e-9, float(lum[sc].mean()))),
         float(mono[sc].mean() / max(1e-9, float(src[sc].mean())))))
print()
print("   the illuminator is additive, so it is the prime suspect by construction; the table above says how much of")
print("   the excess it actually accounts for, which is the number the previous rounds never measured.")
