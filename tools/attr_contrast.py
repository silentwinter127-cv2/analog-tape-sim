"""
Find what removes half the contrast, by attributing it the way the level was attributed.

**The measurement that reframes everything.**  The model's mean luma matches the footage (0.2371 against 0.2346) while
its **contrast is 0.49 of the footage's** -- and the **source photograph's contrast is 0.93 of it**, so the loss happens
**inside the chain**, not in the input.  **A picture with the right level and half the contrast is exactly what "pale and
inverted-looking" describes**, which is what the user reported -- and the previous rounds calibrated the *level*, so they
could not see it.

**And the lamp is not the cause**: turning it off moves the contrast from 0.490 to 0.462, a change of **0.03**, while
the field's own magnitude is **0.0115** of luma.  **So the loss is elsewhere, and it is a property of the chain rather
than of the night-vision mode.**

**The method is the one that worked for the level**: switch each stage off in turn and record the contrast, so the
largest contributor is identified rather than guessed.  The candidates are the camera's own response curve, the
display's gamma and saturation, and the format/player paths -- and each is measured on its own by turning the others
into pass-throughs.
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

COEF = np.array([0.299, 0.587, 0.114], np.float32)
OBJ = (r"C:\Users\Administrator\.dsh\attachments\v1\objects")
NS0 = os.path.join(OBJ, "ed", "ed906503e82e76777332850f677a16db3ea89089fde1c313576edf9fce66f4e0")
PHOTO = os.path.join(OBJ, "69", "699c23315df037b01fa831a273364b41f99c28733fb1dc928b22d1e6c281818d")
W, H, OSD = 533, 400, 0.22
sl = slice(int(H * OSD), H)

real = np.asarray(Image.open(NS0).convert("RGB"), np.float32) / 255.0
real = np.asarray(Image.fromarray((np.clip(real, 0, 1) * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS),
                  np.float32) / 255.0
R_STD = float(np.clip(real[sl] @ COEF, 0, 1).std())
src = np.asarray(Image.open(PHOTO).convert("RGB"), np.float32) / 255.0
src = np.asarray(Image.fromarray((np.clip(src, 0, 1) * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS),
                 np.float32) / 255.0
S_STD = float(np.clip(src[sl] @ COEF, 0, 1).std())


class One:
    height, width = H, W
    fps = 25.0
    frame_count = 1

    def frame(self, i):
        return src


BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])

print("   the target: the real NIGHTSHOT 0's contrast is %.4f" % R_STD)
print("   the source photograph's is      %.4f   (%.3f of the target -- so the input is fine)" % (S_STD, S_STD / R_STD))

print()
print("   === attributing the contrast, by making each stage a pass-through ===")
print("   %-52s %-10s %-11s %s" % ("configuration", "luma", "contrast", "vs the target"))


def run(tag, cam=None, fmt=None, disp=None):
    layers = {"camera": dict(cam or {"nv_mode": 1.0, "nv_green_mix": 1.0}),
              "format": dict(fmt or {"chroma_bandwidth": 3.0, "luma_bandwidth": 6.0}),
              "player": dict(fmt or {"chroma_bandwidth": 3.0, "luma_bandwidth": 6.0}),
              "display": dict(disp or {})}
    rr = Renderer(One(), W, H, 25.0, merge_values(BASE, layers), seed=1)
    a = np.clip(np.asarray(rr.render_frame(0), np.float32), 0, 1)
    lum = np.clip(a[sl] @ COEF, 0, 1)
    print("   %-52s %-10.4f %-11.4f %.3f" % (tag, float(lum.mean()), float(lum.std()), float(lum.std()) / R_STD))
    return float(lum.std())


# the chain as it has been used
run("the chain as used (factory defaults otherwise)", {"nv_mode": 1.0, "nv_green_mix": 1.0})
# the display stage made a pass-through
run("...display disabled", {"nv_mode": 1.0, "nv_green_mix": 1.0}, disp={"enabled": 0.0})
# gamma flat
run("...display gamma = 1.0", {"nv_mode": 1.0, "nv_green_mix": 1.0}, disp={"gamma": 1.0})
# saturation maximised
run("...display saturation = 1.0", {"nv_mode": 1.0, "nv_green_mix": 1.0}, disp={"saturation": 1.0})
# the mode off entirely, so the camera's own path is seen alone
run("...the mode off", {"nv_mode": 0.0})
# and the camera's defects, which the factory sets and this chain has not
run("...the mode off, display disabled", {"nv_mode": 0.0}, disp={"enabled": 0.0})

print()
print("   === and the camera's own factory defaults, which this chain left in place ===")
for k in sorted(BASE["camera"]):
    v = BASE["camera"][k]
    if isinstance(v, (int, float)) and v not in (0.0, 1.0) and not k.startswith("nv_"):
        print("      %-28s %s" % (k, v))
print()
print("   a term above 1.0 that compresses the tonal range would be the candidate, and so would the display's")
print("   saturation or gamma -- both of which act on every pixel and therefore on the contrast.")
