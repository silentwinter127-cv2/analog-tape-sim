"""
Isolate which camera term compresses the tonal range, since the display stage is already ruled out.

**What the attribution established:**

    the real NIGHTSHOT 0's contrast        0.2032        the target
    the source photograph's               0.1894        0.932 of it -- so the input is fine
    the chain, mode off                   0.1347        0.663
    the chain, mode on                    0.0995        0.490

**and the display stage contributes nothing**: disabling it, flattening its gamma, or taking its saturation to 1.0 all
leave the contrast at **0.490**, bit-identical.  So the loss is in the **camera stage**, which this chain has been
running at its factory defaults throughout -- and those defaults include several terms whose whole purpose is to move
the picture toward the middle of the range:

    dynamic_range 8.0     a compressed tonal range, which is exactly a contrast reduction
    ae_target 0.46        the exposure loop's own target level
    halation 0.45         light bleeding out of the highlights
    bloom_threshold 0.82  a bloom term above the threshold
    sensor_noise 0.06     noise, which raises the local variance in the shadows
    noise_grain 1.4       and a further grain term

**The method is the one that has worked twice**: switch each term to its neutral value in turn, one at a time, and record
the contrast.  **The neutral value for each is zero**, since every one of these adds something, and a term whose removal
restores the contrast owns the loss.

**And the target is the footage's 0.2032** -- but the honest comparison is against the **source's own 0.1894**, because
the chain should be *changing* the picture, not halving its contrast before the mode does anything.
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


def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0
    return np.asarray(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS),
                      np.float32) / 255.0


real = load(NS0)
R_STD = float(np.clip(real[sl] @ COEF, 0, 1).std())
src = load(PHOTO)
S_STD = float(np.clip(src[sl] @ COEF, 0, 1).std())


class One:
    height, width = H, W
    fps = 25.0
    frame_count = 1

    def frame(self, i):
        return src


BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])
print("   the target: the footage's contrast %.4f; the source's own %.4f (%.3f of it)"
      % (R_STD, S_STD, S_STD / R_STD))


def run(tag, **camover):
    cam = {"nv_mode": 0.0}
    cam.update(camover)
    layers = {"camera": cam,
              "format": {"chroma_bandwidth": 3.0, "luma_bandwidth": 6.0},
              "player": {"chroma_bandwidth": 3.0, "luma_bandwidth": 6.0}}
    rr = Renderer(One(), W, H, 25.0, merge_values(BASE, layers), seed=1)
    a = np.clip(np.asarray(rr.render_frame(0), np.float32), 0, 1)
    lum = np.clip(a[sl] @ COEF, 0, 1)
    print("   %-44s %-10.4f %-11.4f %.3f" % (tag, float(lum.mean()), float(lum.std()), float(lum.std()) / S_STD))
    return float(lum.std())


print()
print("   === each camera term switched to its neutral value, one at a time (the mode is OFF throughout) ===")
print("   %-44s %-10s %-11s %s" % ("", "luma", "contrast", "vs the SOURCE's own"))
base_std = run("the factory defaults, as the chain runs them")
for name in ("dynamic_range", "halation", "sensor_noise", "noise_grain", "sharpness",
             "bloom_threshold", "fixed_pattern", "spike_spread", "flare_defocus", "ccd_smear",
             "motion_blur", "stuck_fraction", "chroma_aberration", "overflow_width"):
    if name not in BASE["camera"]:
        continue
    # `dynamic_range` and `sharpness` are neutral at 1.0 rather than 0, since they are not additive terms
    neutral = 14.0 if name == "dynamic_range" else (1.0 if name == "sharpness" else 0.0)
    s = run("...%s = %s" % (name, neutral), **{name: neutral})

print()
print("   === and all of them at once ===")
alloff = {k: (14.0 if k == "dynamic_range" else (1.0 if k == "sharpness" else 0.0))
          for k in ("dynamic_range", "halation", "sensor_noise", "noise_grain", "sharpness",
                    "bloom_threshold", "fixed_pattern", "spike_spread", "flare_defocus", "ccd_smear",
                    "motion_blur", "stuck_fraction", "chroma_aberration", "overflow_width")
          if k in BASE["camera"]}
run("every camera term neutral", **alloff)
print()
print("   the term whose removal recovers the most contrast is the one to fix; the source's own %.4f is the"  % S_STD)
print("   reference, since the chain should not halve a picture's contrast before the mode does anything.")
