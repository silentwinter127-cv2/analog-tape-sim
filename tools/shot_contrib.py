"""
What the shot-noise term contributes to the picture, and what the mode-1 banding consists of.

**Established numerically, and it is the only term of eight that does anything:**

    mode 1 as it renders         row-difference 0.13127
    mode 1, shot_noise = 0       0.12500        <- the source's own value is 0.12737
    mode 1, read_noise = 0       0.13127        bit-identical
    mode 1, chroma_noise = 0     0.13127        bit-identical
    mode 1, exposure bias = 0    0.13127        bit-identical
    mode 0 (off)                 0.10040

**so `shot_noise` accounts for about half the excess row structure** (0.0063 of the 0.0133 above the source), and the
other half is not any of the terms tested -- and **mode 0 sits well below the source, i.e. it is smoothing the picture's
own row structure away**, which is why it looked clean in the four-panel figure.

**But five statistics in this work have now disagreed with the eye, so this writes the pictures**: the same crop at three
settings, so the difference the number describes can be looked at rather than inferred.  **And the source's own row
structure is the reference**: the model's input already carries it, so a render that preserves it is not "banded" in the
sense the user means -- the question is whether the render *amplifies* it into visible stripes.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from atsim.core.params import merge_values                                    # noqa: E402
from atsim.core.profiles import FACTORY_PRESETS, SCHEMA                       # noqa: E402
from atsim.render.renderer import Renderer                                    # noqa: E402

COEF = np.array([0.299, 0.587, 0.114], np.float32)
OBJ = (r"C:\Users\Administrator\.dsh\attachments\v1\objects")
NS0 = os.path.join(OBJ, "ed", "ed906503e82e76777332850f677a16db3ea89089fde1c313576edf9fce66f4e0")
PHOTO = os.path.join(OBJ, "69", "699c23315df037b01fa831a273364b41f99c28733fb1dc928b22d1e6c281818d")
OUT = os.path.join(ROOT, "tools", "calib")
W, H, OSD = 533, 400, 0.22
sl = slice(int(H * OSD), H)


def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0
    return np.asarray(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS),
                      np.float32) / 255.0


def rowsig(a):
    g = np.clip(np.asarray(a, np.float32)[sl] @ COEF, 0, 1)
    return float(np.abs(np.diff(g, axis=0)).mean()) / max(1e-9, float(g.std()))


src = load(PHOTO)
real = load(NS0)


class One:
    height, width = H, W
    fps = 25.0
    frame_count = 1

    def frame(self, i):
        return src


BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])
OFF = {"halation": 0.0, "ccd_smear": 0.0, "flare_defocus": 0.0, "lens_flare": 0.0,
       "overflow_drain": 0.0, "chroma_aberration": 0.0, "sensor_noise": 0.0, "noise_grain": 0.0,
       "fixed_pattern": 0.0, "spike_spread": 0.0, "motion_blur": 0.0, "stuck_fraction": 0.0,
       "bloom_threshold": 0.0, "overflow_width": 0.0, "sharpness": 1.0}


def run(mode, **cam):
    layers = {"camera": dict(OFF, nv_mode=mode, **cam),
              "format": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0},
              "player": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0},
              "codec": {"dv_enabled": 0.0}, "display": {"enabled": 0.0}}
    rr = Renderer(One(), W, H, 25.0, merge_values(BASE, layers), seed=20240101)
    return np.clip(np.asarray(rr.render_frame(0), np.float32), 0, 1)


panels = [("the real footage", real),
          ("mode 1 (shot 0.10, as shipped)", run(1.0)),
          ("mode 1, shot_noise = 0", run(1.0, nv_shot_noise=0.0)),
          ("mode 0 for reference", run(0.0))]

gap = 8
canvas = Image.new("RGB", (W * 2 + gap * 3, H * 2 + 120), (16, 16, 18))
d = ImageDraw.Draw(canvas)
d.text((6, 5), "THE SHOT-NOISE TERM'S CONTRIBUTION TO THE PICTURE, AND THE MODE-1 BANDING", fill=(242, 242, 242))
for i, (tag, a) in enumerate(panels):
    x = gap + (i % 2) * (W + gap)
    y = 24 + (i // 2) * (H + 44)
    canvas.paste(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)), (x, y))
    d.text((x, y + H + 3), "%s   row-difference %.5f" % (tag, rowsig(a)), fill=(232, 232, 232))
out = os.path.join(OUT, "SHOT_CONTRIB.png")
canvas.save(out)
print("   %-34s %s" % ("", "row-difference"))
for tag, a in panels:
    print("   %-34s %.5f" % (tag, rowsig(a)))
print("   %-34s %.5f" % ("the source photograph", rowsig(src)))
print()
print("   written: %s" % os.path.relpath(out, ROOT))
