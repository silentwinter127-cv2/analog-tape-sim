"""
The final result: the chain at the settings measured to keep the polarity, beside the footage.

**What changed and why it matters.**  Six of the player's own parameters were being read from the **format** layer --
`chroma_bandwidth`, `luma_bandwidth`, `comb_depth`, `cross_color`, `cross_luma` and `burst_phase_error` -- so the
interface's `player.*` values for those did nothing, and the stage silently used the format layer's.  **Measured, the
player's chroma bandwidth read 2.000 MHz for every value requested, including 0.5 and 3.0.**

**And that was the inversion.**  At 2 MHz the player's second split takes **0.770 of the composite** as chroma, and the
comb's `1/(1-0.55)` multiplies that by **2.143**, so the chroma band carries **1.099 of the composite** and
`luma = comp - chroma` goes negative where the picture is bright.  With the layer fixed and the schema's own 0.75 MHz:

    player chroma_bw   polarity     contrast
    0.50 MHz           +0.935       0.1928
    0.75 MHz           +0.935       0.1928      <- the schema's default
    1.00 MHz           +0.935       0.1928
    2.00 MHz           -0.358       0.1678      <- inverted
    3.00 MHz           -0.358       0.1678

**so the picture is no longer inverted at the default**, and the contrast is **0.95x** of the footage's rather than the
0.64x it has been through this whole investigation.
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

OUT = os.path.join(ROOT, "tools", "calib")
COEF = np.array([0.299, 0.587, 0.114], np.float32)
OBJ = (r"C:\Users\Administrator\.dsh\attachments\v1\objects")
REAL = [("NIGHTSHOT 0", os.path.join(OBJ, "ed",
                                     "ed906503e82e76777332850f677a16db3ea89089fde1c313576edf9fce66f4e0")),
        ("NIGHTSHOT 1", os.path.join(OBJ, "f2",
                                     "f2dd7803fd91d124a7c16be3c6f97df679c8a83d89631a462e9010f13a3e7c5e")),
        ("NIGHTSHOT 2", os.path.join(OBJ, "b9",
                                     "b94a0e8402a11788032bd0af9e81de7fc4713d3a648384f2c9ff57b21c21f4b6"))]
PHOTO = os.path.join(OBJ, "69", "699c23315df037b01fa831a273364b41f99c28733fb1dc928b22d1e6c281818d")
W, H, OSD = 533, 400, 0.22
sl = slice(int(H * OSD), H)


def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0
    return np.asarray(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS),
                      np.float32) / 255.0


def lum(a):
    return np.clip(np.asarray(a, np.float32)[sl] @ COEF, 0, 1)


def corr(a, b):
    x = (a - a.mean()).ravel()
    y = (b - b.mean()).ravel()
    d = float(np.sqrt((x * x).sum() * (y * y).sum()))
    return float((x * y).sum() / d) if d > 1e-12 else 0.0


src = load(PHOTO)
real0 = load(REAL[0][1])
ls, lr = lum(src), lum(real0)


class One:
    height, width = H, W
    fps = 25.0
    frame_count = 1

    def frame(self, i):
        return src


BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])
#: **the schema's own bandwidths and comb**, with the defect terms neutral so the picture is what is being shown
CHAIN = {
    "camera": {"nv_mode": 1.0, "nv_green_mix": 1.0, "nv_illuminator": 0.15,
               "halation": 0.0, "ccd_smear": 0.0, "flare_defocus": 0.0, "lens_flare": 0.0,
               "overflow_drain": 0.0, "chroma_aberration": 0.0,
               "sensor_noise": 0.0, "noise_grain": 0.0, "fixed_pattern": 0.0, "spike_spread": 0.0,
               "motion_blur": 0.0, "stuck_fraction": 0.0, "bloom_threshold": 0.0, "overflow_width": 0.0,
               "sharpness": 1.0},
    "format": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0},
    "player": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0, "comb_depth": 0.55},
    "codec": {"dv_enabled": 0.0},
    "display": {"enabled": 0.0},
}

print("   the chain: night-vision on, the defect terms neutral, the schema's own bandwidths (0.75 / 3.0 MHz),")
print("   comb_depth 0.55, DV off, the display off.")
print()
print("   %-26s %-10s %-11s %-10s %s" % ("", "luma", "contrast", "polarity", "vs the footage"))
rl, rc = float(lr.mean()), float(lr.std())
print("   %-26s %-10.4f %-11.4f %s" % ("the real NIGHTSHOT 0", rl, rc, "-"))

print()
panels = []
for label, cam in (("NIGHTSHOT 0", dict(nv_chroma_residue=0.0, nv_chroma_noise=0.0)),
                   ("NIGHTSHOT 0 + colour noise", dict(nv_chroma_residue=0.0, nv_chroma_noise=0.15)),
                   ("NIGHTSHOT 1", dict(nv_chroma_residue=0.35, nv_chroma_noise=0.15))):
    layers = dict(CHAIN, camera=dict(CHAIN["camera"], **cam))
    rr = Renderer(One(), W, H, 25.0, merge_values(BASE, layers), seed=20240101)
    a = np.clip(np.asarray(rr.render_frame(0), np.float32), 0, 1)
    l = lum(a)
    panels.append((label, a, float(l.mean()), float(l.std()), corr(l, ls)))
    print("   %-26s %-10.4f %-11.4f %+-10.3f luma %+.0f%%, contrast %.2fx"
          % (label, float(l.mean()), float(l.std()), corr(l, ls),
             100 * (float(l.mean()) - rl) / rl, float(l.std()) / rc))

gap = 10
canvas = Image.new("RGB", (W * 3 + gap * 2, H * 2 + 104), (18, 18, 20))
d = ImageDraw.Draw(canvas)
d.text((6, 6), "THE REAL FRAMES (the target)", fill=(240, 240, 240))
for i, (name, p) in enumerate(REAL):
    a = load(p)
    l = lum(a)
    canvas.paste(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)), (i * (W + gap), 24))
    d.text((i * (W + gap) + 6, 24 + H + 2), "%s  luma %.4f  contrast %.4f" % (name, float(l.mean()), float(l.std())),
           fill=(230, 230, 230))
d.text((6, 24 + H + 28), "THE MODEL -- the player's bandwidth now comes from its OWN layer (polarity +0.935, was -0.358)",
       fill=(240, 240, 240))
for i, (label, a, lm, lst, c) in enumerate(panels):
    canvas.paste(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)), (i * (W + gap), 24 + H + 48))
    d.text((i * (W + gap) + 6, 24 + H + 48 + H + 2),
           "%s  luma %.4f  contrast %.4f  polarity %+.3f" % (label, lm, lst, c), fill=(230, 230, 230))
out = os.path.join(OUT, "RESULT_FIXED.png")
canvas.save(out)
print()
print("   written: %s" % os.path.relpath(out, ROOT))
