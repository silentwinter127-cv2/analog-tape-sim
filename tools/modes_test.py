"""
The three-mode test: NightShot 0/1/2 rendered at the schema's own settings, beside the real frames.

**What is fixed and what is not, stated before the pictures.**  The polarity is now correct -- the picture correlates
**+0.93** with its source where it was **-0.358**, from the layer fix in `player.py`.  **The contrast is still short**:
the mode-off chain reaches **0.94x** the footage's contrast and the mode-on chain **0.68x**, so the night-vision scene
stage still costs about a quarter of it.  That gap is left visible rather than hidden, because the pictures are the point
of the test and a panel that flatters the model would not be a test.

**The three modes differ by their own parameters**, since that is what the interface's selector sets:

    mode 0   the mode off, so the chain's own rendering
    mode 1   NightShot: lamp on, green display on, one frame
    mode 2   Super NightShot: the same and frames accumulated

**and each is rendered through the same chain**, with the schema's own bandwidths and comb, DV off, the display off.  The
real frames are beside them at the same size, and each panel carries its own level, contrast and polarity so the figures
travel with the picture.
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
ls = lum(src)


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
CHAIN = {"format": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0},
         "player": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0},
         "codec": {"dv_enabled": 0.0},
         "display": {"enabled": 0.0}}

#: the three modes, as the interface's selector sets them
MODES = [
    ("NIGHTSHOT 0  (mode off)", dict(nv_mode=0.0)),
    ("NIGHTSHOT 1  (lamp + green, one frame)",
     dict(nv_mode=1.0, nv_green_mix=1.0, nv_illuminator=0.15, nv_accumulate=1.0,
          nv_chroma_residue=0.35, nv_chroma_noise=0.15)),
    ("NIGHTSHOT 2  (Super: frames accumulated)",
     dict(nv_mode=1.0, nv_green_mix=1.0, nv_illuminator=0.15, nv_accumulate=8.0,
          nv_chroma_residue=0.35, nv_chroma_noise=0.15)),
]

print("   the chain: the schema's own bandwidths (0.75 / 3.0 MHz), the defect terms neutral, DV off, display off.")
print()
panels = []
for label, cam in MODES:
    layers = dict(CHAIN, camera=dict(OFF, **cam))
    rr = Renderer(One(), W, H, 25.0, merge_values(BASE, layers), seed=20240101)
    a = np.clip(np.asarray(rr.render_frame(0), np.float32), 0, 1)
    l = lum(a)
    panels.append((label, a, float(l.mean()), float(l.std()), corr(l, ls)))
    print("   %-46s luma %.4f  contrast %.4f  polarity %+.3f"
          % (label, float(l.mean()), float(l.std()), corr(l, ls)))

print()
print("   %-46s luma %.4f  contrast %.4f" % ("the real NIGHTSHOT 0", *[float(x) for x in
                                                                     (lum(load(REAL[0][1])).mean(), lum(load(REAL[0][1])).std())]))
print("   %-46s luma %.4f  contrast %.4f" % ("the source photograph", float(ls.mean()), float(ls.std())))

gap = 10
canvas = Image.new("RGB", (W * 3 + gap * 2, H * 2 + 116), (18, 18, 20))
d = ImageDraw.Draw(canvas)
d.text((6, 6), "THE REAL FRAMES (the target)", fill=(240, 240, 240))
for i, (name, p) in enumerate(REAL):
    a = load(p)
    l = lum(a)
    canvas.paste(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)), (i * (W + gap), 24))
    d.text((i * (W + gap) + 6, 24 + H + 2), "%s   luma %.4f  contrast %.4f" % (name, float(l.mean()), float(l.std())),
           fill=(230, 230, 230))
d.text((6, 24 + H + 30), "THE MODEL -- the three modes, at the schema's own settings (polarity is now +0.93, was -0.358)",
       fill=(240, 240, 240))
for i, (label, a, lm, lst, c) in enumerate(panels):
    canvas.paste(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)), (i * (W + gap), 24 + H + 50))
    d.text((i * (W + gap) + 6, 24 + H + 50 + H + 2),
           "%s\nluma %.4f  contrast %.4f  polarity %+.3f" % (label, lm, lst, c), fill=(230, 230, 230))
out = os.path.join(OUT, "MODES_TEST.png")
canvas.save(out)
print()
print("   written: %s" % os.path.relpath(out, ROOT))
