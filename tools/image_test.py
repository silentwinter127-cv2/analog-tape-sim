"""
The image test: real photographs through the chain, beside the real NIGHTSHOT frames.

**What this is for, and what it is not.**  Every measurement in this series has been a scalar on a synthetic input -- a
ramp, a split field, a spread.  **Those locate faults; they cannot say whether the picture looks right**, and the user's
complaint that started all of it was about how the picture looks.  So this runs real photographs through the chain at the
settings the session settled on, at the size the measurements were taken at, and prints the quantities that were being
tracked beside the footage's own.

**The chain settings are stated in full**, because a figure without its configuration is what produced several wrong
conclusions in this series -- two readings taken under different `dynamic_range` values were once compared as if they
were the same measurement.

**And the comparison is against the three real NIGHTSHOT frames**, resized to the same place, so the two columns are
measured on equal terms: level, contrast, and the per-channel chroma.  **Contrast is the quantity this round changed**,
so it is the one to watch -- it was 0.46 of the footage's before, and the ramp test says the chain now returns 1.188 of
its input rather than 0.800.
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


def stats(a):
    sc = a[sl]
    lum = np.clip(sc @ COEF, 0, 1)
    ch = np.abs(sc - np.clip(sc @ COEF, 0, 1)[..., None]).mean(axis=(0, 1))
    return float(lum.mean()), float(lum.std()), ch


src = load(PHOTO)


class One:
    height, width = H, W
    fps = 25.0
    frame_count = 1

    def frame(self, i):
        return src


BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])
#: the chain this session settled on, stated in full so the figure carries its configuration
CHAIN = {
    "camera": {"nv_mode": 1.0, "nv_green_mix": 1.0, "nv_illuminator": 0.15,
               "halation": 0.0, "ccd_smear": 0.0, "flare_defocus": 0.0, "lens_flare": 0.0,
               "overflow_drain": 0.0, "chroma_aberration": 0.0,
               "sensor_noise": 0.0, "noise_grain": 0.0, "fixed_pattern": 0.0, "spike_spread": 0.0,
               "motion_blur": 0.0, "stuck_fraction": 0.0, "bloom_threshold": 0.0, "overflow_width": 0.0,
               "sharpness": 1.0},
    # **the schema's own bandwidths**, because 2 MHz and above made the player's second split take 0.770 of the
    # composite as chroma and invert the picture; the default 0.75 MHz is the value that keeps `comp - chroma` a luma
    "format": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0},
    "player": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0},
    "codec": {"dv_enabled": 0.0},
    "display": {"enabled": 0.0},
}

print("   the chain, stated in full: night-vision on, the defect terms neutral, `dynamic_range` at the schema's 8.0,")
print("   the medium's bandwidths at 3.0/6.0 MHz, DV off, the display off.")
print()
print("   %-22s %-10s %-12s %-26s %s" % ("", "luma", "contrast", "|chroma| per channel", "vs the footage"))

real_stats = {}
for name, p in REAL:
    a = load(p)
    lm, c, ch = stats(a)
    real_stats[name] = (lm, c)
    print("   real %-17s %-10.4f %-12.4f %-26s %s" % (name, lm, c, np.round(ch, 4), "-"))

print()
panels = []
for label, cam in (("NIGHTSHOT 0", dict(nv_chroma_residue=0.0, nv_chroma_noise=0.0)),
                   ("NIGHTSHOT 0 + colour noise", dict(nv_chroma_residue=0.0, nv_chroma_noise=0.15)),
                   ("NIGHTSHOT 1", dict(nv_chroma_residue=0.35, nv_chroma_noise=0.15))):
    layers = dict(CHAIN, camera=dict(CHAIN["camera"], **cam))
    rr = Renderer(One(), W, H, 25.0, merge_values(BASE, layers), seed=20240101)
    a = np.clip(np.asarray(rr.render_frame(0), np.float32), 0, 1)
    lm, c, ch = stats(a)
    panels.append((label, a, lm, c, ch))
    rl, rc = real_stats["NIGHTSHOT 0"]
    print("   model %-16s %-10.4f %-12.4f %-26s luma %+.0f%%, contrast %.2fx"
          % (label, lm, c, np.round(ch, 4), 100 * (lm - rl) / rl, c / rc))

gap = 10
canvas = Image.new("RGB", (W * 3 + gap * 2, H * 2 + 100), (18, 18, 20))
d = ImageDraw.Draw(canvas)
d.text((6, 6), "THE REAL FRAMES (the target)", fill=(240, 240, 240))
for i, (name, p) in enumerate(REAL):
    a = load(p)
    lm, c, ch = stats(a)
    canvas.paste(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)), (i * (W + gap), 24))
    d.text((i * (W + gap) + 6, 24 + H + 2), "%s  luma %.4f  contrast %.4f" % (name, lm, c),
           fill=(230, 230, 230))
d.text((6, 24 + H + 28), "THE MODEL (cross_luma mis-scaling fixed; the ramp now returns 1.188 instead of 0.800)",
       fill=(240, 240, 240))
for i, (label, a, lm, c, ch) in enumerate(panels):
    canvas.paste(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)), (i * (W + gap), 24 + H + 46))
    d.text((i * (W + gap) + 6, 24 + H + 46 + H + 2), "%s  luma %.4f  contrast %.4f" % (label, lm, c),
           fill=(230, 230, 230))
out = os.path.join(OUT, "IMAGE_TEST.png")
canvas.save(out)
print()
print("   written: %s" % os.path.relpath(out, ROOT))
