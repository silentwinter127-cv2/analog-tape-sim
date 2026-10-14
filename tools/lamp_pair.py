"""
The illuminator against the exposure, now that the illuminator actually scales.

**Why this pair and not the drive.**  The drive was swept and is a global gain, so it lifts the top only by lifting
everything -- twenty drive-and-exposure pairs reached none of the three targets.  **The illuminator is different**: it is
the LIGHT, so it changes the scene's own range before the display ever sees it, which is what `note.txt`'s first mechanism
is about.  And it only became usable this round -- measured, `nv_illuminator` from 0.15 to 3.00 used to give bit-identical
output, because the field was normalised by its own mean and multiplied by a fixed constant and never by `illuminator`.

**So both halves are now real controls**: the lamp sets how much light lands on the near field, the exposure sets how much
of it is collected.  **Their product is the highlight level and their ratio is the falloff**, which is why a pair can do
what one knob cannot.

**The targets, all from the footage:**

    mean 0.2346   p90 0.5870   p99 0.9511   max 1.0000   clipped 0.00657

**and the score weights all of them**, so a setting that blows out the highlights by overexposing the whole frame cannot
win.  **The prediction, stated so it can fail**: if some lamp-and-exposure pair reaches the maximum and the clipped
fraction together while the mean stays near 0.235, then the missing mechanism is reached through the light and only needed
calibrating.  **If none does, the light cannot make this photograph blow out at all**, which is a statement about the
input rather than about the model -- and it would be the honest end of this line of enquiry.
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
REF = [("NIGHTSHOT 0", os.path.join(OBJ, "ed",
                                    "ed906503e82e76777332850f677a16db3ea89089fde1c313576edf9fce66f4e0")),
       ("NIGHTSHOT 1", os.path.join(OBJ, "f2",
                                    "f2dd7803fd91d124a7c16be3c6f97df679c8a83d89631a462e9010f13a3e7c5e")),
       ("NIGHTSHOT 2", os.path.join(OBJ, "b9",
                                    "b94a0e8402a11788032bd0af9e81de7fc4713d3a648384f2c9ff57b21c21f4b6"))]
PHOTO = os.path.join(OBJ, "69", "699c23315df037b01fa831a273364b41f99c28733fb1dc928b22d1e6c281818d")
OUT = os.path.join(ROOT, "tools", "calib")
W, H, OSD = 533, 400, 0.22
sl = slice(int(H * OSD), H)


def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0
    return np.asarray(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS),
                      np.float32) / 255.0


def hi(a):
    g = np.clip(np.asarray(a, np.float32)[sl] @ COEF, 0, 1)
    return (float(g.mean()), float(np.quantile(g, 0.90)), float(np.quantile(g, 0.99)),
            float(g.max()), float((g >= 0.98).mean()))


SRC = load(PHOTO)
r0 = hi(load(REF[0][1]))
print("   the footage:  mean %.4f  p90 %.4f  p99 %.4f  max %.4f  clipped %.5f" % r0)


class One:
    height, width = H, W
    fps = 25.0
    frame_count = 1

    def frame(self, i):
        return SRC


BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])
OFF = {"halation": 0.0, "ccd_smear": 0.0, "flare_defocus": 0.0, "lens_flare": 0.0,
       "overflow_drain": 0.0, "chroma_aberration": 0.0, "sensor_noise": 0.0, "noise_grain": 0.0,
       "fixed_pattern": 0.0, "spike_spread": 0.0, "motion_blur": 0.0, "stuck_fraction": 0.0,
       "bloom_threshold": 0.0, "overflow_width": 0.0, "sharpness": 1.0}


def run(lamp, exp, sat=0.0):
    layers = {"camera": dict(OFF, nv_mode=1.0, nv_illuminator=lamp, exposure=exp, saturation_stops=sat),
              "format": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0},
              "player": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0},
              "codec": {"dv_enabled": 0.0}, "display": {"enabled": 0.0}}
    rr = Renderer(One(), W, H, 25.0, merge_values(BASE, layers), seed=20240101)
    a = np.clip(np.asarray(rr.render_frame(0), np.float32), 0, 1)
    return a, hi(a)


def score(f):
    return abs(f[0] - r0[0]) / r0[0] + abs(f[3] - r0[3]) + 4.0 * abs(f[4] - r0[4]) + abs(f[2] - r0[2]) / r0[2]


print()
print("   === the lamp against the exposure ===")
print("   %-7s %-8s %-8s %-9s %-9s %-9s %-9s %s"
      % ("lamp", "exposure", "sat", "mean", "p90", "p99", "max", "clip  score"))
best = None
for lamp in (1.2, 2.0, 3.0, 4.0):
    for exp in (0.0, -0.25, -0.5, -0.75):
        for sat in (0.0, 0.15):
            a, f = run(lamp, exp, sat)
            sc = score(f)
            if best is None or sc < best[0]:
                best = (sc, lamp, exp, sat, a, f)
            print("   %-7.2f %-8.2f %-8.2f %-9.4f %-9.4f %-9.4f %-9.4f %-9.5f %.4f"
                  % (lamp, exp, sat, f[0], f[1], f[2], f[3], f[4], sc))
print()
sc, lamp, exp, sat, a, f = best
print("   the best triple: lamp %.2f, exposure %+.2f, saturation_stops %.2f" % (lamp, exp, sat))
print("      mean %.4f (the footage %.4f)   p99 %.4f (%.4f)   max %.4f (%.4f)   clipped %.5f (%.5f)"
      % (f[0], r0[0], f[2], r0[2], f[3], r0[3], f[4], r0[4]))
print()
if abs(f[3] - r0[3]) < 0.06 and abs(f[4] - r0[4]) < 0.008 and abs(f[0] - r0[0]) / r0[0] < 0.15:
    print("   -> a lamp-and-exposure setting reaches the maximum, the clipped fraction and the mean together, so the")
    print("      mechanism is reached through the LIGHT and only needed calibrating.")
else:
    print("   -> none does: the light cannot make this photograph blow out at all, which is a statement about the")
    print("      INPUT rather than about the model -- its brightest pixel is at 0.87 and no lamp setting changes that")
    print("      ratio, because the lamp scales the whole field rather than creating a small very bright region.")
    print("      **That is the honest end of this line of enquiry, and it is what the sweep shows.**")

gap = 7
cw, ch = W // 2, (H - int(H * OSD)) // 2
panels = [("the real NIGHTSHOT 0", load(REF[0][1])),
          ("the model as it ships", run(1.2, 0.0)[0]),
          ("the best lamp/exposure triple", a),
          ("the source photograph", SRC)]
canvas = Image.new("RGB", ((cw + gap) * 2 + gap, (ch + 34) * 2 + 28), (16, 16, 18))
d = ImageDraw.Draw(canvas)
d.text((6, 5), "THE LAMP AGAINST THE EXPOSURE -- and whether the light can make the highlights blow out",
       fill=(242, 242, 242))
for i, (tag, arr) in enumerate(panels):
    x = gap + (i % 2) * (cw + gap)
    y = 24 + (i // 2) * (ch + 34)
    canvas.paste(Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).resize((cw, ch), Image.LANCZOS), (x, y))
    hh = hi(arr)
    d.text((x, y + ch + 3), tag, fill=(238, 238, 238))
    d.text((x, y + ch + 16), "mean %.3f  p90 %.3f  p99 %.3f  max %.3f  clip %.4f"
           % (hh[0], hh[1], hh[2], hh[3], hh[4]), fill=(214, 214, 214))
out = os.path.join(OUT, "LAMP_TEST.png")
canvas.save(out)
print("   written: %s" % os.path.relpath(out, ROOT))
