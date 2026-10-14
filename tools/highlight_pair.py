"""
Set the highlight drive and the exposure TOGETHER, since the drive raises the whole range and the exposure lowers it.

**Measured, and it is the reason a single knob will not do:**

    drive   mean     p90      p99      max      clipped
    1.00    0.2370   0.5499   0.8061   0.8686   0.00000
    1.35    0.3060   0.7163   0.8844   0.9366   0.00000
    1.90    0.3964   0.8474   0.9481   0.9752   0.00000
    the footage  0.2346   0.5870   0.9511   **1.0000**  **0.00657**

**so the drive does raise the top** -- 0.8686 to 0.9752 -- **but it raises every other level with it**, leaving the mean
36% high at the setting that comes closest.  **The exposure is the counterweight**: it scales the whole picture before the
display, so the pair can put the top where the footage has it while the mean stays where the footage has that.

**And the physics says they are genuinely separate controls**: the drive is the viewfinder's gain, the exposure is how
much light was collected.  A night-vision shot that clips its highlights is one where the **lamp is strong enough** that
the bright surfaces saturate -- which is a statement about the scene's own range, not about a global gain.

**So the pair is swept and scored on all three targets at once** -- the mean, the maximum and the clipped fraction --
because a setting that matches one and misses the others is what the single-knob sweeps kept producing.  **The prediction:
if some pair reaches mean ~0.235, max ~1.0 and clipped ~0.007 together, the mechanism is now complete and only needed
calibrating.  If no pair does, the drive is still the wrong shape** -- a global gain cannot make a small part of the frame
clip while the rest does not.
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


def run(drive, exp):
    layers = {"camera": dict(OFF, nv_mode=1.0, nv_highlight_saturation=drive, exposure=exp),
              "format": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0},
              "player": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0},
              "codec": {"dv_enabled": 0.0}, "display": {"enabled": 0.0}}
    rr = Renderer(One(), W, H, 25.0, merge_values(BASE, layers), seed=20240101)
    a = np.clip(np.asarray(rr.render_frame(0), np.float32), 0, 1)
    return a, hi(a)


print()
print("   === the drive and the exposure together ===")
print("   %-7s %-8s %-9s %-9s %-9s %-9s %-9s %s" % ("drive", "exposure", "mean", "p90", "p99", "max", "clip", "score"))
best = None
for drive in (1.0, 1.2, 1.35, 1.6, 1.9):
    for exp in (-0.4, -0.2, 0.0, 0.2):
        a, (m, p90, p99, mx, cl) = run(drive, exp)
        #: the score weights all three targets, so a setting that matches one and misses the others cannot win
        score = (abs(m - r0[0]) / r0[0] + abs(mx - r0[3]) + 4.0 * abs(cl - r0[4]))
        if best is None or score < best[0]:
            best = (score, drive, exp, a, (m, p90, p99, mx, cl))
        print("   %-7.2f %-8.2f %-9.4f %-9.4f %-9.4f %-9.4f %-9.5f %.4f" % (drive, exp, m, p90, p99, mx, cl, score))
print()
sc, drive, exp, a, st = best
print("   the best pair: drive %.2f, exposure %+.2f" % (drive, exp))
print("      mean %.4f (the footage %.4f)   max %.4f (%.4f)   clipped %.5f (%.5f)"
      % (st[0], r0[0], st[3], r0[3], st[4], r0[4]))
print()
if abs(st[3] - r0[3]) < 0.06 and abs(st[4] - r0[4]) < 0.01 and abs(st[0] - r0[0]) / r0[0] < 0.15:
    print("   -> a pair reaches all three targets, so the mechanism is complete and only needed calibrating.")
else:
    print("   -> no pair reaches all three: a GLOBAL gain cannot make a small part of the frame clip while the rest")
    print("      does not, which is what saturation means.  The drive is the wrong shape and this is the finding.")

gap = 7
cw, ch = W // 2, (H - int(H * OSD)) // 2
panels = [("the real NIGHTSHOT 0", load(REF[0][1])), ("the model, before this round", run(1.0, 0.0)[0]),
          ("the model, best pair", a), ("the source photograph", SRC)]
canvas = Image.new("RGB", ((cw + gap) * 2 + gap, (ch + 34) * 2 + 28), (16, 16, 18))
d = ImageDraw.Draw(canvas)
d.text((6, 5), "HIGHLIGHTS -- the footage, the model before and after the drive, and the input", fill=(242, 242, 242))
for i, (tag, arr) in enumerate(panels):
    x = gap + (i % 2) * (cw + gap)
    y = 24 + (i // 2) * (ch + 34)
    canvas.paste(Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).resize((cw, ch), Image.LANCZOS), (x, y))
    hh = hi(arr)
    d.text((x, y + ch + 3), tag, fill=(238, 238, 238))
    d.text((x, y + ch + 16), "mean %.3f  p90 %.3f  p99 %.3f  max %.3f  clip %.4f"
           % (hh[0], hh[1], hh[2], hh[3], hh[4]), fill=(214, 214, 214))
out = os.path.join(OUT, "HIGHLIGHT_FIX.png")
canvas.save(out)
print("   written: %s" % os.path.relpath(out, ROOT))
