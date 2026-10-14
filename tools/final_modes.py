"""
The final test figure: the footage, the input, and the model across the three modes, with the figures beside each panel.

**What this round changed, and what it did not.**

  * **`illuminator` was fixed.**  It reached its term and the term never used it -- the field was normalised by its own
    mean and multiplied by a fixed constant, so 0.15 and 3.00 gave bit-identical output.  It now scales the field, and the
    default is 1.20, which reproduces the previous picture (`0.062 * 1.20` against the old fixed `0.062`).
  * **The highlight shortfall was DIAGNOSED but not closed.**  The model's top is **0.8686 with nothing clipped** where
    the footage reaches **1.0 with 0.66% clipped**.  Traced through the camera, the loss is in the display branch:
    `monochrome_green`'s weights sum to 1, so a full-scale luminance cannot push any channel past 0.43 and the output's
    own ceiling is 0.853.  **`highlight_saturation` is implemented and reaches 0.9752, but it is a global gain**: across
    20 drive-and-exposure pairs, none satisfies the mean, the maximum and the clipped fraction together.  **Its default is
    therefore 1.0, i.e. it does not act**, and the mechanism is recorded as partly implemented rather than fixed.

**So the figure is honest about both**: it shows a render that is recognisably the scene in the right green with the mean
and the colour distribution where the footage has them, and highlights that stop short of the top.

**Every panel carries its own figures** -- mean, p90, p99, max and the clipped fraction -- because a picture without its
measurement beside it is how a plausible-looking wrong mechanism survives, and because the highlight shortfall is
invisible to every whole-frame quantity that was used earlier in this work.
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


def figures(a):
    g = np.clip(np.asarray(a, np.float32)[sl] @ COEF, 0, 1)
    sc = np.clip(np.asarray(a, np.float32), 0, 1)[sl]
    r, gg, b = sc[..., 0], sc[..., 1], sc[..., 2]
    u, v = r - gg, b - gg
    m = np.sqrt(u * u + v * v)
    k = m > max(1e-4, float(np.quantile(m, 0.5)))
    ang = np.arctan2(v, u)
    hue = float(np.angle(np.exp(1j * (ang[k] - ang[k].mean()))).std()) if k.any() else 0.0
    return dict(mean=float(g.mean()), p90=float(np.quantile(g, 0.90)), p99=float(np.quantile(g, 0.99)),
                mx=float(g.max()), clip=float((g >= 0.98).mean()),
                hue=hue, sat=float(m[k].mean()) if k.any() else 0.0)


SRC = load(PHOTO)


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


def run(mode):
    layers = {"camera": dict(OFF, nv_mode=mode),
              "format": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0},
              "player": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0},
              "codec": {"dv_enabled": 0.0}, "display": {"enabled": 0.0}}
    rr = Renderer(One(), W, H, 25.0, merge_values(BASE, layers), seed=20240101)
    return np.clip(np.asarray(rr.render_frame(0), np.float32), 0, 1)


panels = [("the source photograph", SRC)]
for name, p in REF:
    panels.append(("the real %s" % name, load(p)))
panels.append(("the model, NIGHTSHOT 0 (mode off)", run(0.0)))
panels.append(("the model, NIGHTSHOT 1", run(1.0)))
panels.append(("the model, NIGHTSHOT 2 (Super)", run(2.0)))

print("   %-32s %-8s %-8s %-8s %-8s %-9s %-8s %s"
      % ("", "mean", "p90", "p99", "max", "clipped", "hue", "sat"))
for tag, a in panels:
    f = figures(a)
    print("   %-32s %-8.4f %-8.4f %-8.4f %-8.4f %-9.5f %-8.4f %.4f"
          % (tag, f["mean"], f["p90"], f["p99"], f["mx"], f["clip"], f["hue"], f["sat"]))

gap = 7
cw, ch = W // 3, (H - int(H * OSD)) // 3
cols = 3
rows = (len(panels) + cols - 1) // cols
canvas = Image.new("RGB", ((cw + gap) * cols + gap, (ch + 46) * rows + 26), (16, 16, 18))
d = ImageDraw.Draw(canvas)
d.text((6, 5), "THE MODEL IN ALL THREE MODES, AGAINST THE FOOTAGE -- with each panel's own figures",
       fill=(242, 242, 242))
for i, (tag, a) in enumerate(panels):
    x = gap + (i % cols) * (cw + gap)
    y = 26 + (i // cols) * (ch + 46)
    canvas.paste(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).resize((cw, ch), Image.LANCZOS), (x, y))
    f = figures(a)
    d.text((x, y + ch + 3), tag, fill=(238, 238, 238))
    d.text((x, y + ch + 16), "mean %.3f  p90 %.3f  p99 %.3f" % (f["mean"], f["p90"], f["p99"]), fill=(214, 214, 214))
    d.text((x, y + ch + 29), "max %.3f  clipped %.4f  hue %.3f  sat %.3f"
           % (f["mx"], f["clip"], f["hue"], f["sat"]), fill=(214, 214, 214))
out = os.path.join(OUT, "FINAL_MODES.png")
canvas.save(out)
print()
print("   written: %s" % os.path.relpath(out, ROOT))
