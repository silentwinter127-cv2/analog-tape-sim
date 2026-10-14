"""
The test figure, on the comparison that is actually fair: the camcorder's own frames as the input.

**Why this input and not the photograph.**  The calibration had been fitted against a photograph of an RGB case with red,
magenta and pink LED emitters, and those are *visible-light* sources: under this camcorder's NightShot -- IR-cut filter
removed, near-infrared lamp on -- they contribute to the picture in a way a visible-light photograph does not record.  **So
that input could never reach the footage**, and every "still does not look the same" report was a reference-frame question
rather than a tuning one.

**And the footage's own spread is measured**, one frame per second across the whole NightShot stretch:

    t = 52-53s    mean 0.11-0.18   R/G 0.85-0.81     (a transitional moment)
    t = 54-67s    mean 0.22-0.30   R/G 0.49-0.64     (NIGHTSHOT 1)
    t = 68-82s    mean 0.33-0.39   R/G 0.37-0.55     (NIGHTSHOT 2)

**so the target is a range, not a frame** -- which is the calibration the user chose.

**The figure shows four things side by side**: a real frame, the model fed that same frame, the model fed a frame from
another part of the stretch (so the range is visible rather than asserted), and the input photograph for contrast.  **Each
panel carries its own figures**, because a picture without its measurement beside it is how a plausible-looking wrong
mechanism survives, and because this work has thirteen recorded instances of a statistic that could not see its subject.
"""
from __future__ import annotations

import glob
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
FRAMES = sorted(glob.glob(os.path.join(ROOT, ".server", "nvframes", "*.png")))
PHOTO = os.path.join(r"C:\Users\Administrator\.dsh\attachments\v1\objects", "e0",
                     "e0e36b261fbbad42777702d3b775fcf30d4f8505fdc104e08b24405e69eda14c")
OUT = os.path.join(ROOT, "tools", "calib")
W, H, OSD = 533, 400, 0.22
sl = slice(int(H * OSD), H)


def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0
    return np.asarray(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS),
                      np.float32) / 255.0


def fig(a):
    x = np.clip(np.asarray(a, np.float32), 0, 1)[sl]
    g = np.clip(x @ COEF, 0, 1)
    r, gg, b = (float(x[..., i].mean()) for i in range(3))
    u, v = x[..., 0] - x[..., 1], x[..., 2] - x[..., 1]
    m = np.sqrt(u * u + v * v)
    k = m > max(1e-4, float(np.quantile(m, 0.5)))
    return dict(mean=float(g.mean()), std=float(g.std()), p1=float(np.quantile(g, 0.01)),
                p95=float(np.quantile(g, 0.95)), p99=float(np.quantile(g, 0.99)),
                clip=float((g >= 0.98).mean()), RG=r / max(1e-6, gg),
                hue=float(np.angle(np.exp(1j * np.arctan2(v, u)[k]).mean()) * 180.0 / np.pi) if k.any() else 0.0,
                dark=float((g < 0.02).mean()))


BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])
OFF = {"halation": 0.0, "ccd_smear": 0.0, "flare_defocus": 0.0, "lens_flare": 0.0,
       "overflow_drain": 0.0, "chroma_aberration": 0.0, "sensor_noise": 0.0, "noise_grain": 0.0,
       "fixed_pattern": 0.0, "spike_spread": 0.0, "motion_blur": 0.0, "stuck_fraction": 0.0,
       "bloom_threshold": 0.0, "overflow_width": 0.0, "sharpness": 1.0}


class One:
    height, width = H, W
    fps = 25.0
    frame_count = 1

    def __init__(self, a):
        self.a = a

    def frame(self, i):
        return self.a


def run(src, **cam):
    layers = {"camera": dict(OFF, nv_mode=2.0, **cam),
              "format": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0},
              "player": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0},
              "codec": {"dv_enabled": 0.0}, "display": {"enabled": 0.0}}
    rr = Renderer(One(src), W, H, 25.0, merge_values(BASE, layers), seed=20240101)
    return np.clip(np.asarray(rr.render_frame(0), np.float32), 0, 1)


COLS = ["mean", "std", "p1", "p95", "p99", "clip", "RG", "hue"]
picks = [8, 20, 30]                                            # t = 60s, 72s, 82s
print("   %-30s %s" % ("", "  ".join("%-9s" % c for c in COLS)))
panels = []
for i in picks:
    src = load(FRAMES[i])
    out = run(src)
    panels.append(("t=%ds  the real frame (the target)" % (52 + i), src, fig(src)))
    panels.append(("t=%ds  the model fed that frame" % (52 + i), out, fig(out)))
    print("   %-30s %s" % ("t=%ds  real" % (52 + i), "  ".join("%-9.4f" % fig(src)[c] for c in COLS)))
    print("   %-30s %s" % ("      model", "  ".join("%-9.4f" % fig(out)[c] for c in COLS)))
photo = load(PHOTO)
print("   %-30s %s" % ("the input photograph", "  ".join("%-9.4f" % fig(photo)[c] for c in COLS)))
panels.append(("the input photograph (a different scene)", photo, fig(photo)))

gap, cols = 7, 2
cw, ch = W // cols, (H - int(H * OSD)) // cols
rows = (len(panels) + cols - 1) // cols
canvas = Image.new("RGB", ((cw + gap) * cols + gap, (ch + 52) * rows + 28), (16, 16, 18))
d = ImageDraw.Draw(canvas)
d.text((6, 5), "THE MODEL FED THE CAMCORDER'S OWN FRAMES -- the comparison the input photograph could never make",
       fill=(242, 242, 242))
for i, (tag, arr, s) in enumerate(panels):
    x = gap + (i % cols) * (cw + gap)
    y = 26 + (i // cols) * (ch + 52)
    canvas.paste(Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).resize((cw, ch), Image.LANCZOS), (x, y))
    d.text((x, y + ch + 3), tag, fill=(238, 238, 238))
    d.text((x, y + ch + 16), "mean %.3f  std %.3f  p1 %.3f  p95 %.3f" % (s["mean"], s["std"], s["p1"], s["p95"]),
           fill=(214, 214, 214))
    d.text((x, y + ch + 29), "R/G %.3f  hue %+.1f  below .02 %.4f" % (s["RG"], s["hue"], s["dark"]),
           fill=(214, 214, 214))
out = os.path.join(OUT, "FED_REAL_FRAMES.png")
canvas.save(out)
print()
print("   written: %s" % os.path.relpath(out, ROOT))
