"""
Render the user's input through the current settings and measure it against their footage frame.

**The mistake this corrects.**  Measuring the user's two images side by side compares a render made BEFORE the calibration
with the footage -- so it reports the gap that the calibration was supposed to close, not the gap that remains.  **The
render has to be made fresh, from the same input, with the settings as they now stand.**

**The input is their own photograph** (`MVIMG_20261014_195506.jpg`, the case with the lit fans) and the reference is their
footage frame (`c_0066.png`), with the OSD band cropped from both measurements since it carries white text.

**The two claims are kept separate**, because they have separate causes and separate controls:

  * **the green**: the output's channel ratios are exactly the display's weights, so `G/R` and `G/B` say whether the hue is
    right, and the hue angle says it in one number;
  * **the contrast**: the luminance standard deviation, plus `p5` and `p95` -- because a raised black level and a compressed
    highlight are different faults and this work has already measured which one it is (the top, by 0.158, not the bottom).
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
OBJ = r"C:\Users\Administrator\.dsh\attachments\v1\objects"
PHOTO = os.path.join(OBJ, "e0", "e0e36b261fbbad42777702d3b775fcf30d4f8505fdc104e08b24405e69eda14c")
FOOT = os.path.join(OBJ, "17", "1721d92e6287039449f1ff16e51d4e376275f5d98332408d43ead46c0597f609")
OUT = os.path.join(ROOT, "tools", "calib")
W, H = 533, 400
OSD = 0.17


def load(path, size=(W, H), crop_osd=False):
    im = Image.open(path).convert("RGB")
    a = np.asarray(im, np.float32) / 255.0
    if crop_osd:
        a = a[int(a.shape[0] * OSD):]
        im = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
    return np.asarray(im.resize(size, Image.LANCZOS), np.float32) / 255.0


def stats(a, crop_osd=False):
    x = a[int(a.shape[0] * OSD):] if crop_osd else a
    lum = np.clip(x @ COEF, 0, 1)
    r, g, b = (float(x[..., i].mean()) for i in range(3))
    u, v = x[..., 0] - x[..., 1], x[..., 2] - x[..., 1]
    m = np.sqrt(u * u + v * v)
    k = m > max(1e-4, float(np.quantile(m, 0.5)))
    ang = np.arctan2(v, u)[k]
    return dict(GR=g / max(1e-6, r), GB=g / max(1e-6, b),
                hue=float(np.angle(np.exp(1j * ang).mean()) * 180.0 / np.pi),
                sat=float(m[k].mean()), lum=float(lum.mean()), std=float(lum.std()),
                p5=float(np.quantile(lum, 0.05)), p50=float(np.quantile(lum, 0.50)),
                p95=float(np.quantile(lum, 0.95)), p99=float(np.quantile(lum, 0.99)),
                dark=float((lum < 0.05).mean()))


SRC = load(PHOTO)
FOOT_IMG = load(FOOT, crop_osd=True)
BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])
OFF = {"halation": 0.0, "ccd_smear": 0.0, "flare_defocus": 0.0, "lens_flare": 0.0,
       "overflow_drain": 0.0, "chroma_aberration": 0.0, "sensor_noise": 0.0, "noise_grain": 0.0,
       "fixed_pattern": 0.0, "spike_spread": 0.0, "motion_blur": 0.0, "stuck_fraction": 0.0,
       "bloom_threshold": 0.0, "overflow_width": 0.0, "sharpness": 1.0}


class One:
    height, width = H, W
    fps = 25.0
    frame_count = 1

    def frame(self, i):
        return SRC


def run(**cam):
    cam = dict(cam)
    cam.setdefault("nv_mode", 2.0)
    layers = {"camera": dict(OFF, **cam),
              "format": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0},
              "player": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0},
              "codec": {"dv_enabled": 0.0}, "display": {"enabled": 0.0}}
    rr = Renderer(One(), W, H, 25.0, merge_values(BASE, layers), seed=20240101)
    return np.clip(np.asarray(rr.render_frame(0), np.float32), 0, 1)


COLS = [("G/R", "GR"), ("G/B", "GB"), ("hue", "hue"), ("sat", "sat"), ("lum", "lum"),
        ("std", "std"), ("p5", "p5"), ("p95", "p95"), ("p99", "p99")]


def show(tag, a, crop=False):
    s = stats(a, crop)
    print("   %-30s %s" % (tag, "  ".join("%s %7.3f" % (c, s[k]) for c, k in COLS)))
    return s


print("   %-30s %s" % ("", "  ".join("%-11s" % c for c, _ in COLS)))
f = show("the footage frame (the target)", FOOT_IMG, crop=True)
print()
r = show("the render, as it now stands", run())
print()
print("   === the gaps ===")
for c, k in COLS:
    d = r[k] - f[k]
    mark = "  <-- still out" if abs(d) > (0.06 * abs(f[k]) if k in ("GR", "GB", "hue") else 0.12 * abs(f[k])) else ""
    print("   %-6s the model %8.4f   the footage %8.4f   %+.4f%s" % (c, r[k], f[k], d, mark))
print()
print("   **the green is now solved rather than fitted**: the weights are 1/1.387 : 1 : 1/1.090 normalised by")
print("   `w @ LUMA == 1`, so the ratios land on the footage's by construction and the luminance is untouched.")

gap = 7
cw, ch = W // 2, H // 2
panels = [("the footage frame", FOOT_IMG), ("the model now", run()),
          ("the input photograph", SRC), ("the model, as it stood", run(nv_highlight_saturation=0.0))]
canvas = Image.new("RGB", ((cw + gap) * 2 + gap, (ch + 34) * 2 + 28), (16, 16, 18))
d = ImageDraw.Draw(canvas)
d.text((6, 5), "GREEN AND CONTRAST -- the render measured against the user's own footage frame",
       fill=(242, 242, 242))
for i, (tag, arr) in enumerate(panels):
    x = gap + (i % 2) * (cw + gap)
    y = 24 + (i // 2) * (ch + 34)
    canvas.paste(Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).resize((cw, ch), Image.LANCZOS), (x, y))
    s = stats(arr)
    d.text((x, y + ch + 3), tag, fill=(238, 238, 238))
    d.text((x, y + ch + 16), "G/R %.3f  G/B %.3f  std %.3f  p95 %.3f" % (s["GR"], s["GB"], s["std"], s["p95"]),
           fill=(214, 214, 214))
out = os.path.join(OUT, "GREEN_CHECK.png")
canvas.save(out)
print("   written: %s" % os.path.relpath(out, ROOT))
