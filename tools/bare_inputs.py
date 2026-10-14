"""
The bare photographs as inputs, which is what the user asked for -- no camcorder frames anywhere in the input path.

**Why this is the right material, and it is measurable rather than a preference.**  This work's last several rounds drifted
into feeding the chain the camcorder's OWN NightShot frames, and that is not a test of the model: those frames already
carry the camcorder's green, its tone curve and its black level, so the chain is being asked to preserve a transformation
rather than to perform one.  **A bare photograph is a picture of a scene under visible light, which is what a camera
actually receives.**

**And the naive objection to bare photographs -- that they cannot contain near-infrared information -- is only half true.**
`ir_highlights` does not invent near-infrared content: it reorders the picture by the *material* albedo the vision model
supplies, and the note's own headline is that this material reordering is the point.  **What a bare photograph cannot carry
is a different exposure**, which is why the level has to be right rather than the picture having to look like the footage.

**The two photographs test different ends of the range**, which is why both are used:

  * the **case** is dominated by blue, cyan and magenta emitters, so it tests whether the display's solved weights hold up
    on a picture whose own chroma fights them;
  * the **basement** is an almost-black room with **one incandescent lamp**, so it tests the dark floor, the highlight and
    the lamp's own diffusion at the same time -- and it has no coloured emitters at all, so the green cast is the chain's
    alone.

**Every figure is reported against the footage's measured range** rather than against one frame, because the footage's own
spread was measured at mean 0.11-0.39 and R/G 0.37-0.85 across the stretch.
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
CASE = os.path.join(OBJ, "e0", "e0e36b261fbbad42777702d3b775fcf30d4f8505fdc104e08b24405e69eda14c")
BASEMENT = os.path.join(OBJ, "0c", "0cc4906c4997112e2e1833ec1f0514bfb7f459fd7362c2fb04781761fdd42cef")
#: the footage's measured range, from one frame per second across the whole NightShot stretch
RANGE = {"mean": (0.11, 0.39), "std": (0.09, 0.26), "RG": (0.37, 0.85), "p1": (0.024, 0.055)}
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
                p5=float(np.quantile(g, 0.05)), p95=float(np.quantile(g, 0.95)),
                p99=float(np.quantile(g, 0.99)), clip=float((g >= 0.98).mean()),
                RG=r / max(1e-6, gg), BG=b / max(1e-6, gg),
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


COLS = ["mean", "std", "p1", "p5", "p95", "p99", "clip", "RG", "BG", "hue", "dark"]


def show(tag, a):
    s = fig(a)
    print("   %-34s %s" % (tag, "  ".join("%.4f" % s[c] for c in COLS)))
    return s


print("   the footage's measured RANGE across the stretch: %s"
      % ", ".join("%s %.2f-%.2f" % (k, v[0], v[1]) for k, v in RANGE.items()))
print()
print("   %-34s %s" % ("", "  ".join("%-8s" % c for c in COLS)))
case, base_img = load(CASE), load(BASEMENT)
show("the case photograph (the input)", case)
show("the basement photograph (the input)", base_img)
print()
case_out = run(case)
base_out = run(base_img)
sc, sb = show("the case, mode 2", case_out), show("the basement, mode 2", base_out)
print()
print("   === against the footage's range ===")
for tag, s in (("the case", sc), ("the basement", sb)):
    inside = [k for k, (lo, hi) in RANGE.items() if lo <= s[k] <= hi]
    outside = [k for k, (lo, hi) in RANGE.items() if not (lo <= s[k] <= hi)]
    print("   %-14s inside: %-26s outside: %s" % (tag, ", ".join(inside), ", ".join(outside)))
    for k in outside:
        lo, hi = RANGE[k]
        print("        %-6s is %.4f, the footage spans %.3f-%.3f" % (k, s[k], lo, hi))

gap, cols = 7, 2
cw, ch = W // cols, (H - int(H * OSD)) // cols
panels = [("the case photograph (input)", case, fig(case)),
          ("the case, mode 2", case_out, sc),
          ("the basement photograph (input)", base_img, fig(base_img)),
          ("the basement, mode 2", base_out, sb)]
canvas = Image.new("RGB", ((cw + gap) * cols + gap, (ch + 52) * 2 + 28), (16, 16, 18))
d = ImageDraw.Draw(canvas)
d.text((6, 5), "BARE PHOTOGRAPHS ONLY -- no camcorder frame anywhere in the input path", fill=(242, 242, 242))
for i, (tag, arr, s) in enumerate(panels):
    x = gap + (i % cols) * (cw + gap)
    y = 26 + (i // cols) * (ch + 52)
    canvas.paste(Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).resize((cw, ch), Image.LANCZOS), (x, y))
    d.text((x, y + ch + 3), tag, fill=(238, 238, 238))
    d.text((x, y + ch + 16), "mean %.3f  std %.3f  p1 %.3f  p5 %.3f" % (s["mean"], s["std"], s["p1"], s["p5"]),
           fill=(214, 214, 214))
    d.text((x, y + ch + 29), "R/G %.3f  B/G %.3f  hue %+.1f  below .02 %.4f"
           % (s["RG"], s["BG"], s["hue"], s["dark"]), fill=(214, 214, 214))
out = os.path.join(OUT, "BARE_INPUTS.png")
canvas.save(out)
print()
print("   written: %s" % os.path.relpath(out, ROOT))
