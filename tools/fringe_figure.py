"""
The test figure: the rainbow fringe measured as a function of brightness, on the input that can produce it.

**Why this figure and not another.**  The user named four things, and this one took three wrong attempts to measure:

  * the first attempt measured the user's *exported* render, which is a picture of the model before any change;
  * the second left `chroma_aberration` at zero along with the rest of the optics, **so the mechanism that produces a
    fringe was switched off in the rig that was testing for a fringe**;
  * and the third used the basement photograph, which is a dark room with one lamp and **no saturated emitters at all** --
    so even with the optics on there was nothing to fringe.  Measured: its brightest band holds 1,622 pixels against the
    footage's 5,463 and 35,263, so the statistic was being computed on a handful of pixels.

**The case photograph is the right input** because it is the one with saturated RGB emitters, which is what the user was
describing when they said the fan lights have rainbow edges.

**The statistic is the hue spread WITHIN brightness bands**, because a fringe is hue that varies with position near a
source: a whole-frame hue spread averages it away, and so does any per-pixel rule with no brightness dependence -- which is
exactly what the previous implementation was.

**The four bands are printed for every panel**, and the pixel count with them, so a band computed on almost no pixels
cannot be mistaken for a measurement.
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
P2 = os.path.join(OBJ, "ee", "ee23c56ac527467f74f6f9140400a172db2577aea1e13c0baab70e78305c9c7e")
P3 = os.path.join(OBJ, "f6", "f6f945abfa1a8a09e497c35b897e15b6b9e5417b82c3d50ae827957abce5015d")
OUT = os.path.join(ROOT, "tools", "calib")
W, H, OSD = 533, 400, 0.17
sl = slice(int(H * OSD), H)
BANDS = ("dark", "mid", "bright", "brightest")


def load(p, crop_osd=False):
    im = Image.open(p).convert("RGB")
    a = np.asarray(im, np.float32) / 255.0
    if crop_osd:
        a = a[int(a.shape[0] * OSD):]
        im = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
    return np.asarray(im.resize((W, H), Image.LANCZOS), np.float32) / 255.0


def analyse(a, crop_osd=False):
    x = np.clip(np.asarray(a, np.float32), 0, 1)
    x = x[int(H * OSD):] if crop_osd else x[sl]
    lum = np.clip(x @ COEF, 0, 1)
    u, v = x[..., 0] - x[..., 1], x[..., 2] - x[..., 1]
    m = np.sqrt(u * u + v * v)
    ang = np.arctan2(v, u)
    spreads, counts, sats = [], [], []
    for lo, hi in ((0.0, 0.25), (0.25, 0.6), (0.6, 0.85), (0.85, 1.01)):
        k = (lum >= lo) & (lum < hi) & (m > 0.02)
        if k.sum() < 50:
            spreads.append(0.0)
            counts.append(int(k.sum()))
            sats.append(0.0)
            continue
        aa = ang[k]
        spreads.append(float(np.angle(np.exp(1j * (aa - np.angle(np.exp(1j * aa).mean())))).std()))
        counts.append(int(k.sum()))
        sats.append(float(m[k].mean()))
    r, g, b = (float(x[..., i].mean()) for i in range(3))
    return dict(primary="#%02X%02X%02X" % (int(round(r * 255)), int(round(g * 255)), int(round(b * 255))),
                mean=float(lum.mean()), std=float(lum.std()), p50=float(np.quantile(lum, 0.50)),
                below05=float((lum < 0.05).mean()), spreads=spreads, counts=counts, sats=sats)


BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])


class One:
    height, width = H, W
    fps = 25.0
    frame_count = 1

    def __init__(self, a):
        self.a = a

    def frame(self, i):
        return self.a


def run(src, **cam):
    layers = {"camera": {"nv_mode": 2.0, **cam},
              "format": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0},
              "player": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0},
              "codec": {"dv_enabled": 0.0}, "display": {"enabled": 0.0}}
    rr = Renderer(One(src), W, H, 25.0, merge_values(BASE, layers), seed=20240101)
    return np.clip(np.asarray(rr.render_frame(0), np.float32), 0, 1)


case_src, base_src = load(CASE), load(BASEMENT)
case_out, base_out = run(case_src), run(base_src)

print("   %-30s %-9s %-7s %-7s %-8s %s" % ("", "primary", "mean", "p50", "<0.05", "hue spread: " + "  ".join(BANDS)))
for tag, arr, crop in (("the real NIGHTSHOT 2", load(P2, True), True), ("the real NIGHTSHOT 1", load(P3, True), True),
                       ("the case (input)", case_src, False), ("the case, mode 2", case_out, False),
                       ("the basement (input)", base_src, False), ("the basement, mode 2", base_out, False)):
    s = analyse(arr, crop)
    print("   %-30s %-9s %-7.4f %-7.4f %-8.4f %s" % (tag, s["primary"], s["mean"], s["p50"], s["below05"],
                                                     "  ".join("%.3f" % v for v in s["spreads"])))
print("   %-30s %-9s %-7s %-7s %-8s %s" % ("", "", "", "", "pixels in each band", ""))
for tag, arr, crop in (("the real NIGHTSHOT 2", load(P2, True), True), ("the case, mode 2", case_out, False),
                       ("the basement, mode 2", base_out, False)):
    s = analyse(arr, crop)
    print("   %-30s %s" % (tag, "  ".join("%9d" % c for c in s["counts"])))

gap, cols = 7, 2
cw, ch = W // cols, (H - int(H * OSD)) // cols
panels = [("the real NIGHTSHOT 2 (target)", load(P2, True), True, True),
          ("the case photograph (input)", case_src, False, False),
          ("the real NIGHTSHOT 1 (target)", load(P3, True), True, True),
          ("the case, mode 2", case_out, False, False),
          ("the basement photograph (input)", base_src, False, False),
          ("the basement, mode 2", base_out, False, False)]
canvas = Image.new("RGB", ((cw + gap) * cols + gap, (ch + 60) * 3 + 28), (16, 16, 18))
d = ImageDraw.Draw(canvas)
d.text((6, 5), "THE RAINBOW FRINGE -- hue spread within brightness bands, on the input that has emitters to fringe",
       fill=(242, 242, 242))
for i, (tag, arr, crop, is_real) in enumerate(panels):
    x = gap + (i % cols) * (cw + gap)
    y = 26 + (i // cols) * (ch + 60)
    canvas.paste(Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).resize((cw, ch), Image.LANCZOS), (x, y))
    s = analyse(arr, crop)
    d.text((x, y + ch + 3), tag, fill=(238, 238, 238) if not is_real else (255, 236, 180))
    d.text((x, y + ch + 17), "primary %s   mean %.3f   p50 %.3f   <0.05 %.3f"
           % (s["primary"], s["mean"], s["p50"], s["below05"]), fill=(214, 214, 214))
    d.text((x, y + ch + 30), "hue spread  %s" % "  ".join("%.3f" % v for v in s["spreads"]), fill=(214, 214, 214))
    d.text((x, y + ch + 43), "pixels      %s" % "  ".join("%7d" % c for c in s["counts"]), fill=(150, 150, 150))
out = os.path.join(OUT, "FRINGE.png")
canvas.save(out)
print()
print("   written: %s" % os.path.relpath(out, ROOT))
