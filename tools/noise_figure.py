"""
The noise test figure: what each term looks like in the two ends of the tonal range.

**`note.txt`'s requirement is visual as much as numeric** -- "不要简单地对整张图叠加均匀颗粒；暗部和高光的表现应该不同"
-- so a table of spreads cannot show whether it is met.  **This is the figure for it**: a frame rendered with each noise
configuration, cropped to a dark region and a bright one, so the grain can be compared where the note says it must
differ.

**The four panels hold everything else fixed** and change only which noise term is on:

    neither        the control, so the picture's own structure is visible
    read only      the fixed term -- equally grainy everywhere, which is what the note calls wrong
    shot only      the signal-dependent term -- silence in the black, grain on the lit surface
    both           what a sensor actually does

**and each is shown twice**, once full-frame and once as matched crops of the darkest and brightest parts, because a
difference of a few thousandths in a standard deviation is invisible at full size while it is obvious in a crop.

**Every panel carries its own measured spread in the two regions**, so the picture and the number travel together -- a
figure with no measurement beside it is how a plausible-looking wrong mechanism survives.
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
PHOTO = os.path.join(OBJ, "69", "699c23315df037b01fa831a273364b41f99c28733fb1dc928b22d1e6c281818d")
OUT = os.path.join(ROOT, "tools", "calib")
W, H, OSD = 420, 315, 0.22
sl = slice(int(H * OSD), H)

_a = np.asarray(Image.open(PHOTO).convert("RGB"), np.float32) / 255.0
SRC = np.asarray(Image.fromarray((np.clip(_a, 0, 1) * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS),
                 np.float32) / 255.0


class Still:
    height, width = H, W
    fps = 25.0
    frame_count = 400

    def frame(self, i: int) -> np.ndarray:
        return SRC


BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])
OFF = {"halation": 0.0, "ccd_smear": 0.0, "flare_defocus": 0.0, "lens_flare": 0.0,
       "overflow_drain": 0.0, "chroma_aberration": 0.0, "sensor_noise": 0.0, "noise_grain": 0.0,
       "fixed_pattern": 0.0, "spike_spread": 0.0, "motion_blur": 0.0, "stuck_fraction": 0.0,
       "bloom_threshold": 0.0, "overflow_width": 0.0, "sharpness": 1.0}


def render(**cam):
    layers = {"camera": dict(OFF, nv_mode=1.0, **cam),
              "format": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0},
              "player": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0},
              "codec": {"dv_enabled": 0.0}, "display": {"enabled": 0.0}}
    outs = []
    for idx in range(50, 58):
        rr = Renderer(Still(), W, H, 25.0, merge_values(BASE, layers), seed=3)
        for _, rgb in rr.render_frames([idx]):
            outs.append(np.clip(np.asarray(rgb, np.float32), 0, 1))
    st = np.stack(outs)
    return st


def regions(st):
    """The darkest and brightest fifths of the frame, as the cells to compare, plus their temporal spreads."""
    g = np.clip(st @ COEF, 0, 1)
    mean = g.mean(axis=0)[sl]
    noise = g.std(axis=0)[sl]
    lo, hi = np.quantile(mean, 0.2), np.quantile(mean, 0.8)
    dmask, bmask = mean <= lo, mean >= hi
    return float(noise[dmask].mean()), float(noise[bmask].mean())


CASES = [("neither (the control)", dict(nv_read_noise=0.0, nv_shot_noise=0.0)),
         ("read noise only", dict(nv_read_noise=0.20, nv_shot_noise=0.0)),
         ("shot noise only", dict(nv_read_noise=0.0, nv_shot_noise=0.25)),
         ("both (a real sensor)", dict(nv_read_noise=0.20, nv_shot_noise=0.25))]

results = []
for tag, cam in CASES:
    st = render(**cam)
    d, b = regions(st)
    results.append((tag, st[-1], d, b, st))
    print("   %-24s dark %.5f   bright %.5f   bright/dark %.3f" % (tag, d, b, b / max(1e-9, d)))

#: the first panel's own luminance decides which cells are "dark" and "bright", so every panel is cropped at the SAME
#: places and the comparison is of grain rather than of framing
g0 = np.clip(results[0][4] @ COEF, 0, 1).mean(axis=0)[sl]
lo, hi = np.quantile(g0, 0.2), np.quantile(g0, 0.8)
ys, xs = np.where(g0 <= lo)
dy, dx = int(np.median(ys)), int(np.median(xs))
ys, xs = np.where(g0 >= hi)
by, bx = int(np.median(ys)), int(np.median(xs))
C = 46
ysl = slice(int(H * OSD) + max(0, dy - C // 2), int(H * OSD) + max(0, dy - C // 2) + C)
xsl = slice(max(0, dx - C // 2), max(0, dx - C // 2) + C)
ysb = slice(int(H * OSD) + max(0, by - C // 2), int(H * OSD) + max(0, by - C // 2) + C)
xsb = slice(max(0, bx - C // 2), max(0, bx - C // 2) + C)

gap = 6
cell = (C, C)
panel_w = W
canvas = Image.new("RGB", (panel_w * 3 + cell[0] * 2 + gap * 5,
                           H * 2 + cell[1] * 2 + 120), (16, 16, 18))
dr = ImageDraw.Draw(canvas)
dr.text((6, 5), "THE NOISE TERMS AT THE TWO ENDS OF THE TONAL RANGE", fill=(242, 242, 242))
dr.text((6, 20), "full frames across the top; matched crops of the DARK region and the BRIGHT region below each",
        fill=(190, 190, 190))

for i, (tag, frame, d, b, st) in enumerate(results):
    x = 6 + i * (panel_w // 2 + gap) if False else 6 + i * (W // 2 + gap)
    small = Image.fromarray((np.clip(frame, 0, 1) * 255).astype(np.uint8)).resize((W // 2, H // 2), Image.LANCZOS)
    canvas.paste(small, (x, 40))
    dr.text((x, 40 + H // 2 + 3), "%s" % tag, fill=(235, 235, 235))
    dr.text((x, 40 + H // 2 + 16), "dark %.4f  bright %.4f  ratio %.2f" % (d, b, b / max(1e-9, d)),
            fill=(225, 225, 225))
    # the two crops, blown up so the grain is visible
    dark = Image.fromarray((np.clip(frame[ysl, xsl], 0, 1) * 255).astype(np.uint8)).resize((C * 3, C * 3), Image.NEAREST)
    bright = Image.fromarray((np.clip(frame[ysb, xsb], 0, 1) * 255).astype(np.uint8)).resize((C * 3, C * 3), Image.NEAREST)
    y0 = 40 + H // 2 + 32
    canvas.paste(dark, (x, y0))
    canvas.paste(bright, (x + C * 3 + gap, y0))
    dr.text((x, y0 + C * 3 + 2), "dark x3", fill=(210, 210, 210))
    dr.text((x + C * 3 + gap, y0 + C * 3 + 2), "bright x3", fill=(210, 210, 210))

out = os.path.join(OUT, "NOISE_TEST.png")
canvas.save(out)
print()
print("   written: %s" % os.path.relpath(out, ROOT))
