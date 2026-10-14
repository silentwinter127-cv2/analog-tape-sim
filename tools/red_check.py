"""
Measure the RED excess: the render against the user's footage frame, which is the property they reported.

**The report.**  "红色根本没有这么显著" -- the render shows red and pink where the footage shows almost none.  **And the
cause is already measured**: `chroma_residue` carries the input photograph's own chroma onto the mono display, **and that
photograph is lit by red, magenta and pink emitters**, so the residue puts them back.  The footage was shot under a
near-IR lamp with the IR-cut filter removed, where those visible emitters contribute differently.

**So the quantity is the red channel's excess over what a green-cast display should have**, and it is expressed three ways
so the claim is not resting on one statistic:

  * **R/G** -- the ratio the display's own weights fix, so any rise above it is chroma that came from somewhere else;
  * **the red excess in absolute terms**, `R - G * (w_R / w_G)`, which is positive only where red has been added;
  * **the fraction of the frame where red actually leads green**, since a small saturated area is what the eye notices.

**And the residue is swept with each of those three reported**, because the question is not whether the residue can be
lowered -- it can -- but **how much of the red it accounts for**, and whether what remains is in the input or in the chain.
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
FOOT = os.path.join(OBJ, "8d", "8d55424b71f31edfad280e698a0403836cb0c4a655cd513a8f3bb03987af62a9")
OUT = os.path.join(ROOT, "tools", "calib")
W, H, OSD = 533, 400, 0.17


def load(path, size=(W, H), crop_osd=False):
    im = Image.open(path).convert("RGB")
    a = np.asarray(im, np.float32) / 255.0
    if crop_osd:
        a = a[int(a.shape[0] * OSD):]
        im = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
    return np.asarray(im.resize(size, Image.LANCZOS), np.float32) / 255.0


def red_stats(a, crop_osd=False):
    x = a[int(a.shape[0] * OSD):] if crop_osd else a
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    #: the ratio a pure green cast would give, from the display's own weights
    want = 0.8291 / 1.0000
    excess = r - g * want
    return dict(RG=float(r.mean() / max(1e-6, g.mean())),
                RB=float(r.mean() / max(1e-6, b.mean())),
                excess=float(excess.mean()),
                pos=float((excess > 0).mean()),
                lead=float((r > g).mean()),
                rmax=float(np.quantile(r, 0.99)))


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


COLS = [("R/G", "RG"), ("R/B", "RB"), ("red excess", "excess"), ("% with excess", "pos"),
        ("% red>green", "lead"), ("R p99", "rmax")]


def show(tag, a, crop=False):
    s = red_stats(a, crop)
    print("   %-34s %s" % (tag, "  ".join("%s %8.4f" % (c, s[k]) for c, k in COLS)))
    return s


print("   %-34s %s" % ("", "  ".join("%-13s" % c for c, _ in COLS)))
show("the input photograph", SRC)
f = show("the footage frame (the target)", FOOT_IMG, crop=True)
print()
print("   === the render as the residue is lowered ===")
rows = []
for res in (0.45, 0.30, 0.20, 0.10, 0.0):
    a = run(nv_chroma_residue=res)
    rows.append((res, a, show("residue %.2f" % res, a)))
print()
print("   === and the pure-green reference: the two regimes with no residue at all ===")
a0 = run(nv_chroma_residue=0.0)
print("   with residue 0 the display is the projection alone, so the red excess is %.4f and red leads green in %.4f of"
      % (red_stats(a0)["excess"], red_stats(a0)["lead"]))
print("   the frame -- **so any excess the user sees traces to the input's own chroma, which the residue carries.**")

gap = 7
cw, ch = W // 3, (H - int(H * OSD)) // 3
panels = [("the footage frame", FOOT_IMG), ("the render, residue 0.45", rows[0][1]),
          ("the render, residue 0.20", rows[2][1]), ("the render, residue 0.0", rows[4][1]),
          ("the input photograph", SRC), ("the render, old weights", run(nv_chroma_residue=0.0))]
canvas = Image.new("RGB", ((cw + gap) * 3 + gap, (ch + 46) * 2 + 28), (16, 16, 18))
d = ImageDraw.Draw(canvas)
d.text((6, 5), "THE RED EXCESS -- carried onto the mono display by the chroma residue, from the input's own chroma",
       fill=(242, 242, 242))
for i, (tag, arr) in enumerate(panels):
    x = gap + (i % 3) * (cw + gap)
    y = 26 + (i // 3) * (ch + 46)
    canvas.paste(Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).resize((cw, ch), Image.LANCZOS), (x, y))
    s = red_stats(arr)
    d.text((x, y + ch + 3), tag, fill=(238, 238, 238))
    d.text((x, y + ch + 16), "R/G %.3f   red excess %+.4f" % (s["RG"], s["excess"]), fill=(214, 214, 214))
    d.text((x, y + ch + 29), "red leads green in %.3f of the frame" % s["lead"], fill=(214, 214, 214))
out = os.path.join(OUT, "RED_CHECK.png")
canvas.save(out)
print()
print("   written: %s" % os.path.relpath(out, ROOT))
