"""
Test the hypothesis the stage image produced: the bands are the 8x8 labelling grid's own edges.

**What the trace showed.**  Traced stage by stage and written as images, **mode 1's output is already coarse-banded at the
`camera` stage** (row-difference 0.17834 against mode 0's 0.09386), and every later stage leaves it alone.  **And the
`camera` is where the night-vision terms run**, so the fault is one of them.

**The hypothesis, from the bands' SIZE.**  `ir_highlights` multiplies the picture by `spec.ir_albedo`, which
`albedo_map` builds from the vision model's **8x8 grid** by **nearest-neighbour repeat** -- and 533x400 divided by 8 is
**66x50 pixels per cell**.  **The bands in the picture are about that wide.**  So the "banding" is not a noise or a filter
artefact at all: **it is the label grid's own cell edges, magnified to the frame.**

**Why the earlier ablations could not find it.**  Each night-vision piece was tested *alone on the source*, where
`ir_highlights` multiplies by a map whose cells are uniform **and where the picture has no chain processing** -- and the
row-difference moved by 0.002.  **In the chain the same term acts on an already-processed picture and the cells' edges
land on real structure**, which is where a 66-pixel step becomes visible.  **And `nv_ir_albedo = 0` or
`nv_nir_material = 0` would have shown it, but neither was among the eight switches swept.**

**So the test is direct**: the same render at grid sizes from 2 to 24, which changes the cell size and therefore the band
spacing, and with the material labelling off entirely.  **If the banding is the grid, its spacing must follow the grid
size** -- a prediction that can fail, unlike the six statistics before it.
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
NS0 = os.path.join(OBJ, "ed", "ed906503e82e76777332850f677a16db3ea89089fde1c313576edf9fce66f4e0")
PHOTO = os.path.join(OBJ, "69", "699c23315df037b01fa831a273364b41f99c28733fb1dc928b22d1e6c281818d")
OUT = os.path.join(ROOT, "tools", "calib")
W, H, OSD = 533, 400, 0.22
sl = slice(int(H * OSD), H)


def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0
    return np.asarray(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS),
                      np.float32) / 255.0


def rowsig(a):
    g = np.clip(np.asarray(a, np.float32)[sl] @ COEF, 0, 1)
    return float(np.abs(np.diff(g, axis=0)).mean()) / max(1e-9, float(g.std()))


src = load(PHOTO)
real = load(NS0)
print("   the source's row-difference %.5f; the footage's %.5f" % (rowsig(src), rowsig(real)))


class One:
    height, width = H, W
    fps = 25.0
    frame_count = 1

    def frame(self, i):
        return src


BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])
OFF = {"halation": 0.0, "ccd_smear": 0.0, "flare_defocus": 0.0, "lens_flare": 0.0,
       "overflow_drain": 0.0, "chroma_aberration": 0.0, "sensor_noise": 0.0, "noise_grain": 0.0,
       "fixed_pattern": 0.0, "spike_spread": 0.0, "motion_blur": 0.0, "stuck_fraction": 0.0,
       "bloom_threshold": 0.0, "overflow_width": 0.0, "sharpness": 1.0}


def run(mode=1.0, **cam):
    layers = {"camera": dict(OFF, nv_mode=mode, **cam),
              "format": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0},
              "player": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0},
              "codec": {"dv_enabled": 0.0}, "display": {"enabled": 0.0}}
    rr = Renderer(One(), W, H, 25.0, merge_values(BASE, layers), seed=20240101)
    return np.clip(np.asarray(rr.render_frame(0), np.float32), 0, 1)


print()
print("   === the prediction: if the bands are the label grid, their SPACING follows the grid size ===")
print("   %-40s %-14s %-14s %s" % ("", "row-diff", "cell size (px)", "vs the source"))
CASES = [("grid 8 (the default)", dict(nv_nir_grid=8)),
         ("grid 2 (cells 266x200)", dict(nv_nir_grid=2)),
         ("grid 4 (cells 133x100)", dict(nv_nir_grid=4)),
         ("grid 16 (cells 33x25)", dict(nv_nir_grid=16)),
         ("grid 24 (cells 22x17)", dict(nv_nir_grid=24)),
         ("**material labelling OFF**", dict(nv_nir_material=0.0)),
         ("labelling on, albedo mix 0", dict(nv_nir_material=1.0, nv_ir_albedo=0.0)),
         ("mode 0 for reference", None)]
shots = []
base = rowsig(src)
for tag, cam in CASES:
    if cam is None:
        a = run(0.0)
        cell = "-"
    else:
        a = run(1.0, **cam)
        g = int(cam.get("nv_nir_grid", 8))
        cell = "%dx%d" % (W // g, H // g)
    shots.append((tag, a))
    print("   %-40s %-14.5f %-14s %+.5f" % (tag, rowsig(a), cell, rowsig(a) - base))

gap = 6
big = lambda x: Image.fromarray((np.clip(np.asarray(x, np.float32), 0, 1) * 255).astype(np.uint8)
                                ).resize((W // 3, H // 3), Image.LANCZOS)
cols = 4
rows = (len(shots) + cols - 1) // cols
cw, ch = W // 3, H // 3
canvas = Image.new("RGB", (cw * cols + gap * (cols + 1), (ch + 34) * rows + 26), (16, 16, 18))
d = ImageDraw.Draw(canvas)
d.text((6, 5), "IS THE BANDING THE LABEL GRID?  cell size from 266 px down to 22 px, and the labelling off",
       fill=(242, 242, 242))
for i, (tag, a) in enumerate(shots):
    cx = gap + (i % cols) * (cw + gap)
    cy = 26 + (i // cols) * (ch + 34)
    canvas.paste(big(a), (cx, cy))
    d.text((cx, cy + ch + 3), tag[:46], fill=(234, 234, 234))
    d.text((cx, cy + ch + 16), "row-diff %.4f" % rowsig(a), fill=(214, 214, 214))
out = os.path.join(OUT, "GRID_TEST.png")
canvas.save(out)
print()
print("   written: %s" % os.path.relpath(out, ROOT))
