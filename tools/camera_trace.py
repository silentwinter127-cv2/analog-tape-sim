"""
Trace INSIDE the camera stage: one image per call, between the night-vision terms and the sharpen.

**Why here.**  Writing every stage's output as an image located the banding to the `camera` stage in one step, after seven
statistics had failed -- mode 1 reads 0.17834 against mode 0's 0.09386 at that stage and the later stages leave it alone.
**And reading `apply` narrows it further**, because the call order is:

    1043  rgb = self._night_vision(rgb, "scene", buf.index)    <- the only thing the modes differ by
    1044  rgb = self._roll_off(rgb)
    1051  rgb = self._shadow_depth(rgb)
    ...   halation, overflow, flare, the lens terms, sharpen -- all run in BOTH modes

**so the banding appears between those lines and the terms before them are shared.**  The candidate is a term that runs in
both modes and behaves differently at a **different input level**: the night-vision terms raise the luma from 0.1973 to
0.2303, and a non-linear curve applied at a different level redistributes structure differently.  **That is consistent
with every measurement made**, because each night-vision piece applied *alone* to the source moved the row-difference by
at most 0.002 -- applied alone it never reaches these curves.

**Each call's output is wrapped and written**, so the picture at which the bands first appear names the line.  **Both
modes are traced**, since the pair is what isolates the term: whatever differs between them at a given line is the cause,
and nothing has to be guessed.
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
from atsim.stages.camera import CameraStage                                   # noqa: E402

COEF = np.array([0.299, 0.587, 0.114], np.float32)
OBJ = (r"C:\Users\Administrator\.dsh\attachments\v1\objects")
PHOTO = os.path.join(OBJ, "69", "699c23315df037b01fa831a273364b41f99c28733fb1dc928b22d1e6c281818d")
OUT = os.path.join(ROOT, "tools", "calib")
W, H, OSD = 533, 400, 0.22

_a = np.asarray(Image.open(PHOTO).convert("RGB"), np.float32) / 255.0
SRC = np.asarray(Image.fromarray((np.clip(_a, 0, 1) * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS),
                 np.float32) / 255.0


class One:
    height, width = W, H
    fps = 25.0
    frame_count = 1

    def frame(self, i):
        return SRC


def rowsig(a):
    g = np.clip(np.asarray(a, np.float32)[int(H * OSD):] @ COEF, 0, 1)
    return float(np.abs(np.diff(g, axis=0)).mean()) / max(1e-9, float(g.std()))


#: **the calls in the order `apply` makes them**, with the ones that take an extra argument marked, so the trace follows
#: the real sequence rather than a list assembled from memory
TRACED = [
    ("_night_vision", True),
    ("_roll_off", False),
    ("_shadow_depth", False),
    ("_halation", False),
    ("_overflow_drain", False),
    ("_lens_flare", False),
    ("_flare", False),
    ("_sharpen", False),
]

traces = {}
for mode, name in ((0.0, "mode0"), (1.0, "mode1")):
    shots = []
    originals = {}
    for meth, takes_stage in TRACED:
        fn = getattr(CameraStage, meth, None)
        if fn is None:
            continue
        originals[meth] = fn

        def make(m, f, takes, _shots=shots):
            def wrapped(self, rgb, *a, **kw):
                out = f(self, rgb, *a, **kw) if takes else f(self, rgb, *a, **kw)
                if isinstance(out, np.ndarray) and out.ndim == 3:
                    _shots.append((m, np.clip(np.asarray(out, np.float32), 0, 1).copy()))
                return out
            return wrapped
        setattr(CameraStage, meth, make(meth, fn, takes_stage))
    layers = {"camera": {"nv_mode": mode, "halation": 0.0, "ccd_smear": 0.0, "flare_defocus": 0.0,
                         "lens_flare": 0.0, "overflow_drain": 0.0, "chroma_aberration": 0.0,
                         "sensor_noise": 0.0, "noise_grain": 0.0, "fixed_pattern": 0.0,
                         "spike_spread": 0.0, "motion_blur": 0.0, "stuck_fraction": 0.0,
                         "bloom_threshold": 0.0, "overflow_width": 0.0, "sharpness": 1.0},
              "format": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0},
              "player": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0},
              "codec": {"dv_enabled": 0.0}, "display": {"enabled": 0.0}}
    rr = Renderer(One(), W, H, 25.0, merge_values(SCHEMA.resolve_all(
        FACTORY_PRESETS["broadcast_1992_weak"]["layers"]), layers), seed=20240101)
    rr.render_frame(0)
    for meth, fn in originals.items():
        setattr(CameraStage, meth, fn)
    traces[name] = shots

names = [m for m, _ in traces["mode1"]]
print("   inside the camera, in call order: %s" % ", ".join(names))
print()
print("   %-18s %-12s %-12s %s" % ("call", "mode0", "mode1", "difference"))
print("   %-18s %-12.5f %-12s" % ("(the input)", rowsig(SRC), "-"))
for k, m in enumerate(names):
    a0 = traces["mode0"][k][1]
    a1 = traces["mode1"][k][1]
    b0, b1 = rowsig(a0), rowsig(a1)
    print("   %-18s %-12.5f %-12.5f %+.5f" % (m, b0, b1, b1 - b0))

gap = 5
cw = W // 4
ch = (H - int(H * OSD)) // 4
canvas = Image.new("RGB", ((cw + gap) * (len(names) + 1) + gap, ch * 2 + 80), (16, 16, 18))
d = ImageDraw.Draw(canvas)
d.text((6, 5), "INSIDE THE CAMERA -- every call's output, mode 0 (top row) and mode 1 (bottom row)",
       fill=(242, 242, 242))


def crop(a):
    return np.clip(np.asarray(a, np.float32), 0, 1)[int(H * OSD):]


canvas.paste(Image.fromarray((crop(SRC) * 255).astype(np.uint8)).resize((cw, ch), Image.LANCZOS), (gap, 24))
d.text((gap, 24 + ch + 3), "the INPUT", fill=(236, 236, 236))
d.text((gap, 24 + ch + 16), "%.4f" % rowsig(SRC), fill=(214, 214, 214))
for k, m in enumerate(names):
    x = gap + (k + 1) * (cw + gap)
    for row, tag in ((0, "mode0"), (1, "mode1")):
        a = crop(traces[tag][k][1])
        y = 24 + row * (ch + 36)
        canvas.paste(Image.fromarray((a * 255).astype(np.uint8)).resize((cw, ch), Image.LANCZOS), (x, y))
        if row == 1:
            d.text((x, y + ch + 3), m, fill=(236, 236, 236))
            d.text((x, y + ch + 16), "%.4f" % rowsig(traces[tag][k][1]), fill=(214, 214, 214))
out = os.path.join(OUT, "CAMERA_TRACE.png")
canvas.save(out)
print()
print("   written: %s" % os.path.relpath(out, ROOT))
