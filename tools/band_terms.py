"""
Test each remaining mode-1-only term in the chain, since the night-vision pieces applied alone band nothing.

**What the figure established.**  Four panels, one switch at a time:

    the source photograph                  clean -- the case, the monitor and the cables are legible
    mode 0 (off)                           slightly soft, subject legible
    **mode 1**                             **coarse diagonal bands, the subject broken up**
    mode 1, green display off              **identically broken**
    mode 1, residue 0                      **identically broken**

**so the banding is not the green display and not the chroma residue** -- and the pieces applied *alone* moved the
row-difference by at most -0.002, so it is not any of them in isolation.  **It is therefore one of the terms a mode
switches on *in the chain* but which was not measured alone**, and the candidates are few:

  * **`shot_noise`, which defaults to 0.10 and was added two rounds ago.**  It multiplies the noise by `sqrt(lum)`
    **per pixel**, and `lum` varies strongly from row to row in a picture like this -- **so a term whose amplitude
    follows the picture can print the picture's row structure into the noise**, which is what coarse bands are;
  * `read_noise` (0.0 by default, so unlikely);
  * the exposure bias and gain the mode applies, which change the levels the noise is added at;
  * `accumulate` (1.0 in mode 1).

**`shot_noise` is the first suspect and the newest code**, which is worth stating plainly: **the banding was not reported
before it was added**, and a per-pixel signal-dependent term is exactly the kind of thing that can produce this.  **So it
is switched off first**, and every other candidate with it, one at a time.
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
print("   the source photograph: row-difference %.5f" % rowsig(src))


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


def run(mode, **cam):
    layers = {"camera": dict(OFF, nv_mode=mode, **cam),
              "format": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0},
              "player": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0},
              "codec": {"dv_enabled": 0.0}, "display": {"enabled": 0.0}}
    rr = Renderer(One(), W, H, 25.0, merge_values(BASE, layers), seed=20240101)
    return np.clip(np.asarray(rr.render_frame(0), np.float32), 0, 1)


CASES = [
    ("mode 1 as it renders", 1.0, {}),
    ("mode 1, shot_noise = 0", 1.0, dict(nv_shot_noise=0.0)),
    ("mode 1, read_noise = 0", 1.0, dict(nv_read_noise=0.0)),
    ("mode 1, shot and read = 0", 1.0, dict(nv_shot_noise=0.0, nv_read_noise=0.0)),
    ("mode 1, chroma_noise = 0", 1.0, dict(nv_chroma_noise=0.0)),
    ("mode 1, exposure bias = 0", 1.0, dict(nv_exposure_steps=0.0)),
    ("mode 1, sharpness = 1 (default)", 1.0, dict(sharpness=1.0)),
    ("mode 1, all noise off", 1.0, dict(nv_shot_noise=0.0, nv_read_noise=0.0, nv_chroma_noise=0.0)),
    ("mode 0 for reference", 0.0, {}),
]
print()
print("   %-38s %-14s %s" % ("", "row-difference", "vs the source"))
base = rowsig(src)
shots = []
for tag, mode, cam in CASES:
    a = run(mode, **cam)
    r = rowsig(a)
    shots.append((tag, a, r))
    print("   %-38s %-14.5f %+.5f" % (tag, r, r - base))

gap = 6
big = lambda x: Image.fromarray((np.clip(np.asarray(x, np.float32), 0, 1) * 255).astype(np.uint8)
                                ).resize((W // 3, H // 3), Image.LANCZOS)
n = len(shots)
cols = 5
rows = (n + cols - 1) // cols
cw, ch = W // 3, H // 3
canvas = Image.new("RGB", (cw * cols + gap * (cols + 1), (ch + 34) * rows + 24), (16, 16, 18))
d = ImageDraw.Draw(canvas)
d.text((6, 5), "EACH MODE-1 TERM SWITCHED OFF IN TURN", fill=(242, 242, 242))
for i, (tag, a, r) in enumerate(shots):
    cx = gap + (i % cols) * (cw + gap)
    cy = 24 + (i // cols) * (ch + 34)
    canvas.paste(big(a), (cx, cy))
    d.text((cx, cy + ch + 3), tag[:44], fill=(232, 232, 232))
    d.text((cx, cy + ch + 16), "row-diff %.4f" % r, fill=(215, 215, 215))
out = os.path.join(OUT, "BANDING_TERMS.png")
canvas.save(out)
print()
print("   written: %s" % os.path.relpath(out, ROOT))
