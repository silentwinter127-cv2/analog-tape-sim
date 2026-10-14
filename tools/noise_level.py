"""
The shot-noise level against the footage's own grain, since it is the largest measurable contributor to the residual.

**Where this stands.**  The visible defect -- coarse diagonal bands over the whole frame -- was the vision model's albedo
grid upsampled with nearest-neighbour, with a 7.5x step at every cell boundary, and fixing that removed it.  **The
residual is smaller and was then traced**: the camera's row-structure measure goes 0.12737 -> 0.16272 there, every tone
curve and lens term after it leaves the figure untouched, and **`shot_noise` alone accounts for far more than the whole of
the excess** -- measured on the source, 0.10 takes it from 0.12737 to 0.23482.

**So the term is the candidate, and it was added by me two rounds ago as `note.txt`'s signal-dependent noise.**  Its
default is 0.10.  **The question is what level matches the footage's own grain**, which is the only thing that can decide
it -- a term that is physically right can still be set too high.

**The measurement is the picture and the figures together**, because eleven statistics have now failed to characterise
this defect and the picture has been right each time.  **The prediction**: if the render's row measure falls towards the
footage's 0.07562 as `shot_noise` falls, the level was too high; **if it does not, the residual is not this term either**
and the honest answer is that it is the input's own resampling grain, since the photograph is 2364x1773 reduced to 533x400
and the footage is natively 693x516.
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


def grain(a):
    """The picture's fine-grain energy, which is what 'too noisy' means when a stripe measure cannot see it."""
    g = np.clip(np.asarray(a, np.float32)[sl] @ COEF, 0, 1)
    return float(np.abs(np.diff(g, axis=1)).mean()) / max(1e-9, float(g.std()))


REAL = load(NS0)
SRC = load(PHOTO)
print("   %-34s %-10s %-10s" % ("", "row struct", "fine grain"))
print("   %-34s %-10.5f %-10.5f" % ("the real NIGHTSHOT 0", rowsig(REAL), grain(REAL)))
print("   %-34s %-10.5f %-10.5f" % ("the source photograph", rowsig(SRC), grain(SRC)))


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


def run(**cam):
    cam = dict(cam)
    cam.setdefault("nv_mode", 1.0)
    layers = {"camera": dict(OFF, **cam),
              "format": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0},
              "player": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0},
              "codec": {"dv_enabled": 0.0}, "display": {"enabled": 0.0}}
    rr = Renderer(One(), W, H, 25.0, merge_values(BASE, layers), seed=20240101)
    return np.clip(np.asarray(rr.render_frame(0), np.float32), 0, 1)


print()
print("   === the shot-noise level ===")
print("   %-24s %-10s %-10s %-10s %s" % ("", "row struct", "fine grain", "mean", "vs the footage's row struct"))
shots = []
for v in (0.0, 0.02, 0.05, 0.10, 0.18):
    a = run(nv_shot_noise=v)
    shots.append(("shot_noise %.2f" % v, a, rowsig(a), grain(a)))
    print("   %-24s %-10.5f %-10.5f %-10.4f %+.5f"
          % ("shot_noise %.2f" % v, rowsig(a), grain(a),
             float(np.clip(a[sl] @ COEF, 0, 1).mean()), rowsig(a) - rowsig(REAL)))
print()
best = min(shots, key=lambda s: abs(s[2] - rowsig(REAL)))
print("   closest to the footage on row structure: %s (%.5f against %.5f)"
      % (best[0], best[2], rowsig(REAL)))
print("   **and its fine grain is %.5f against the footage's %.5f**, which is the other half of the question."
      % (best[3], grain(REAL)))

gap = 7
cw, ch = W // 2, (H - int(H * OSD)) // 2
panels = [("the real NIGHTSHOT 0", REAL), ("the model, shot_noise 0.10 (as shipped)", shots[3][1]),
          ("the model, shot_noise %.2f" % (0.02 if shots[1][0].endswith("0.02") else 0.0), shots[1][1]),
          ("the source photograph", SRC)]
canvas = Image.new("RGB", ((cw + gap) * 2 + gap, (ch + 34) * 2 + 28), (16, 16, 18))
d = ImageDraw.Draw(canvas)
d.text((6, 5), "THE SHOT-NOISE LEVEL -- the footage, the render as it ships, a lower level, and the input",
       fill=(242, 242, 242))
for i, (tag, arr) in enumerate(panels):
    x = gap + (i % 2) * (cw + gap)
    y = 24 + (i // 2) * (ch + 34)
    canvas.paste(Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).resize((cw, ch), Image.LANCZOS), (x, y))
    d.text((x, y + ch + 3), tag, fill=(238, 238, 238))
    d.text((x, y + ch + 16), "row struct %.4f   fine grain %.4f" % (rowsig(arr), grain(arr)), fill=(214, 214, 214))
out = os.path.join(OUT, "NOISE_LEVEL.png")
canvas.save(out)
print("   written: %s" % os.path.relpath(out, ROOT))
