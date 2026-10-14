"""
Look at the two maps the vision model produces, since `_night_vision` is where the banding is made.

**What the trace established.**  Tracing inside the camera, writing an image per call:

    call               mode0      mode1      difference
    (the input)        0.12737
    **_night_vision**  0.12737   **0.19899**  **+0.07162**   <- the whole excess, in one call
    _roll_off          0.12695    0.19785    +0.07090       (and every later call leaves it)
    _shadow_depth .. _sharpen   0.12695    0.19785    unchanged

**and the picture shows what it is**: mode 1's `_night_vision` output is **a green diagonally-banded layer over the
picture**, not a degraded version of it.  **So the banding is a PATTERN, and the two things inside that call which can
produce a pattern are the two maps the vision model supplies** -- the near-IR albedo (multiplied into the picture by
`ir_highlights`) and the relative depth (multiplied into the illuminator's field).

**And neither map has ever been looked at.**  They have been measured through statistics of their effect on the picture,
and seven of those statistics failed.  **So they are written out here directly**: the albedo map, the depth map, and each
one's own row-difference -- because a map that itself bands would explain everything and would be visible immediately.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from atsim.codecs import material                                             # noqa: E402
from atsim.codecs.nightvision import albedo_map, illuminator_field, NightVisionSpec  # noqa: E402
from atsim.core.standards import STANDARDS                                    # noqa: E402

COEF = np.array([0.299, 0.587, 0.114], np.float32)
OBJ = (r"C:\Users\Administrator\.dsh\attachments\v1\objects")
PHOTO = os.path.join(OBJ, "69", "699c23315df037b01fa831a273364b41f99c28733fb1dc928b22d1e6c281818d")
OUT = os.path.join(ROOT, "tools", "calib")
W, H = 533, 400

_a = np.asarray(Image.open(PHOTO).convert("RGB"), np.float32) / 255.0
SRC = np.asarray(Image.fromarray((np.clip(_a, 0, 1) * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS),
                 np.float32) / 255.0


def rowsig(a):
    """**A single-plane map has no channels to project**, so it is taken as it is; a picture is projected to luma."""
    arr = np.asarray(a, np.float32)
    g = np.clip(arr @ COEF, 0, 1) if arr.ndim == 3 else np.clip(arr, 0, 1)
    return float(np.abs(np.diff(g, axis=0)).mean()) / max(1e-9, float(g.std()))


print("   === the vision model's two maps, computed exactly as the stage does ===")
labelled = material.analyse(SRC, grid=8) if material.available() else None
if labelled is None:
    print("      no vision model available")
    raise SystemExit(0)
alb = labelled["albedo"]
conf = labelled["confidence"]
print("      the albedo grid %s: min %.3f max %.3f  (IR_ALBEDO's own range)" % (alb.shape, alb.min(), alb.max()))
print("      per cell, by row:")
for j in range(alb.shape[0]):
    print("         %s   conf %s" % (" ".join("%5.2f" % v for v in alb[j]),
                                    " ".join("%4.2f" % v for v in conf[j])))

dep = material.analyse_depth(SRC, grid=8) if material.available() else None
if dep is not None:
    print()
    print("      the depth grid %s: min %.3f max %.3f" % (dep["depth"].shape, dep["depth"].min(), dep["depth"].max()))
    for j in range(dep["depth"].shape[0]):
        print("         %s" % " ".join("%5.2f" % v for v in dep["depth"][j]))

#: **the maps as the stage builds them** -- `albedo_map` is what gets multiplied into the picture
est = np.power(np.clip(SRC @ COEF, 0, 1), 0.55)
am = albedo_map(H, W, labelled, fallback=est)
print()
print("   === `albedo_map`'s output, which `ir_highlights` multiplies into the picture ===")
print("      %s  min %.3f max %.3f  row-difference %.5f" % (am.shape, am.min(), am.max(), rowsig(am)))
print("      the source's row-difference for scale: %.5f" % rowsig(SRC))

std = STANDARDS["ntsc"] if isinstance(STANDARDS, dict) else STANDARDS
spec = NightVisionSpec(mode=1.0, illuminator=0.15, ir_response=1.0, ir_albedo_mix=0.7, green_mix=1.0,
                       ir_depth=dep["depth"] if dep else None)
field = illuminator_field(H, W, spec)
print()
print("   === and `illuminator_field`, which `add_illuminator` multiplies in ===")
print("      min %.5f max %.5f mean %.5f  row-difference %.5f  (max/min %.2f)"
      % (field.min(), field.max(), field.mean(), rowsig(field), field.max() / max(1e-9, field.min())))

gap = 8
panels = [("the source", SRC), ("the albedo map (x1)", np.repeat(am[..., None], 3, 2)),
          ("the illuminator field (normalised)", np.repeat((field / max(1e-9, field.max()))[..., None], 3, 2)),
          ("source x albedo, i.e. what ir_highlights does",
           np.clip(SRC * (am / max(1e-9, am.mean()))[..., None], 0, 1))]
cw, ch = W // 2, H // 2
canvas = Image.new("RGB", ((cw + gap) * 2 + gap, (ch + 34) * 2 + 26), (16, 16, 18))
d = ImageDraw.Draw(canvas)
d.text((6, 5), "THE TWO MAPS THE VISION MODEL SUPPLIES, AND WHAT `ir_highlights` DOES WITH THE ALBEDO",
       fill=(242, 242, 242))
for i, (tag, a) in enumerate(panels):
    x = gap + (i % 2) * (cw + gap)
    y = 26 + (i // 2) * (ch + 34)
    canvas.paste(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).resize((cw, ch), Image.LANCZOS), (x, y))
    d.text((x, y + ch + 3), tag, fill=(236, 236, 236))
    d.text((x, y + ch + 16), "row-diff %.4f" % rowsig(a), fill=(214, 214, 214))
out = os.path.join(OUT, "MAPS.png")
canvas.save(out)
print()
print("   written: %s" % os.path.relpath(out, ROOT))
