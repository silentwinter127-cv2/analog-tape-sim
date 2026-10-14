"""
Test whether the banding residual is INHERITED from the source photograph or ADDED by the chain.

**The three numbers that pose the question:**

    the footage             0.07562
    the render              0.10953
    the source photograph   **0.12737**

**so the input already carries more row structure than the footage, and the render sits BETWEEN them.**  The chain is
therefore *reducing* what it is given, not amplifying it -- and if that is the whole story, then the model's residual is
not a model defect at all but the input's own content, and the right response is to say so rather than to keep looking for
a term that adds it.

**The test removes the ambiguity by construction.**  A copy of the source is made whose row structure is scaled down --
row means pulled towards a smoothed version of themselves -- until the copy matches the footage's 0.07562, with everything
else about it untouched.  **Then the same render is run on both**, and the prediction is stated so it can fail:

  * **if the render on the flattened source falls to about the footage's value**, the residual is inherited and the chain
    is adding nothing -- the earlier search was for a term that does not exist;
  * **if the render on the flattened source stays well above it**, the chain IS adding structure of its own and there is a
    term left to find.

**And a second control goes the other way**: the source's row structure is *amplified*, and if the render follows it
proportionally then the chain is transparent to this quantity, which is the cleanest form of the same answer.
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


def rescale_rows(a, factor):
    """
    Scale the picture's ROW structure by `factor`, leaving everything else alone.

    The row means are separated from the row-to-row detail: a picture is `rowmean(x) + detail(x, y)`, so scaling the
    row-to-row part and putting the original means back changes exactly the quantity `rowsig` measures and nothing else.
    """
    arr = np.asarray(a, np.float32)
    g = arr @ COEF
    k = 5
    trend = np.apply_along_axis(lambda c: np.convolve(c, np.ones(k, np.float32) / k, mode="same"), 0, g)
    delta = g - trend                       # the row-to-row structure about a local trend
    g2 = trend + factor * delta
    g2 = np.clip(g2, 1e-6, None)
    #: carry the change through all three channels by the same ratio, so the colour is untouched
    return np.clip(arr * (g2 / np.maximum(g, 1e-6))[..., None], 0, 1).astype(np.float32)


src = load(PHOTO)
real = load(NS0)
print("   the footage            rowsig %.5f" % rowsig(real))
print("   the source photograph  rowsig %.5f" % rowsig(src))


class One:
    height, width = H, W
    fps = 25.0
    frame_count = 1

    def __init__(self, a):
        self.a = a

    def frame(self, i):
        return self.a


BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])
OFF = {"halation": 0.0, "ccd_smear": 0.0, "flare_defocus": 0.0, "lens_flare": 0.0,
       "overflow_drain": 0.0, "chroma_aberration": 0.0, "sensor_noise": 0.0, "noise_grain": 0.0,
       "fixed_pattern": 0.0, "spike_spread": 0.0, "motion_blur": 0.0, "stuck_fraction": 0.0,
       "bloom_threshold": 0.0, "overflow_width": 0.0, "sharpness": 1.0}


def run(src_img, mode=1.0):
    layers = {"camera": dict(OFF, nv_mode=mode),
              "format": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0},
              "player": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0},
              "codec": {"dv_enabled": 0.0}, "display": {"enabled": 0.0}}
    rr = Renderer(One(src_img), W, H, 25.0, merge_values(BASE, layers), seed=20240101)
    return np.clip(np.asarray(rr.render_frame(0), np.float32), 0, 1)


print()
print("   === the source's row structure scaled, and what the render does with it ===")
print("   %-12s %-14s %-14s %-14s %s" % ("factor", "source rowsig", "render rowsig", "render cut", "verdict"))
rows = []
for f in (0.3, 0.5, 0.7, 1.0, 1.4, 1.8):
    s2 = rescale_rows(src, f)
    a = run(s2)
    rows.append((f, rowsig(s2), rowsig(a), a, s2))
    print("   %-12.2f %-14.5f %-14.5f %+.5f" % (f, rowsig(s2), rowsig(a), rowsig(a) - rowsig(src)))

f1 = [r for r in rows if abs(r[0] - 1.0) < 1e-6][0]
print()
print("   === and the flattest case, to answer the question directly ===")
#: a factor that takes the SOURCE to the footage's own value, so the comparison is at matched input
target = rowsig(real)
lo, hi = 0.2, 1.0
for _ in range(18):
    mid = 0.5 * (lo + hi)
    if rowsig(rescale_rows(src, mid)) > target:
        hi = mid
    else:
        lo = mid
flat = rescale_rows(src, 0.5 * (lo + hi))
a_flat = run(flat)
print("   the source flattened to %.5f (the footage's own value)" % rowsig(flat))
print("   the render of THAT source:  %.5f" % rowsig(a_flat))
print("   the render of the original: %.5f" % rowsig(f1[3]))
print("   the footage:                %.5f" % target)
print()
if rowsig(a_flat) < 1.3 * target:
    print("   -> the residual is INHERITED: with the input matched to the footage, the render lands near it, so the")
    print("      chain adds little and the earlier search was for a term that does not exist.")
else:
    print("   -> the chain ADDS structure of its own: even from a matched input the render stays well above it.")

gap = 6
panels = [("the real footage", real), ("the original source", src),
          ("the source flattened to the footage", flat), ("the render of the original", f1[3]),
          ("the render of the flattened source", a_flat)]
cw, ch = W // 2, (H - int(H * OSD)) // 2
canvas = Image.new("RGB", ((cw + gap) * 3 + gap, (ch + 30) * 2 + 26), (16, 16, 18))
d = ImageDraw.Draw(canvas)
d.text((6, 5), "IS THE RESIDUAL INHERITED OR ADDED?  the source scaled, and what the chain does with it",
       fill=(242, 242, 242))
for i, (tag, a) in enumerate(panels):
    x = gap + (i % 3) * (cw + gap)
    y = 26 + (i // 3) * (ch + 30)
    canvas.paste(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).resize((cw, ch), Image.LANCZOS), (x, y))
    d.text((x, y + ch + 3), "%s  rowsig %.4f" % (tag, rowsig(a)), fill=(234, 234, 234))
out = os.path.join(OUT, "INHERITED.png")
canvas.save(out)
print()
print("   written: %s" % os.path.relpath(out, ROOT))
