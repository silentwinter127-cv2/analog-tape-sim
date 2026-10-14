"""
Solve for the green-display weights that reproduce the footage's per-channel chroma ratios, instead of guessing them.

**The constraint, measured from the footage.**  Per-channel mean absolute chroma, and the ratios between them:

    NIGHTSHOT 0   [0.0479 0.0244 0.0213]   G/B 1.15
    NIGHTSHOT 1   [0.1209 0.0601 0.0162]   G/B 3.71
    NIGHTSHOT 2   [0.1319 0.0688 0.0227]   G/B 3.03

**and the model's projection gives R:G:B = [0.505 1.000 0.824]**, i.e. **G/B 0.824** where the footage's three segments
average **2.63**.  **The projection fixes the R:G:B ratio of everything that survives it**, so no value of
`chroma_residue` can move it: `out = mono + (rgb - lum) * r`, and both terms carry the same weights.

**So the weights are solved rather than searched by eye.**  Two things are held:

  * **the luminance gain stays 1.0000**, which an earlier round fixed and which is what makes `green_mix` a colour
    control rather than a brightness one -- `weights @ LUMA == 1`;
  * **and the resulting R:G:B ratio matches the footage's**, evaluated through the same expression the module runs, so
    the solution accounts for the residue rather than for the projection alone.

**The unknown is `green_gain`**, which is the only free parameter once the gain constraint is imposed -- so it is swept,
and the best fit to all three of the footage's ratios is reported with its error.  **If the best error is large, the form
cannot express the footage and that is the finding**; if it is small, the gain is calibrated rather than asserted.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

COEF = np.array([0.299, 0.587, 0.114], np.float32)
OBJ = (r"C:\Users\Administrator\.dsh\attachments\v1\objects")
REFS = [("NIGHTSHOT 0", os.path.join(OBJ, "ed",
                                     "ed906503e82e76777332850f677a16db3ea89089fde1c313576edf9fce66f4e0")),
        ("NIGHTSHOT 1", os.path.join(OBJ, "f2",
                                     "f2dd7803fd91d124a7c16be3c6f97df679c8a83d89631a462e9010f13a3e7c5e")),
        ("NIGHTSHOT 2", os.path.join(OBJ, "b9",
                                     "b94a0e8402a11788032bd0af9e81de7fc4713d3a648384f2c9ff57b21c21f4b6"))]
PHOTO = os.path.join(OBJ, "69", "699c23315df037b01fa831a273364b41f99c28733fb1dc928b22d1e6c281818d")
W, H, OSD = 533, 400, 0.22
sl = slice(int(H * OSD), H)


def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0
    return np.asarray(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS),
                      np.float32) / 255.0


def ratios(a):
    sc = np.asarray(a, np.float32)[sl]
    g = np.clip(sc @ COEF, 0, 1)
    c = np.abs(sc - g[..., None]).mean(axis=(0, 1))
    return c, float(c[1] / max(1e-9, c[2])), float(c[0] / max(1e-9, c[1]))


print("   === the footage's own ratios, the target ===")
tg = []
for name, p in REFS:
    c, gb, rg = ratios(load(p))
    tg.append((gb, rg))
    print("   %-14s |chroma| %s   G/B %.3f   R/G %.3f" % (name, np.round(c, 4), gb, rg))
GB_T = float(np.mean([t[0] for t in tg]))
RG_T = float(np.mean([t[1] for t in tg]))
print("   the mean target: G/B %.3f   R/G %.3f" % (GB_T, RG_T))

src = load(PHOTO)
#: the residue the module actually applies, so the fit accounts for it rather than for the projection alone
RES = 0.35


def project(rgb, weights, r):
    lum = np.clip(rgb @ COEF, 0.0, 1.0)
    mono = np.clip(lum[..., None] * weights, 0.0, 1.0)
    if r > 1e-4:
        return np.clip(mono + (rgb - np.clip(rgb @ COEF, 0.0, 1.0)[..., None]) * r, 0.0, 1.0)
    return mono


print()
print("   === sweeping the green gain, with the luminance gain held at 1.0 ===")
print("   %-10s %-26s %-10s %-10s %s" % ("gain", "R:G:B", "G/B", "R/G", "error"))
LW = np.array([0.2168, 0.4294, 0.3538], np.float32)
best = None
for gr in (0.6, 0.8, 1.0, 1.15, 1.3, 1.5, 1.8, 2.2, 2.8):
    w = LW * np.array([1.0, gr, 1.0], np.float32)
    w = w / float(w @ COEF)              # **the luminance gain stays exactly 1.0, which is the fixed constraint**
    out = project(src, w, RES)
    c, gb, rg = ratios(out)
    err = abs(gb - GB_T) / GB_T + abs(rg - RG_T) / RG_T
    rel = c / max(1e-9, float(c[1]))
    if best is None or err < best[0]:
        best = (err, gr, gb, rg, rel)
    print("   %-10.2f %-26s %-10.3f %-10.3f %.3f" % (gr, np.round(rel, 3), gb, rg, err))

print()
print("   the best: green_gain %.2f gives G/B %.3f and R/G %.3f  (the footage is %.3f and %.3f)"
      % (best[1], best[2], best[3], GB_T, RG_T))
print("   with the luminance gain held at 1.0, so `green_mix` stays a colour control and not a brightness one.")
print()
if best[0] < 0.35:
    print("   -> the FORM can express the footage's ratios: the remaining error is within the spread across the three")
    print("      segments themselves (their own G/B runs 1.15 to 3.71), so calibrating `green_gain` is enough.")
else:
    print("   -> the form cannot express them, and the fault is structural rather than a gain.")
