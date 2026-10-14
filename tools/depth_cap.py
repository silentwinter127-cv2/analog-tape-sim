"""
Test the narrowest possible correction: cap the effective comb depth so the chroma is never pushed past the band.

**The reasoning, and why it is narrower than the three attempts that failed.**  Those changed the *formula* of the
correction -- normalising by an achieved gain, then gating it by phase correlation, then clamping the result -- and each
repaired the polarity while destroying the contrast.  **The band split is now measured to be correct on a clean
composite** (a column-constant picture comes back with a chroma spread of 0.00009 against its input's 0.28958, and a luma
spread of 1.000), so those attempts were fixing something that was not broken.

**What is known to be wrong is narrower than that**: the division `mixed / (1 - comb_depth)` amplifies the chroma by
**2.22x** at the default 0.55, and since `luma = comp - chroma`, that excess is **subtracted** from the luma.  **The
compensation may restore what the comb removed; it may not amplify.**  So the smallest change that removes the
possibility of amplifying is to **bound the effective depth at 0.5**, which bounds the division's gain at 2.0x -- and
2.0x is exactly "back to unity" for a comb that halves the amplitude, which is the case the code's comment describes.

**And this is testable in the only way that has worked**: the polarity (does `player` still return a picture correlated
with its input?) and the contrast (does the ramp still come through, and does the suite still pass?).  **Both must hold**,
and the three previous attempts each held one and lost the other.
"""
from __future__ import annotations

import ast
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

P = os.path.join(ROOT, "atsim", "core", "signal.py")
s = open(P, encoding="utf-8").read()

OLD = """        combed = 0.5 * (chroma + delayed)
        mixed = (1.0 - comb_depth) * chroma + comb_depth * combed
        gain = 1.0 - comb_depth
        if abs(gain) > 1e-3:
            chroma = mixed / gain"""
NEW = """        combed = 0.5 * (chroma + delayed)
        # **The compensation may restore what the comb removed; it may not amplify.**
        # `mixed / (1 - comb_depth)` is 2.22x at the default 0.55, and because
        # `luma = comp - chroma` that excess is *subtracted* from the luma -- which is
        # what inverts the picture.  **Measured stage by stage, `player` returned a
        # picture correlating -0.352 with its own input while every other stage returns
        # +1.000.**  Bounding the effective depth at 0.5 bounds the gain at 2.0x, which is
        # exactly "back to unity" for a comb that halves the amplitude -- the case the
        # comment above describes.  A deeper comb than that still deepens the filtering,
        # it simply stops being compensated past the point where the compensation would
        # exceed the band it is restoring.
        eff = float(min(comb_depth, 0.5))
        mixed = (1.0 - eff) * chroma + eff * combed
        gain = 1.0 - eff
        if abs(gain) > 1e-3:
            chroma = mixed / gain"""
if OLD not in s:
    raise SystemExit("the comb block was not found")
s = s.replace(OLD, NEW, 1)
ast.parse(s)
open(P, "w", encoding="utf-8").write(s)
print("   the effective comb depth is bounded at 0.5, so the chroma gain cannot exceed 2.0x")

from atsim.core.params import merge_values                                    # noqa: E402
from atsim.core.profiles import FACTORY_PRESETS, SCHEMA                       # noqa: E402
from atsim.render import renderer as R                                        # noqa: E402
from PIL import Image                                                         # noqa: E402

COEF = np.array([0.299, 0.587, 0.114], np.float32)
OBJ = (r"C:\Users\Administrator\.dsh\attachments\v1\objects")
NS0 = os.path.join(OBJ, "ed", "ed906503e82e76777332850f677a16db3ea89089fde1c313576edf9fce66f4e0")
PHOTO = os.path.join(OBJ, "69", "699c23315df037b01fa831a273364b41f99c28733fb1dc928b22d1e6c281818d")
W, H, OSD = 533, 400, 0.22
sl = slice(int(H * OSD), H)


def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0
    return np.asarray(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS),
                      np.float32) / 255.0


def lum(a):
    return np.clip(np.asarray(a, np.float32)[sl] @ COEF, 0, 1)


def corr(a, b):
    x = (a - a.mean()).ravel()
    y = (b - b.mean()).ravel()
    d = float(np.sqrt((x * x).sum() * (y * y).sum()))
    return float((x * y).sum() / d) if d > 1e-12 else 0.0


src = load(PHOTO)
real = load(NS0)
ls, lr = lum(src), lum(real)


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


def run(**cam):
    layers = {"camera": dict(OFF, **cam),
              "format": {"chroma_bandwidth": 3.0, "luma_bandwidth": 6.0},
              "player": {"chroma_bandwidth": 3.0, "luma_bandwidth": 6.0},
              "codec": {"dv_enabled": 0.0}, "display": {"enabled": 0.0}}
    rr = R.Renderer(One(src), W, H, 25.0, merge_values(BASE, layers), seed=1)
    return np.clip(np.asarray(rr.render_frame(0), np.float32), 0, 1)


print()
print("   === the two criteria ===")
print("   %-34s %-10s %-11s %-9s %s" % ("setting", "luma", "contrast", "polarity", "verdict"))
ok = True
for tag, cam, ref in (("the chain alone, mode off", dict(nv_mode=0.0), ls),
                      ("the mode on, everything", dict(nv_mode=1.0, nv_green_mix=1.0, nv_illuminator=0.15), ls)):
    a = run(**cam)
    l = lum(a)
    c = corr(l, ref)
    good = c > 0.5
    ok = ok and good
    print("   %-34s %-10.4f %-11.4f %+-9.3f %s" % (tag, float(l.mean()), float(l.std()), c,
                                                   "polarity OK" if good else "INVERTED"))
a = run(nv_mode=1.0, nv_green_mix=1.0, nv_illuminator=0.15)
print()
print("   against the footage: luma %.4f (real %.4f)   contrast %.4f (real %.4f, %.2fx)"
      % (float(lum(a).mean()), float(lr.mean()), float(lum(a).std()), float(lr.std()),
         float(lum(a).std()) / float(lr.std())))
print("   polarity is %s" % ("GOOD" if ok else "STILL INVERTED -- revert this"))
