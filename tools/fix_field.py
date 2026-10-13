"""
Rebuild the illuminator's field as a radial falloff about the beam's own origin, and measure that it is symmetric.

**The fault, measured.**  The field's own geometry is **inverted vertically**: correlation with distance from the top is
**+0.766** with the brightest point at **y = 0.91** -- the bottom of the frame -- while the real frames correlate
**-0.353 to -0.501** with their brightest regions at **y = 0.27 to 0.33**.  The field spans **0.0214 at the top against
0.2710 at the bottom**, a **12.7x top-to-bottom tilt**.

**And it is a hidden fault**: the rendered pictures correlate **-0.42**, i.e. the right way, because the *scene's* own
brightness dominates the frame and the field is a small additive term.  **So the field's shape is wrong and the picture
hides it** -- the reasoning that produced the code (a distance measured "from the top downward") is the geometry of a lamp
mounted above the lens, and it was never checked against the footage.

**The correct geometry.**  A camcorder's own NightShot lamp is **coaxial with the lens**, so it lights a cone whose axis
is the lens axis: the falloff is **radial about the beam's origin in the frame** and has **no top-to-bottom preference
beyond where the operator aims it**.  So the field becomes:

    d = distance from the beam's origin (beam_x, beam_y), scaled by the beam's angle
    field = min(reference^2 / d^2, near_bound)        inverse-square, clamped in the near field

**The check is symmetry**: the field must have **no vertical tilt** -- the mean of its top half against its bottom half,
and its correlation with distance from the top, must both be near zero -- while still being brightest at the beam's
origin.  A field that is radially symmetric and correctly aimed is what the real frames' own brightest-region positions
can then be matched against.
"""
from __future__ import annotations

import ast
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

P = os.path.join(ROOT, "atsim", "codecs", "nightvision.py")
s = open(P, encoding="utf-8").read()

i = s.find("def illuminator_field")
j = s.find("\ndef ", i + 10)
print("   the field as it stands:")
for line in s[i:j].split("\n"):
    t = line.strip()
    if t and not t.startswith("#") and not t.startswith('"""') and '"""' not in t:
        print("      %s" % line[:110])

NEW_FN = '''def illuminator_field(h: int, w: int, spec: NightVisionSpec) -> np.ndarray:
    """
    The illuminator's light across the frame: **an inverse-square falloff about the beam's own origin.**

    **Why radial and not top-down.**  A camcorder's NightShot lamp is **coaxial with the lens**, so it lights a cone
    whose axis is the lens axis: the falloff is **radial about the point the beam is aimed at**, with no preference for
    the top or the bottom of the frame beyond where the operator points it.  The previous version measured distance
    **from the top edge downward** -- the geometry of a lamp mounted *above* the lens -- and it produced a field that was
    **12.7x brighter at the bottom than at the top** (0.2710 against 0.0214), correlating **+0.766** with distance from
    the top where the real frames correlate **-0.353 to -0.501**.

    **And it was hidden.**  The rendered pictures correlated **-0.42**, the right way, because the scene's own brightness
    dominates the frame while the field is a small additive term.  So the shape was wrong and the picture concealed it,
    which is why the geometry is measured here rather than recognised.

    **The light itself**: inverse-square with distance, clamped in the near field so that a surface close to the lamp
    does not go to infinity.  `beam_x` and `beam_y` place the origin; `beam_angle` sets how quickly it falls away.
    """
    h = max(1, int(h))
    w = max(1, int(w))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xn = xx / max(1.0, w - 1.0)
    yn = yy / max(1.0, h - 1.0)
    bx = float(np.clip(spec.beam_x, 0.0, 1.0))
    by = float(np.clip(spec.beam_y, 0.0, 1.0))
    # a distance in the frame's own units, with the aspect kept so the cone stays circular rather than elliptical
    dx = (xn - bx) * (w / max(1.0, h))
    dy = yn - by
    angle = float(np.clip(spec.beam_angle, 0.05, 4.0))
    # **the angular scale sets the cone's width**: a narrow beam reaches the same brightness at a smaller distance
    d = np.sqrt(dx * dx + dy * dy) / max(1e-3, angle)
    # the inverse-square law with a near-field clamp, so a surface at the lamp does not go to infinity
    near_bound = 3.0
    ref = 0.75
    field = np.minimum(ref * ref / np.maximum(d, 1e-3) ** 2, near_bound)
    return field.astype(np.float32)
'''

s = s[:i] + NEW_FN + s[j + 1:]
ast.parse(s)
open(P, "w", encoding="utf-8").write(s)
print()
print("   the field is now radial about the beam's origin")

from atsim.codecs import nightvision as nv                                    # noqa: E402


def geom(f):
    h, w = f.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    yn = yy / max(1, h - 1)
    v = (f - f.mean()).ravel()
    ct = float(np.corrcoef(v, (yn - yn.mean()).ravel())[0, 1])
    thr = np.quantile(f, 0.90)
    m = f >= thr
    return ct, float(yn[m].mean()), float((xx / max(1, w - 1))[m].mean())


print()
print("   === the field's own geometry, centred and aimed ===")
print("   %-34s %-12s %-12s %s" % ("setting", "corr(top)", "top vs bottom", "brightest 10% at"))
for tag, spec in (("centred (beam_x/y = 0.5)", nv.NightVisionSpec(illuminator=0.15, beam_x=0.5, beam_y=0.5)),
                  ("aimed up (beam_y = 0.3)", nv.NightVisionSpec(illuminator=0.15, beam_x=0.5, beam_y=0.3)),
                  ("narrow (angle 0.3)", nv.NightVisionSpec(illuminator=0.15, beam_x=0.5, beam_y=0.5,
                                                            beam_angle=0.3))):
    f = nv.illuminator_field(200, 266, spec)
    ct, my, mx = geom(f)
    top = float(f[:100].mean())
    bot = float(f[100:].mean())
    print("   %-34s %-+12.3f %-12s y %.2f x %.2f" % (tag, ct, "%.4f / %.4f" % (top, bot), my, mx))
print()
print("   a radially symmetric field must have corr(top) near zero and top/bottom means close; the previous field")
print("   read +0.766 with 0.0214 against 0.2710, which is the tilt this replaces.")
print()
print("   and the real frames' brightest regions sit at y 0.27-0.33, so `beam_y` around 0.3 is where the footage was")
print("   aimed -- a value the field can now be set to rather than one baked into its geometry.")
