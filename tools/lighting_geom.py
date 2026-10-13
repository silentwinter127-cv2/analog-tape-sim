"""
Measure the illuminator's actual spatial falloff, because the model's field was built on an assumption.

**The assumption in the code.**  `illuminator_field` computes `dy = (1.0 - yy) * 1.6 + 0.25` -- a distance measured
**from the top of the frame downward** -- so the light is brightest along the top edge and falls away below it.  That is
the geometry of a **lamp mounted above the lens**, which is where a separate IR illuminator often sits.

**But the camcorder's own NightShot lamp is coaxial with the lens** -- it is the same optical axis, which is why the
effect is even across the frame and why the subject is lit from the camera's direction.  **On that geometry the picture
is brightest at the centre and falls off radially**, and the two are easy to tell apart: one has its maximum on an edge,
the other at a point.

**So the measurement is the frame's own brightness as a function of position**, compared against both models:

  * **the radial model** -- brightness against distance from the frame's centre;
  * **the top-down model** -- brightness against distance from the top edge.

**Whichever the data follows is the one the field should be built on**, and the statistic is the correlation of the
brightness with each, plus the location of the maximum.  A model built on the wrong geometry would be a *plausible*
lighting fault rather than an obvious one, which is why it has to be measured rather than recognised.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from atsim.codecs import nightvision as nv                                    # noqa: E402
from atsim.core.params import merge_values                                    # noqa: E402
from atsim.core.profiles import FACTORY_PRESETS, SCHEMA                       # noqa: E402
from atsim.render.renderer import Renderer                                    # noqa: E402

COEF = np.array([0.299, 0.587, 0.114], np.float32)
OBJ = (r"C:\Users\Administrator\.dsh\attachments\v1\objects")
FRAMES = [("NIGHTSHOT 0", os.path.join(OBJ, "ed",
                                       "ed906503e82e76777332850f677a16db3ea89089fde1c313576edf9fce66f4e0")),
          ("NIGHTSHOT 1", os.path.join(OBJ, "f2",
                                       "f2dd7803fd91d124a7c16be3c6f97df679c8a83d89631a462e9010f13a3e7c5e")),
          ("NIGHTSHOT 2", os.path.join(OBJ, "b9",
                                       "b94a0e8402a11788032bd0af9e81de7fc4713d3a648384f2c9ff57b21c21f4b6"))]
OSD = 0.22


def geometry(lum):
    """How well the brightness follows each model, and where the maximum sits."""
    h, w = lum.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    yn, xn = yy / max(1, h - 1), xx / max(1, w - 1)
    # the radial model: distance from the frame's centre
    rad = np.sqrt((yn - 0.5) ** 2 + (xn - 0.5) ** 2)
    # the top-down model: distance from the top edge, as the code computes it
    top = yn
    v = (lum - lum.mean()).ravel()
    c_rad = float(np.corrcoef(v, (rad - rad.mean()).ravel())[0, 1])
    c_top = float(np.corrcoef(v, (top - top.mean()).ravel())[0, 1])
    # where the brightest 10% actually is
    thr = np.quantile(lum, 0.90)
    m = lum >= thr
    return c_rad, c_top, float(yn[m].mean()), float(xn[m].mean())


print("   === the real frames: does the light fall off from the centre or from the top? ===")
print("   %-14s %-14s %-14s %-24s %s"
      % ("frame", "corr(radial)", "corr(top)", "the brightest 10% sits at", "which model?"))
for name, p in FRAMES:
    a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0
    lum = np.clip(a[int(a.shape[0] * OSD):] @ COEF, 0, 1)
    cr, ct, my, mx = geometry(lum)
    verdict = ("RADIAL (stronger negative correlation with distance from centre)"
               if abs(cr) > abs(ct) + 0.02 else
               ("TOP-DOWN" if abs(ct) > abs(cr) + 0.02 else "neither clearly"))
    print("   %-14s %-14.3f %-14.3f y %-6.2f x %-6.2f      %s" % (name, cr, ct, my, mx, verdict))

print()
print("   and the model's own field, which is what it uses:")
for tag, spec in (("the code's field", nv.NightVisionSpec(illuminator=0.15)),
                  ("beam at the default", nv.NightVisionSpec(illuminator=0.15, beam_x=0.5, beam_y=0.5, beam_angle=0.55))):
    f = nv.illuminator_field(200, 266, spec)
    cr, ct, my, mx = geometry(f)
    print("   %-22s corr(radial) %+.3f  corr(top) %+.3f   brightest at y %.2f x %.2f"
          % (tag, cr, ct, my, mx))
    print("      the field's own range: %.4f at the top-centre to %.4f at the bottom" % (float(f[0].max()), float(f[-1].mean())))

print()
print("   === and what the model actually renders, measured the same way ===")
PHOTO = os.path.join(OBJ, "69", "699c23315df037b01fa831a273364b41f99c28733fb1dc928b22d1e6c281818d")
W, H = 533, 400
src = np.asarray(Image.open(PHOTO).convert("RGB"), np.float32) / 255.0
src = np.asarray(Image.fromarray((np.clip(src, 0, 1) * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS),
                 np.float32) / 255.0


class One:
    height, width = H, W
    fps = 25.0
    frame_count = 1

    def frame(self, i):
        return src


BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])
for tag, cam in (("the model, factory settings", {"nv_mode": 1.0, "nv_green_mix": 1.0}),
                 ("the model, beam aimed at centre", {"nv_mode": 1.0, "nv_green_mix": 1.0,
                                                      "nv_beam_x": 0.5, "nv_beam_y": 0.5, "nv_beam_angle": 0.9})):
    layers = {"camera": cam,
              "format": {"chroma_bandwidth": 3.0, "luma_bandwidth": 6.0},
              "player": {"chroma_bandwidth": 3.0, "luma_bandwidth": 6.0}}
    rr = Renderer(One(), W, H, 25.0, merge_values(BASE, layers), seed=20240101)
    a = np.clip(np.asarray(rr.render_frame(0), np.float32), 0, 1)
    lum = np.clip(a[int(H * OSD):] @ COEF, 0, 1)
    cr, ct, my, mx = geometry(lum)
    print("   %-30s corr(radial) %+.3f  corr(top) %+.3f   brightest at y %.2f x %.2f" % (tag, cr, ct, my, mx))
