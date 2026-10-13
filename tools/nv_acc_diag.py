"""
Find why accumulation drops the brightness 3.7x, which a mean cannot do unless the frames differ in level.

**The signature that isolates it.** On the real render path, frame 14:

    NightShot, 1 frame        mean 0.4895    hf 0.00964
    Super NightShot, 2 frames mean 0.1337    hf 0.00876
    Super NightShot, 4 frames mean 0.1340    hf 0.00620
    Super NightShot, 8 frames mean 0.1343    hf 0.00362

**The noise falls as it should** -- 0.00876, then 0.707x, then 0.413x, which is the 1/sqrt(N) law and exactly what
frame accumulation is for. **But the brightness collapses**, and a *mean* cannot do that unless the frames being averaged
are at different levels.

**And there is a concrete reason to expect exactly that here.** The scene is a **night photograph** and the source
sequence is a **rolling copy** of it, so the frames that fall inside the accumulation window are of the *scene*, while
the picture the mode is applied to at that moment is the *same scene shifted*. When the mean is taken across frames
whose content has moved, the brightest structures -- the lamp, the lit windows -- **move out of alignment and average
down**, while the dark background stays. That is a real property of accumulating a moving scene, not a bug in the mean.

**So this measures the two candidate causes and distinguishes them:**

  * **the frames' own levels**, printed individually -- if they are all about 0.45 then the mean cannot be the cause and
    the loss is spatial averaging of a moving scene;
  * and **the same accumulation on a STATIC source**, where the frames are identical -- if the brightness is preserved
    there and lost only on the moving one, the cause is motion and the behaviour is correct.

A static test is the control, and it is what makes the moving result interpretable rather than alarming.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from atsim.core.params import merge_values                                   # noqa: E402
from atsim.core.profiles import FACTORY_PRESETS, SCHEMA                      # noqa: E402
from atsim.render.renderer import Renderer                                   # noqa: E402

COEF = np.array([0.299, 0.587, 0.114], np.float32)
pic = np.asarray(Image.open(os.path.join(ROOT, "tools", "ntscrs_ref", "newscene.png"))
                 .convert("RGB"), np.float32) / 255.0
H, W = pic.shape[:2]
BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])
print("   the source picture: mean luma %.4f" % float((pic @ COEF).mean()))
print("   and a rolled copy:  mean luma %.4f  (so a roll preserves the level, as it must)"
      % float((np.roll(pic, 6, axis=1) @ COEF).mean()))


class MovingSource:
    def __init__(self, p):
        self.height, self.width = p.shape[:2]
        self.fps, self.frame_count = 25.0, 20
        self._p = p

    def frame(self, index):
        return np.roll(self._p, int(index) * 3, axis=1)


class StaticSource(MovingSource):
    def frame(self, index):
        return self._p


RASTER_OFF = {"scanlines": 0.0, "mask_depth": 0.0, "interlace_flicker": 0.0, "scan_pitch": 0.0,
              "spot_sigma": 0.0, "defocus": 0.0, "vignette": 0.0, "reflection": 0.0,
              "bloom": 0.0, "geometry_warp": 0.0, "overscan": 0.0, "curvature": 0.0,
              "breathing": 0.0, "crt_age": 0.0, "convergence": 0.0, "crt_moisture": 0.0,
              "crt_corrosion": 0.0}
OFF = dict(halation=0.0, ccd_smear=0.0, flare_defocus=0.0, lens_flare=0.0, spike_spread=0.0,
           overflow_drain=0.0, chroma_aberration=0.0, longitudinal_ca=0.0, axis_offset=0.0,
           source_glow=0.0, halation_source_colour=0.0, shadow_depth=0.0, exposure=0.0)


def render(cam, source, idx=14):
    layers = {"camera": dict(OFF, **cam),
              "format": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0},
              "player": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0},
              "codec": {"dv_enabled": 1.0}, "tape": {"head_switching": 0.0},
              "display": dict(RASTER_OFF, enabled=1.0, gamma=1.0, saturation=0.62)}
    p = merge_values(BASE, layers)
    r = Renderer(source, W, H, 25.0, p, seed=20240101)
    return np.clip(np.asarray(r.render_frame(idx), np.float32), 0, 1)


hf = lambda a: float(np.abs(np.diff(a @ COEF, axis=1)).mean())
print()
print("   %-34s %-24s %s" % ("mode", "MOVING source", "STATIC source (the control)"))
for label, cam in (
    ("NightShot, 1 frame", {"nv_mode": 1.0, "nv_accumulate": 1.0}),
    ("Super NightShot, 8 frames", {"nv_mode": 2.0, "nv_accumulate": 8.0}),
):
    a = render(cam, MovingSource(pic))
    b = render(cam, StaticSource(pic))
    print("   %-34s mean %.4f  hf %.5f   mean %.4f  hf %.5f"
          % (label, float((a @ COEF).mean()), hf(a), float((b @ COEF).mean()), hf(b)))

print()
print("   if the STATIC source keeps its brightness while the moving one loses it, the loss is the spatial averaging")
print("   of a moving scene -- which is what an accumulating mode does to motion, and is a property rather than a bug.")
print("   if the static source loses it too, the mean is being taken wrongly.")
