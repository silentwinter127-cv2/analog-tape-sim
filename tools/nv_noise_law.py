"""
Verify that accumulation reduces the read noise, which is the reason the accumulating mode exists.

**The physical claim under test.** Read noise is added per frame at the sensor. Summing N frames adds the noise N times
**in quadrature** while the signal adds N times **coherently**, so the noise falls as `1/sqrt(N)`. **Applied after the
sum it would not fall at all** -- and then the model could not show why a night-vision camera has an accumulating mode,
which is the whole content of Super NightShot.

**So the test is a ratio, and it has an independent prediction.** With the read noise at a fixed level and everything
else held constant, the high-frequency energy should fall by about `sqrt(N)` between 1 frame and N frames. The earlier
unit test already established `sqrt(8) = 2.83` against a measured `2.878` on synthetic frames; this checks that the
same law survives **on the real render path, through the night-vision mode, with the read noise applied where it is
supposed to be**.

**And the control is the placement itself**: if the noise were added after the sum, the ratio would be 1.0 for every N.
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
L = lambda a: float((np.asarray(a, np.float32) @ COEF).mean())
hf = lambda a: float(np.abs(np.diff(np.asarray(a, np.float32) @ COEF, axis=1)).mean())
pic = np.asarray(Image.open(os.path.join(ROOT, "tools", "ntscrs_ref", "newscene.png"))
                 .convert("RGB"), np.float32) / 255.0
H, W = pic.shape[:2]
BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])


class MovingSource:
    """Frames that differ, so the noise is the only thing common to them that averages down -- a static source would
    make the sum trivially exact and could not show the law."""

    def __init__(self, p):
        self.height, self.width = p.shape[:2]
        self.fps, self.frame_count = 25.0, 20
        self._p = p

    def frame(self, index):
        return np.roll(self._p, int(index) * 3, axis=1)


RASTER_OFF = {"scanlines": 0.0, "mask_depth": 0.0, "interlace_flicker": 0.0, "scan_pitch": 0.0,
              "spot_sigma": 0.0, "defocus": 0.0, "vignette": 0.0, "reflection": 0.0,
              "bloom": 0.0, "geometry_warp": 0.0, "overscan": 0.0, "curvature": 0.0,
              "breathing": 0.0, "crt_age": 0.0, "convergence": 0.0, "crt_moisture": 0.0,
              "crt_corrosion": 0.0}
OFF = dict(halation=0.0, ccd_smear=0.0, flare_defocus=0.0, lens_flare=0.0, spike_spread=0.0,
           overflow_drain=0.0, chroma_aberration=0.0, longitudinal_ca=0.0, axis_offset=0.0,
           source_glow=0.0, halation_source_colour=0.0, shadow_depth=0.0, exposure=0.0,
           nv_read_noise=0.06)


def render(cam):
    layers = {"camera": dict(OFF, **cam),
              "format": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0},
              "player": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0},
              "codec": {"dv_enabled": 1.0}, "tape": {"head_switching": 0.0},
              "display": dict(RASTER_OFF, enabled=1.0, gamma=1.0, saturation=0.62)}
    p = merge_values(BASE, layers)
    r = Renderer(MovingSource(pic), W, H, 25.0, p, seed=20240101)
    return np.clip(np.asarray(r.render_frame(14), np.float32), 0, 1)


print("   with nv_read_noise = 0.06, on the real render path through the NightShot mode:")
print("   %-30s %-10s %-11s %-11s %s" % ("mode", "mean", "hf (noise)", "1/sqrt(N)", "measured ratio"))
first = None
for label, cam in (
    ("NightShot (1 frame)", {"nv_mode": 1.0, "nv_accumulate": 1.0}),
    ("Super NightShot, 2 frames", {"nv_mode": 2.0, "nv_accumulate": 2.0}),
    ("Super NightShot, 4 frames", {"nv_mode": 2.0, "nv_accumulate": 4.0}),
    ("Super NightShot, 8 frames", {"nv_mode": 2.0, "nv_accumulate": 8.0}),
    ("Super NightShot, 16 frames", {"nv_mode": 2.0, "nv_accumulate": 16.0}),
):
    a = render(cam)
    h = hf(a)
    n = int(label.split()[-2]) if "frames" in label else 1
    if first is None:
        first = h
    print("   %-30s %-10.5f %-11.5f %-11.4f %.4f"
          % (label, L(a), h, 1.0 / np.sqrt(n), h / max(1e-9, first)))
print()
print("   and it must fall, and fall by more as N grows -- a reading of 1.00 for every N would mean the noise")
print("   is being added AFTER the sum, where accumulation cannot touch it.")
