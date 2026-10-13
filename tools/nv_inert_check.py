"""
Check whether `nv_green_gain` and `nv_read_noise` are actually read, or inert like so many before them.

**Why this is checked rather than assumed.** This project has found the "declared, reachable, and never read" fault more
than a dozen times -- `agc_range`, `chroma_under_luma`, `player.line_delay`, `tape.row_chroma_*`,
`player.chroma_vert_blend`, and in this very work `nv_*` before it reached the schema. Both `nv_green_gain` and
`nv_read_noise` were declared in the night-vision spec and **neither is referenced anywhere in the module or the
stage**, so they are almost certainly inert -- and the way to establish it is to set each to its extremes and see
whether the picture moves, not to read the code and conclude.

**The test is the same shape that has worked throughout**: vary the parameter alone, over its full declared range, and
compare the outputs. A parameter whose whole range changes nothing is a fault to record, and one that changes the
picture is a mechanism that needs calibrating.
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
pic = np.asarray(Image.open(os.path.join(ROOT, "tools", "ntscrs_ref", "newscene.png"))
                 .convert("RGB"), np.float32) / 255.0
H, W = pic.shape[:2]
BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])


class StaticSource:
    def __init__(self, p):
        self.height, self.width = p.shape[:2]
        self.fps, self.frame_count = 25.0, 20
        self._p = p

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


def render(cam):
    layers = {"camera": dict(OFF, **cam),
              "format": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0},
              "player": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0},
              "codec": {"dv_enabled": 1.0}, "tape": {"head_switching": 0.0},
              "display": dict(RASTER_OFF, enabled=1.0, gamma=1.0, saturation=0.62)}
    p = merge_values(BASE, layers)
    r = Renderer(StaticSource(pic), W, H, 25.0, p, seed=20240101)
    return np.clip(np.asarray(r.render_frame(14), np.float32), 0, 1)


base = render({"nv_mode": 1.0})
print("   %-30s %-10s %-10s %-10s %s" % ("parameter", "mean", "G share", "hf", "changes the picture?"))
for label, cam in (
    ("nv_green_gain = 0.5 (min)", {"nv_mode": 1.0, "nv_green_gain": 0.5}),
    ("nv_green_gain = 1.6 (max)", {"nv_mode": 1.0, "nv_green_gain": 1.6}),
    ("nv_read_noise = 0.0", {"nv_mode": 1.0, "nv_read_noise": 0.0}),
    ("nv_read_noise = 0.2 (max)", {"nv_mode": 1.0, "nv_read_noise": 0.2}),
    ("nv_beam_x = 0.1", {"nv_mode": 1.0, "nv_beam_x": 0.1}),
    ("nv_beam_y = 0.2", {"nv_mode": 1.0, "nv_beam_y": 0.2}),
    ("nv_beam_angle = 2.0", {"nv_mode": 1.0, "nv_beam_angle": 2.0}),
    ("nv_ir_response = 0.0", {"nv_mode": 1.0, "nv_ir_response": 0.0}),
    ("nv_illuminator_visible = 0.25", {"nv_mode": 1.0, "nv_illuminator_visible": 0.25}),
):
    a = render(cam)
    tot = max(1e-9, float(a.sum(axis=-1).mean()))
    m = a.reshape(-1, 3).mean(axis=0) / tot
    same = np.allclose(a, base, atol=1e-6)
    print("   %-30s %-10.5f %-10.4f %-10.5f %s"
          % (label, L(a), m[1], float(np.abs(np.diff(a @ COEF, axis=1)).mean()),
             "NO -- identical frame" if same else "yes"))
print()
print("   for reference, the mode's own default output: mean %.5f" % L(base))
