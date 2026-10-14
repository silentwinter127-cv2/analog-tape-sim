"""Probe `_roll_off` from inside, to see the white point and the saturation stops it actually runs with."""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from atsim.core.params import merge_values                                    # noqa: E402
from atsim.core.profiles import FACTORY_PRESETS, SCHEMA                       # noqa: E402
from atsim.render.renderer import Renderer                                    # noqa: E402
from atsim.stages.camera import CameraStage                                   # noqa: E402

COEF = np.array([0.299, 0.587, 0.114], np.float32)
W, H = 320, 240
_a = np.asarray(Image.open(os.path.join(r"C:\Users\Administrator\.dsh\attachments\v1\objects", "69",
                                        "699c23315df037b01fa831a273364b41f99c28733fb1dc928b22d1e6c281818d")
                             ).convert("RGB"), np.float32) / 255.0
SRC = np.asarray(Image.fromarray((np.clip(_a, 0, 1) * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS),
                 np.float32) / 255.0


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

seen = []
o = CameraStage._roll_off


def probe(self, rgb):
    out = o(self, rgb)
    dr = float(np.clip(self.dynamic_range, 3.0, 16.0))
    sat = float(getattr(self, "saturation_stops", 0.0))
    w = float(2.0 ** ((dr - 8.0) / 8.0)) * (float(2.0 ** (-sat)) if sat > 1e-4 else 1.0)
    seen.append((sat, w, float(np.asarray(rgb, np.float32).max()),
                 float(np.asarray(out, np.float32).max()),
                 float((np.asarray(out, np.float32) >= 0.999).mean())))
    return out


CameraStage._roll_off = probe

print("   the values `_roll_off` actually runs with, at several settings:")
print("   %-12s %-10s %-12s %-12s %-12s %s" % ("configured", "attr", "white", "input max", "output max", "clipped"))
for v in (0.0, 0.3, 0.6, 0.9):
    seen.clear()
    layers = {"camera": dict(OFF, nv_mode=1.0, saturation_stops=v),
              "format": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0},
              "player": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0},
              "codec": {"dv_enabled": 0.0}, "display": {"enabled": 0.0}}
    rr = Renderer(One(), W, H, 25.0, merge_values(BASE, layers), seed=1)
    rr.render_frame(0)
    if seen:
        sat, w, imx, omx, cl = seen[-1]
        print("   %-12.2f %-10.2f %-12.4f %-12.4f %-12.4f %.5f" % (v, sat, w, imx, omx, cl))
    else:
        print("   %-12.2f %-10s (never called)" % (v, "-"))

CameraStage._roll_off = o
print()
print("   **if the attr column does not follow the configured column, the value is lost between the config and the")
print("   stage; if the white column does not follow the attr column, the calculation is not using it; and if the")
print("   output max does not follow the white, the curve is not clipping where it should.**")
