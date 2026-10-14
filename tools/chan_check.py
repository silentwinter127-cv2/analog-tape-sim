"""
The end-to-end check the calibration was made for: does the model's chroma now match the footage per channel?

**This is the one item of the note's mechanisms with a target recovered from the footage rather than argued about**, so
it is also the one that can be *verified* rather than merely demonstrated.  The target, and what the model produced
before the calibration:

    the real NIGHTSHOT 0    [0.0479  0.0244  0.0213]      G/B 1.150   R/G 1.962
    the model, green_gain 1.0  [0.0728  0.0240  0.0955]   -- blue 4.48x, green 1.00x

**and the calibration moved `nv_green_gain` to 1.15**, solved against the three segments' own ratios with the luminance
gain held at exactly 1.0.  **A calibration that is not re-measured end to end is an assertion**, so the same statistic is
taken again here, on the whole chain, at the schema's own settings.
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
NS1 = os.path.join(OBJ, "f2", "f2dd7803fd91d124a7c16be3c6f97df679c8a83d89631a462e9010f13a3e7c5e")
NS2 = os.path.join(OBJ, "b9", "b94a0e8402a11788032bd0af9e81de7fc4713d3a648384f2c9ff57b21c21f4b6")
PHOTO = os.path.join(OBJ, "69", "699c23315df037b01fa831a273364b41f99c28733fb1dc928b22d1e6c281818d")
W, H, OSD = 533, 400, 0.22
sl = slice(int(H * OSD), H)


def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0
    return np.asarray(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS),
                      np.float32) / 255.0


def chan(a):
    sc = np.asarray(a, np.float32)[sl]
    g = np.clip(sc @ COEF, 0, 1)
    return np.abs(sc - g[..., None]).mean(axis=(0, 1))


src = load(PHOTO)


class One:
    height, width = H, W
    fps = 25.0
    frame_count = 1

    def frame(self, i):
        return src


BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])
OFF = {"halation": 0.0, "ccd_smear": 0.0, "flare_defocus": 0.0, "lens_flare": 0.0,
       "overflow_drain": 0.0, "chroma_aberration": 0.0, "sensor_noise": 0.0, "noise_grain": 0.0,
       "fixed_pattern": 0.0, "spike_spread": 0.0, "motion_blur": 0.0, "stuck_fraction": 0.0,
       "bloom_threshold": 0.0, "overflow_width": 0.0, "sharpness": 1.0}


def run(mode, **cam):
    layers = {"camera": dict(OFF, nv_mode=mode, **cam),
              "format": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0},
              "player": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0},
              "codec": {"dv_enabled": 0.0}, "display": {"enabled": 0.0}}
    rr = Renderer(One(), W, H, 25.0, merge_values(BASE, layers), seed=20240101)
    return np.clip(np.asarray(rr.render_frame(0), np.float32), 0, 1)


print("   === per-channel chroma, model against footage, at the schema's own settings ===")
print("   %-26s %-28s %-9s %-9s %s" % ("", "|chroma| per channel", "G/B", "R/G", "vs the footage"))
rows = [("the real NIGHTSHOT 0", load(NS0), None),
        ("the real NIGHTSHOT 1", load(NS1), None),
        ("the real NIGHTSHOT 2", load(NS2), None)]
for tag, a, _ in rows:
    c = chan(a)
    print("   %-26s %-28s %-9.3f %-9.3f" % (tag, np.round(c, 4), c[1] / c[2], c[0] / c[1]))

print()
#: **Both gains are measured at the SAME mode.**  The first version of this figure took the 1.0 reading at mode 2 and
#: compared it with a mode-1 seam measurement, and the two are not comparable because the modes carry different
#: `chroma_residue` values (0.90 against 0.35) -- the same cross-configuration fault this work has recorded twice
#: before.  Mode 1 matches the seam trace, so both are taken there.
for gain in (1.0, 1.15):
    a = run(1.0, nv_green_gain=gain)
    c = chan(a)
    tgt = chan(load(NS0))
    err = float(np.abs(c / max(1e-9, c[1]) - tgt / max(1e-9, tgt[1])).mean())
    print("   %-26s %-28s %-9.3f %-9.3f mean ratio error %.3f"
          % ("the model, green_gain %.2f" % gain, np.round(c, 4), c[1] / c[2], c[0] / c[1], err))
print()
print("   mode 1 (`chroma_residue` 0.35), the same configuration as the seam trace, so the two are comparable.")
print("   the target's own spread across the three segments is G/B 1.150 to 3.714, so a fit inside that range is")
print("   as close as this footage can pin the weights down.")
