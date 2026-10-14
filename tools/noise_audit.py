"""
Settle which of two contradictory measurements is wrong, by taking all four combinations in one run.

**The contradiction.**  On a real photograph, raising `read_noise` from 0 to 0.25 moved the dark-to-bright temporal
ratio from **0.281 to 0.448**.  On a synthetic flat field, **every noise setting gave bit-identical output** -- neither
`read_noise` nor `shot_noise` reached the picture at all.  **Two of my own measurements disagree, so at least one is
measuring something other than what it says**, and adding more mechanisms on top of a contradiction is wasted work.

**Four things differed between them, and this takes all four combinations rather than guessing which mattered:**

    the source       a photograph (structure)  vs  a flat field (no structure)
    the window       frames 60-79              vs  frames 80-99

**and each is a knob the run below turns.**  **The discriminator is whether a noise parameter changes the number at
all**: if it does not, that cell's measurement is blind to noise and its verdict means nothing -- whatever its figures
look like.  **So the table prints the spread at three noise settings side by side**, which is the difference the earlier
measurements never showed.

**The prediction, so it can fail:** the spread should rise in every cell as `read_noise` rises.  **Any cell where it does
not is the instrument's blind spot, and that cell is where the contradiction came from.**
"""
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

COEF = np.array([0.299, 0.587, 0.114], np.float32)
OBJ = (r"C:\Users\Administrator\.dsh\attachments\v1\objects")
PHOTO = os.path.join(OBJ, "69", "699c23315df037b01fa831a273364b41f99c28733fb1dc928b22d1e6c281818d")
W, H = 320, 240

_ph = np.asarray(Image.open(PHOTO).convert("RGB"), np.float32) / 255.0
PHOTO_SMALL = np.asarray(Image.fromarray((np.clip(_ph, 0, 1) * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS),
                         np.float32) / 255.0


class Still:
    """A photograph, so the picture has real structure and the *temporal* spread isolates the noise."""

    height, width = H, W
    fps = 25.0
    frame_count = 400

    def frame(self, i: int) -> np.ndarray:
        return PHOTO_SMALL


class Flat:
    """Two uniform patches, so every variation in them is noise by construction."""

    height, width = H, W
    fps = 25.0
    frame_count = 400

    def frame(self, i: int) -> np.ndarray:
        a = np.zeros((H, W, 3), np.float32)
        a[:, : W // 2] = 0.10
        a[:, W // 2:] = 0.60
        return a


class Flicker:
    """Spectrally flat noise per frame, so the frames genuinely differ but the picture has no structure."""

    height, width = H, W
    fps = 25.0
    frame_count = 400

    def frame(self, i: int) -> np.ndarray:
        rng = np.random.default_rng(1000 + int(i))
        return rng.random((H, W, 3)).astype(np.float32) * 0.0 + 0.35


BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])
OFF = {"halation": 0.0, "ccd_smear": 0.0, "flare_defocus": 0.0, "lens_flare": 0.0,
       "overflow_drain": 0.0, "chroma_aberration": 0.0, "sensor_noise": 0.0, "noise_grain": 0.0,
       "fixed_pattern": 0.0, "spike_spread": 0.0, "motion_blur": 0.0, "stuck_fraction": 0.0,
       "bloom_threshold": 0.0, "overflow_width": 0.0, "sharpness": 1.0}


def spread(src, indices, **cam):
    layers = {"camera": dict(OFF, nv_mode=1.0, nv_green_mix=0.0, nv_illuminator=0.0, **cam),
              "format": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0},
              "player": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0},
              "codec": {"dv_enabled": 0.0}, "display": {"enabled": 0.0}}
    rr = Renderer(src, W, H, 25.0, merge_values(BASE, layers), seed=7)
    st = np.stack([np.clip(np.asarray(r, np.float32), 0, 1) for _, r in rr.render_frames(list(indices))])
    g = np.clip(st @ COEF, 0, 1)
    return float(g.std(axis=0).mean())


SETTINGS = [("neither", dict(nv_read_noise=0.0, nv_shot_noise=0.0)),
            ("read 0.15", dict(nv_read_noise=0.15, nv_shot_noise=0.0)),
            ("read 0.30", dict(nv_read_noise=0.30, nv_shot_noise=0.0))]

print("   the discriminator: does a NOISE PARAMETER change the number?  A cell where it does not is blind to noise,")
print("   and its verdict means nothing however its figures look.")
print()
print("   === every combination of source and frame window ===")
print("   %-26s %-14s %-14s %-14s %s" % ("source / window", "neither", "read 0.15", "read 0.30", "responds?"))
BLIND = []
for srcname, src in (("photograph", Still()), ("flat patches", Flat()), ("flat noise field", Flicker())):
    for wname, idx in (("60-79", range(60, 80)), ("80-99", range(80, 100))):
        vals = [spread(src, idx, **cam) for _, cam in SETTINGS]
        responds = max(vals) - min(vals) > 1e-5
        if not responds:
            BLIND.append((srcname, wname))
        print("   %-26s %-14.5f %-14.5f %-14.5f %s"
              % ("%s / %s" % (srcname, wname), vals[0], vals[1], vals[2],
                 "yes" if responds else "NO -- blind"))
print()
if BLIND:
    print("   === the blind cells, which is where the contradiction came from ===")
    for s, w in BLIND:
        print("      %s at %s: the noise parameters do nothing, so a verdict taken there is about something else."
              % (s, w))
else:
    print("   every cell responds, so both earlier measurements were sensitive and the difference lies elsewhere.")
print()
print("   === and the same for the shot term, on the cells that do respond ===")
for srcname, src in (("photograph", Still()), ("flat patches", Flat())):
    vals = [spread(src, range(60, 80), nv_read_noise=0.0, nv_shot_noise=s) for s in (0.0, 0.15, 0.30)]
    print("   %-26s shot 0.00 %.5f   shot 0.15 %.5f   shot 0.30 %.5f   %s"
          % (srcname, vals[0], vals[1], vals[2], "responds" if max(vals) - min(vals) > 1e-5 else "NO"))
