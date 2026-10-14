"""
The shot noise on a flat field, where its effect cannot hide behind the picture's own structure.

**What the last measurement could not distinguish.**  On a real photograph the dark-to-bright temporal ratio was
0.281 with neither noise term, 0.448 with read noise alone, and **0.280 with shot noise alone -- a change of 0.001 where
the read term moved it 0.166**.  Two explanations fit that and the statistic cannot separate them:

  * **the term is too weak** -- one field across three channels, applied to luminance only, so it is diluted;
  * **or the metric cannot see it** -- the bright third of that frame carries real detail whose 3x3 residual is larger
    than any noise added, so a physically correct term barely moves the number.

**A flat field separates them**, because a flat field has no structure to confound the measurement: every variation in
it is noise, by construction.  **Two patches, one dark and one bright, both perfectly uniform**, so the temporal spread
in each is the noise and nothing else -- and the prediction is exact: **the shot term's standard deviation in the bright
patch should be `sqrt(bright/dark)` times its value in the dark patch**, which for a 0.10 against 0.60 pair is 2.45x.

**Measured with read noise held at zero**, so the only term is the one under test, and **against a control with it at
zero** so the structure-free baseline is known.
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from atsim.core.params import merge_values                                    # noqa: E402
from atsim.core.profiles import FACTORY_PRESETS, SCHEMA                       # noqa: E402
from atsim.render.renderer import Renderer                                    # noqa: E402

COEF = np.array([0.299, 0.587, 0.114], np.float32)
W, H = 320, 240
DARK, BRIGHT = 0.10, 0.60


class Flat:
    """Two perfectly uniform patches, so every variation in them is noise by construction."""

    height, width = H, W
    fps = 25.0
    frame_count = 400

    def frame(self, i: int) -> np.ndarray:
        a = np.zeros((H, W, 3), np.float32)
        a[:, : W // 2] = DARK
        a[:, W // 2:] = BRIGHT
        return a


BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])
OFF = {"halation": 0.0, "ccd_smear": 0.0, "flare_defocus": 0.0, "lens_flare": 0.0,
       "overflow_drain": 0.0, "chroma_aberration": 0.0, "sensor_noise": 0.0, "noise_grain": 0.0,
       "fixed_pattern": 0.0, "spike_spread": 0.0, "motion_blur": 0.0, "stuck_fraction": 0.0,
       "bloom_threshold": 0.0, "overflow_width": 0.0, "sharpness": 1.0}


def measure(**cam):
    """The temporal spread in each patch, over indices -- which for a flat field is exactly the noise."""
    layers = {"camera": dict(OFF, nv_green_mix=0.0, nv_illuminator=0.0, **cam),
              "format": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0},
              "player": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0},
              "codec": {"dv_enabled": 0.0}, "display": {"enabled": 0.0}}
    rr = Renderer(Flat(), W, H, 25.0, merge_values(BASE, layers), seed=7)
    st = np.stack([np.clip(np.asarray(r, np.float32), 0, 1) for _, r in rr.render_frames(list(range(80, 100)))])
    g = np.clip(st @ COEF, 0, 1)
    n = g.std(axis=0)
    # a margin away from the boundary so no edge effect enters either patch
    return float(n[:, 40:W // 2 - 40].mean()), float(n[:, W // 2 + 40:W - 40].mean())


print("   two uniform patches, dark %.2f and bright %.2f, so every variation in them is noise by construction."
      % (DARK, BRIGHT))
print("   the prediction: a shot term's standard deviation should go as sqrt(lum), so bright/dark = %.2f"
      % ((BRIGHT / DARK) ** 0.5))
print()
print("   === the two terms on a flat field ===")
print("   %-32s %-14s %-14s %-10s %s" % ("", "dark noise", "bright noise", "ratio", "vs the prediction"))
for tag, cam in (("neither", dict(nv_read_noise=0.0, nv_shot_noise=0.0)),
                 ("read 0.20 only", dict(nv_read_noise=0.20, nv_shot_noise=0.0)),
                 ("shot 0.10 only", dict(nv_read_noise=0.0, nv_shot_noise=0.10)),
                 ("shot 0.20 only", dict(nv_read_noise=0.0, nv_shot_noise=0.20)),
                 ("read 0.20 + shot 0.20", dict(nv_read_noise=0.20, nv_shot_noise=0.20))):
    d, b = measure(**cam)
    pred = (BRIGHT / DARK) ** 0.5 if "shot" in tag and "only" in tag else 1.0
    print("   %-32s %-14.5f %-14.5f %-10.3f %.3f" % (tag, d, b, b / max(1e-9, d), pred))
print()
print("   **a read-noise-only model gives a ratio of 1.000** -- the same absolute grain in both patches, which is")
print("   exactly what `note.txt` calls wrong: \"不要简单地对整张图叠加均匀颗粒；暗部和高光的表现应该不同\".")
print()
print("   and the check on the arithmetic itself: for shot noise alone at sigma, the dark patch's spread should be")
print("   sigma*sqrt(%.2f) = %.4f*sigma and the bright patch's sigma*sqrt(%.2f) = %.4f*sigma"
      % (DARK, DARK ** 0.5, BRIGHT, BRIGHT ** 0.5))
