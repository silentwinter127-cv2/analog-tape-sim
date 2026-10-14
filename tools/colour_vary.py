"""
Measure how the footage's colour VARIES across the frame, since the model's green is uniform and the footage's is not.

**What has been measured before, and why it did not catch this.**  The per-channel mean absolute chroma -- `[0.0479
0.0244 0.0213]` for NIGHTSHOT 0 -- and its ratios, which the green-gain calibration was fitted to.  **That is an average
over the whole frame, so it says nothing about whether the colour is the same everywhere.**  A picture with a single
uniform green cast and a picture whose green varies from region to region both give one channel triple.

**And the model's green IS uniform by construction**: `monochrome_green` projects the frame onto fixed weights, so every
pixel's hue is the same and only its brightness differs -- plus a fraction of the picture's own chroma through
`chroma_residue`.  **The footage has no such constraint.**  A camcorder's NightShot output with the IR-cut filter removed
is not a fixed projection: the sensor's response varies with wavelength, the illuminator fades with distance, and the
viewfinder's phosphor or LCD has its own behaviour -- **so the real frames' hue should drift across the picture, and the
model's should not.**

**Two quantities separate them:**

  * **the hue's spread across the frame** -- the standard deviation of the angle of the (r-g, b-g) chroma vector, which
    is zero for a rigid projection and non-zero for the footage;
  * **the correlation between hue and position** -- whether the drift has a spatial structure (towards the lamp, towards
    a corner) rather than being random noise, which is what would make it a mechanism rather than grain.

**And the same is measured for saturation**, since "not uniformly green" could mean either the hue drifting or the
strength of the cast varying.
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
NS = [("NIGHTSHOT 0", os.path.join(OBJ, "ed",
                                   "ed906503e82e76777332850f677a16db3ea89089fde1c313576edf9fce66f4e0")),
      ("NIGHTSHOT 1", os.path.join(OBJ, "f2",
                                   "f2dd7803fd91d124a7c16be3c6f97df679c8a83d89631a462e9010f13a3e7c5e")),
      ("NIGHTSHOT 2", os.path.join(OBJ, "b9",
                                   "b94a0e8402a11788032bd0af9e81de7fc4713d3a648384f2c9ff57b21c21f4b6"))]
PHOTO = os.path.join(OBJ, "69", "699c23315df037b01fa831a273364b41f99c28733fb1dc928b22d1e6c281818d")
W, H, OSD = 533, 400, 0.22
sl = slice(int(H * OSD), H)


def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0
    return np.asarray(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS),
                      np.float32) / 255.0


def colour_stats(a, tag):
    """
    The hue's spread and its spatial structure.

    The chroma vector is `(r - g, b - g)`, whose angle is the hue offset from green.  **A rigid projection onto fixed
    weights gives every pixel the same angle**, so the spread is the direct measure of "is the green uniform".
    """
    sc = np.clip(np.asarray(a, np.float32), 0, 1)[sl]
    r, g, b = sc[..., 0], sc[..., 1], sc[..., 2]
    u, v = r - g, b - g
    mag = np.sqrt(u * u + v * v)
    ang = np.arctan2(v, u)
    #: only where there IS chroma, so near-neutral pixels do not dominate the angle statistics
    m = mag > max(1e-4, float(np.quantile(mag, 0.5)))
    spread = float(np.angle(np.exp(1j * (ang[m] - ang[m].mean()))).std()) if m.any() else 0.0
    sat = float(mag[m].mean()) if m.any() else 0.0
    #: and does the hue DRIFT with position, rather than being noise?
    h, w = ang.shape
    yy, xx = np.mgrid[0:h, 0:w]
    dy = float(np.corrcoef(yy[m], ang[m])[0, 1]) if m.sum() > 8 else 0.0
    dx = float(np.corrcoef(xx[m], ang[m])[0, 1]) if m.sum() > 8 else 0.0
    return spread, sat, dx, dy, ang, m


print("   === the hue's spread across the frame, and whether it drifts with position ===")
print("   %-24s %-12s %-12s %-10s %-10s %s" % ("", "hue spread", "sat", "corr x", "corr y", "reading"))
for name, p in NS:
    sp, sa, dx, dy, _, _ = colour_stats(load(p), name)
    print("   %-24s %-12.4f %-12.4f %+-10.3f %+-10.3f %s"
          % (name, sp, sa, dx, dy, "the hue varies" if sp > 0.05 else "uniform"))
sp, sa, dx, dy, _, _ = colour_stats(load(PHOTO), "photo")
print("   %-24s %-12.4f %-12.4f %+-10.3f %+-10.3f" % ("the source photograph", sp, sa, dx, dy))


class One:
    height, width = H, W
    fps = 25.0
    frame_count = 1

    def frame(self, i):
        return load(PHOTO)


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


print()
print("   === the model, at three settings ===")
for tag, mode, cam in (("mode 1 (green_mix 1)", 1.0, {}),
                       ("mode 1, residue 1 (chroma kept)", 1.0, dict(nv_chroma_residue=1.0)),
                       ("mode 1, residue 0 (rigid projection)", 1.0, dict(nv_chroma_residue=0.0)),
                       ("mode 2 (Super)", 2.0, {})):
    a = run(mode, **cam)
    sp, sa, dx, dy, _, _ = colour_stats(a, tag)
    print("   %-34s %-12.4f %-12.4f %+-10.3f %+-10.3f" % (tag, sp, sa, dx, dy))
print()
print("   a rigid projection gives a hue spread near ZERO; the footage's is the target.")
