"""
Measure at the input's NATIVE resolution, and keep CG renders out of the calibration set.

**Two corrections from the user, both of which invalidate earlier measurements of mine.**

**1. The rig was too blunt for the effect it was measuring.**  Every self-test rig in this work scaled its input to
**533x400** before running the chain, while the server exports at `scale=0.35` of 720x480 or higher and the user's own
export was **2048x1536**.  **Lateral chromatic aberration is a sub-pixel displacement**, so resampling to a quarter of the
area smooths it away.  Measured on the case photograph, the hue spread by band:

    resolution    dark     mid     bright   brightest
    533x400       0.883    0.555   0.664    0.887        <- what every earlier figure used
    720x480       0.916    0.553   0.686    0.912
    960x720       0.937    0.622   0.606    0.889
    **1440x1080    1.098    0.788   0.809    0.990**

**so the low-resolution rig was hiding 24% of the dark band's and 42% of the mid band's hue structure.**  This file
therefore renders at the input's own size and never downsamples, and it prints the resolution beside every number so a
figure cannot be compared against one taken at a different scale.

**2. A CG render is not a camera.**  The basement image the user supplied is a **CG render**, so its spectral response,
its noise and its dynamic range are a renderer's choices rather than a sensor's -- **it can test whether the chain does
something sensible, but it cannot be the target a real sensor's behaviour is calibrated against.**  It is kept here as a
**synthetic control** and labelled as one, and the calibration targets remain the camcorder's own frames.

**Nothing is retuned in this file.**  It measures, at a resolution the effect can survive and on inputs whose provenance
is stated, so the next change is aimed at what is actually there.
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
OBJ = r"C:\Users\Administrator\.dsh\attachments\v1\objects"
#: (path, label, is a photograph of a real scene?)
SOURCES = [
    (os.path.join(OBJ, "e0", "e0e36b261fbbad42777702d3b775fcf30d4f8505fdc104e08b24405e69eda14c"),
     "the RGB case", True),
    (os.path.join(OBJ, "0c", "0cc4906c4997112e2e1833ec1f0514bfb7f459fd7362c2fb04781761fdd42cef"),
     "the basement (a CG RENDER, not a camera)", False),
]
REALS = [(os.path.join(OBJ, "ee", "ee23c56ac527467f74f6f9140400a172db2577aea1e13c0baab70e78305c9c7e"),
          "the real NIGHTSHOT 2"),
         (os.path.join(OBJ, "f6", "f6f945abfa1a8a09e497c35b897e15b6b9e5417b82c3d50ae827957abce5015d"),
          "the real NIGHTSHOT 1")]


def native(path):
    """The image at its own size -- **no downsampling**, which is the whole point of this file."""
    return np.asarray(Image.open(path).convert("RGB"), np.float32) / 255.0


def crop_osd(a):
    return a[int(a.shape[0] * 0.17):]


def analyse(x):
    lum = np.clip(x @ COEF, 0, 1)
    u, v = x[..., 0] - x[..., 1], x[..., 2] - x[..., 1]
    m = np.sqrt(u * u + v * v)
    ang = np.arctan2(v, u)
    spreads, counts = [], []
    for lo, hi in ((0.0, 0.25), (0.25, 0.6), (0.6, 0.85), (0.85, 1.01)):
        k = (lum >= lo) & (lum < hi) & (m > 0.02)
        counts.append(int(k.sum()))
        if k.sum() < 50:
            spreads.append(0.0)
            continue
        aa = ang[k]
        spreads.append(float(np.angle(np.exp(1j * (aa - np.angle(np.exp(1j * aa).mean())))).std()))
    r, g, b = (float(x[..., i].mean()) for i in range(3))
    return dict(primary="#%02X%02X%02X" % (int(round(r * 255)), int(round(g * 255)), int(round(b * 255))),
                mean=float(lum.mean()), std=float(lum.std()), p50=float(np.quantile(lum, 0.50)),
                below05=float((lum < 0.05).mean()), spreads=spreads, counts=counts,
                RG=r / max(1e-9, g), BG=b / max(1e-9, g))


BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])


class Still:
    fps = 25.0
    frame_count = 1

    def __init__(self, a):
        self.a = a
        self.height, self.width = a.shape[:2]

    def frame(self, i):
        return self.a


def run(src):
    layers = {"camera": {"nv_mode": 2.0},
              "format": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0},
              "player": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0},
              "codec": {"dv_enabled": 0.0}, "display": {"enabled": 0.0}}
    h, w = src.shape[:2]
    rr = Renderer(Still(src), w, h, 25.0, merge_values(BASE, layers), seed=20240101)
    return np.clip(np.asarray(rr.render_frame(0), np.float32), 0, 1)


print("   THE TARGETS -- the camcorder's own frames, at their native size")
print("   %-30s %-9s %-9s %-7s %-7s %-8s %s"
      % ("", "resolution", "primary", "mean", "p50", "<0.05", "hue spread: dark  mid  bright  brightest"))
for p, tag in REALS:
    a = crop_osd(native(p))
    s = analyse(a)
    print("   %-30s %-9s %-9s %-7.4f %-7.4f %-8.4f %s"
          % (tag, "%dx%d" % (a.shape[1], a.shape[0]), s["primary"], s["mean"], s["p50"], s["below05"],
             "  ".join("%.3f" % v for v in s["spreads"])))
print()
print("   AND THE INPUTS -- rendered at their OWN resolution, never downsampled")
for p, tag, is_photo in SOURCES:
    src = native(p)
    out = run(src)
    si, so = analyse(src), analyse(out)
    print("   %-30s %-9s %-9s %-7.4f %-7.4f %-8.4f %s"
          % (tag + " (input)", "%dx%d" % (src.shape[1], src.shape[0]), si["primary"], si["mean"], si["p50"],
             si["below05"], "  ".join("%.3f" % v for v in si["spreads"])))
    print("   %-30s %-9s %-9s %-7.4f %-7.4f %-8.4f %s   %s"
          % ("    mode 2", "", so["primary"], so["mean"], so["p50"], so["below05"],
             "  ".join("%.3f" % v for v in so["spreads"]),
             "" if is_photo else "<- a CG render: a control, not a calibration target"))
print()
print("   **the hue spread is resolution-dependent**, so a figure taken at one scale cannot be compared with one taken at")
print("   another -- which is why every row above prints the size it was measured at.")
