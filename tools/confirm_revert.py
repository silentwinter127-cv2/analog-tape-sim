"""Confirm the reverted state: the calibrated features work and the display's brightness fault is back as it was."""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from atsim.codecs import nightvision as nv                                    # noqa: E402

LUMA = np.array([0.299, 0.587, 0.114], np.float32)
print("   === the reverted display, on grey and coloured inputs ===")
for name, v in (("grey 0.4", [0.4, 0.4, 0.4]), ("blue-ish", [0.30, 0.35, 0.35]),
                ("strongly green", [0.15, 0.50, 0.20]),
                ("the photo's own", [0.3209, 0.2569, 0.4250])):
    v = np.array(v, np.float32)
    a = np.zeros((8, 8, 3), np.float32)
    a[:] = v
    o = nv.monochrome_green(a, nv.NightVisionSpec(green_mix=1.0))
    print("   %-18s sum-gain %.5f   luma-gain %.5f" % (name, o.mean() / v.mean(),
                                                       float((o @ LUMA).mean()) / float(v @ LUMA)))
print()
print("   **the sum-gain is not 1.0, which is the fault this round found and did not fix** -- the display's weights")
print("   are normalised so that `weights @ LUMA == 1`, while a picture's brightness is the SUM of the three")
print("   channels.  On a grey input the two coincide, which is why the property had appeared to hold.")

print()
print("   === and the calibrated features are intact ===")
d = np.stack([nv.read_noise(np.full((16, 16, 3), 0.4, np.float32),
                            nv.NightVisionSpec(read_noise=0.05, chroma_noise=0.15), seed=q) - 0.4
              for q in range(200)])
sd = d.std(axis=0).mean(axis=(0, 1))
print("   chroma noise: per-channel sigma %s, spread %.4f (the real frames read 1.27), sum %+.6f"
      % (np.round(sd, 5), sd.max() / sd.min(), float(d.mean())))
r = []
for res in (0.0, 0.35):
    a = np.zeros((8, 8, 3), np.float32)
    a[:] = (0.3209, 0.2569, 0.4250)
    o = nv.monochrome_green(a, nv.NightVisionSpec(green_mix=1.0, chroma_residue=res))
    r.append(float(o[..., 0].mean() / o.sum(axis=-1).mean()))
print("   chroma residue: red share %.4f at 0.0, %.4f at 0.35" % (r[0], r[1]))
print()
print("   the residue and the colour noise are the two things this session added and calibrated; both are present.")
print("   **the brightness normalisation is the open item**, with the measurement that identifies it recorded above.")
