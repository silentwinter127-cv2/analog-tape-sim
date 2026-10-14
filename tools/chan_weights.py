"""
Reproduce the chroma channel-weight fault, and read the weights that cause it.

**The measurement that named it.**  Per-channel mean absolute chroma on the reference frames against the model's, both
at 533x400 below the OSD band:

    the real NIGHTSHOT 0    [0.0479  0.0244  0.0213]
    the model               [0.0728  0.0240  0.0955]
    ratio                    1.52x   1.00x   **4.48x**

**so the total is roughly right and the DISTRIBUTION is wrong**, and specifically the **green-to-blue ratio is inverted**:
the real frame's green exceeds its blue by 1.15x, the model's blue exceeds its green by 4x.  **And that is a target
recovered from the footage itself**, which is why this is the one item of the note's six that can be checked rather than
argued about.

**Where a channel ratio comes from.**  `monochrome_green` projects the picture through green-display weights, and the
weights are normalised -- by their sum, in the version that measured a 1.0000 luminance gain.  **But a mono display's
weights set the R:G:B ratio of everything that survives it**, including the chroma residue that
`chroma_residue` deliberately passes through, so a weight set that is right for luminance can still be wrong for
chroma.  The two are separate constraints and the second was never checked against the footage.

**So the weights and the chroma they produce are measured together here**, on the real frame and the model, so the
arithmetic that has to change is identified rather than guessed.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from atsim.codecs import nightvision as NV                                    # noqa: E402
import inspect                                                                # noqa: E402

COEF = np.array([0.299, 0.587, 0.114], np.float32)
OBJ = (r"C:\Users\Administrator\.dsh\attachments\v1\objects")
REFS = [("NIGHTSHOT 0", os.path.join(OBJ, "ed",
                                     "ed906503e82e76777332850f677a16db3ea89089fde1c313576edf9fce66f4e0")),
        ("NIGHTSHOT 1", os.path.join(OBJ, "f2",
                                     "f2dd7803fd91d124a7c16be3c6f97df679c8a83d89631a462e9010f13a3e7c5e")),
        ("NIGHTSHOT 2", os.path.join(OBJ, "b9",
                                     "b94a0e8402a11788032bd0af9e81de7fc4713d3a648384f2c9ff57b21c21f4b6"))]
W, H, OSD = 533, 400, 0.22
sl = slice(int(H * OSD), H)


def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0
    return np.asarray(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS),
                      np.float32) / 255.0


def chroma(a):
    """Mean absolute per-channel departure from the pixel's own luminance -- what `chroma` means in this project."""
    sc = np.asarray(a, np.float32)[sl]
    g = np.clip(sc @ COEF, 0, 1)
    return np.abs(sc - g[..., None]).mean(axis=(0, 1))


print("   === the target, from the footage ===")
print("   %-16s %-28s %-10s %s" % ("", "|chroma| per channel", "G/B", "R/G"))
targets = {}
for name, p in REFS:
    c = chroma(load(p))
    targets[name] = c
    print("   %-16s %-28s %-10.3f %.3f" % (name, np.round(c, 4), c[1] / c[2], c[0] / c[1]))

print()
print("   === and the weights the module uses ===")
src = inspect.getsource(NV.monochrome_green)
inside = False
for line in src.split("\n"):
    n = line.count('"""')
    if n:
        if inside:
            inside = False
        elif n == 1:
            inside = True
        continue
    if not inside and line.strip() and not line.strip().startswith("#"):
        print("      %s" % line[:106])

print()
print("   === what those weights do to a flat white field, per channel ===")
spec = NV.NightVisionSpec(mode=1.0, green_mix=1.0, green_gain=1.0, chroma_residue=0.0,
                          chroma_noise=0.0, read_noise=0.0)
white = np.ones((8, 8, 3), np.float32) * 0.6
out = np.clip(np.asarray(NV.monochrome_green(white, spec), np.float32), 0, 1)
print("      a grey 0.6 through the projection: %s  (ratio R:G:B = %s)"
      % (np.round(out.mean(axis=(0, 1)), 4),
         np.round(out.mean(axis=(0, 1)) / max(1e-9, float(out.mean(axis=(0, 1))[1])), 3)))
print()
print("   **the real frames' own G/B is %.3f; if the projection gives a different ratio then everything surviving**"
      % (np.mean([c[1] / c[2] for c in targets.values()])))
print("   **it -- including the chroma residue -- carries that ratio, which is the fault being chased.**")
