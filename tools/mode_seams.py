"""
Find which part of the night-vision scene stage costs 22.5% of the contrast, by probing its own seams.

**What is already measured on a real photograph, at one size and one configuration:**

```
source photograph                 contrast 0.1894
real NIGHTSHOT 0                  0.2032        the footage is HIGHER than its own source
the chain, mode off               0.1678        the chain costs 11.4%
the chain, mode on, display only  0.1674        the green display costs nothing
the chain, mode on, everything    0.1297        the mode's scene stage costs 22.5%
```

**so the remaining loss is inside `_night_vision`'s scene branch**, and the branch is three statements:

    accumulate(...)        the frame summing
    add_illuminator(...)   the lamp's light
    ir_highlights(...)     the near-infrared response

**Each is applied to a picture whose contrast is known, so each can be measured on its own** -- which is the seam method
that found the inversion after four scalar measurements had failed to.  **And the expectation is stated so it can be
falsified**: the mode raises the level by about 19% on this subject (0.2346 against 0.1973) and the footage's contrast is
slightly *above* its source's, so no single term should cost a large fraction of the spread.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from atsim.codecs import nightvision as nv                                    # noqa: E402

COEF = np.array([0.299, 0.587, 0.114], np.float32)
OBJ = (r"C:\Users\Administrator\.dsh\attachments\v1\objects")
NS0 = os.path.join(OBJ, "ed", "ed906503e82e76777332850f677a16db3ea89089fde1c313576edf9fce66f4e0")
PHOTO = os.path.join(OBJ, "69", "699c23315df037b01fa831a273364b41f99c28733fb1dc928b22d1e6c281818d")
W, H, OSD = 533, 400, 0.22
sl = slice(int(H * OSD), H)


def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0
    return np.asarray(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS),
                      np.float32) / 255.0


src = load(PHOTO)
real = load(NS0)
L0 = float(np.clip(src[sl] @ COEF, 0, 1).std())
print("   the source's own contrast %.4f;  the footage's %.4f (%.3f of it)"
      % (L0, float(np.clip(real[sl] @ COEF, 0, 1).std()),
         float(np.clip(real[sl] @ COEF, 0, 1).std()) / L0))

#: the spec the chain's defaults produce for NightShot
spec = nv.NightVisionSpec(mode=1.0, illuminator=0.15, ir_response=1.0, ir_albedo_mix=0.7,
                          green_mix=1.0, accumulate=1.0, read_noise=0.0, chroma_noise=0.15,
                          chroma_residue=0.35, beam_x=0.5, beam_y=0.5, beam_angle=0.55)


def stat(tag, a):
    a = np.clip(np.asarray(a, np.float32), 0, 1)
    l = np.clip(a[sl] @ COEF, 0, 1)
    print("   %-40s luma %-9.4f contrast %-9.4f  (%.3f of the source)"
          % (tag, float(l.mean()), float(l.std()), float(l.std()) / L0))
    return float(l.std())


print()
print("   === each statement of the scene branch, applied alone ===")
c = stat("the source", src)
c = stat("add_illuminator alone", nv.add_illuminator(src, spec))
c = stat("ir_highlights alone", nv.ir_highlights(src, spec))
c = stat("monochrome_green alone (the display)", nv.monochrome_green(src, spec))

print()
print("   === and in the order the stage runs them ===")
a = src
for tag, fn in (("add_illuminator", nv.add_illuminator),
                ("ir_highlights", nv.ir_highlights)):
    a = np.clip(np.asarray(fn(a, spec), np.float32), 0, 1)
    stat("after %s" % tag, a)
before = float(np.clip(a[sl] @ COEF, 0, 1).std())
a = np.clip(np.asarray(nv.monochrome_green(a, spec), np.float32), 0, 1)
after = stat("after monochrome_green (the whole mode)", a)
print()
print("   the display alone costs %.4f (%.1f%%)" % (before - after, 100 * (before - after) / max(1e-9, before)))

print()
print("   === the level, which is what the mode is meant to change ===")
print("      the source's luma %.4f; the footage's %.4f" % (float(np.clip(src[sl] @ COEF, 0, 1).mean()),
                                                           float(np.clip(real[sl] @ COEF, 0, 1).mean())))
