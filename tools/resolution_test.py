"""
Settle the resolution question first, because it is measurable and it contaminates every figure taken so far.

**The suspicion is reasonable and has never been checked.**  Every comparison in this series rendered the model at
**533x400** -- chosen early because that is the size the OSD band was measured at -- while the source photograph is
**2364x1773** and the NIGHTSHOT frames are whatever the clip frames were.  Each was resized to 533x400 with LANCZOS
before being measured.  **So "the model lacks detail" may be a statement about a downsample rather than about the
model**: a photograph taken to 533x400 and an HD frame taken to 533x400 do not carry the same detail, and neither does a
render made *at* 533x400 compared with one made at full size and then reduced.

**The three quantities that decide it:**

  * **the real sizes** of every reference, read from the files rather than assumed;
  * **the detail each carries at 533x400**, measured as the high-frequency energy that survives the reduction -- **if
    the photograph and the footage differ there, the comparison has been unfair to one of them**;
  * **and the model rendered at the references' own size** rather than at 533x400, since a renderer that synthesises a
    raster should be asked at the raster the comparison uses.

**Nothing is changed by this measurement**; it says whether the earlier ones were taken on comparable inputs.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

COEF = np.array([0.299, 0.587, 0.114], np.float32)
OBJ = (r"C:\Users\Administrator\.dsh\attachments\v1\objects")
REFS = [("NIGHTSHOT 0", os.path.join(OBJ, "ed",
                                     "ed906503e82e76777332850f677a16db3ea89089fde1c313576edf9fce66f4e0")),
        ("NIGHTSHOT 1", os.path.join(OBJ, "f2",
                                     "f2dd7803fd91d124a7c16be3c6f97df679c8a83d89631a462e9010f13a3e7c5e")),
        ("NIGHTSHOT 2", os.path.join(OBJ, "b9",
                                     "b94a0e8402a11788032bd0af9e81de7fc4713d3a648384f2c9ff57b21c21f4b6")),
        ("the photograph", os.path.join(OBJ, "69",
                                        "699c23315df037b01fa831a273364b41f99c28733fb1dc928b22d1e6c281818d"))]

print("   === the references' own sizes, read from the files ===")
print("   %-18s %-14s %-10s %s" % ("", "pixels", "aspect", "megapixels"))
for name, p in REFS:
    with Image.open(p) as im:
        w, h = im.size
        fmt = im.format
    print("   %-18s %-14s %-10.4f %s" % (name, "%dx%d" % (w, h), w / h, "%.2f MP  (%s)" % (w * h / 1e6, fmt)))


def hf_energy(a):
    """The energy in the picture's fine detail: the spread of a one-pixel high-pass, relative to the picture's spread."""
    g = np.clip(np.asarray(a, np.float32) @ COEF, 0, 1)
    d = np.abs(np.diff(g, axis=1))
    return float(d.mean()) / max(1e-9, float(g.std()))


print()
print("   === the detail each one carries, at the size every measurement used (533x400) ===")
print("   %-18s %-12s %-12s %-12s %s" % ("", "at native", "native hf", "at 533x400", "hf at 533x400"))
for name, p in REFS:
    a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0
    nat = hf_energy(a)
    small = np.asarray(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).resize((533, 400), Image.LANCZOS),
                       np.float32) / 255.0
    sm = hf_energy(small)
    print("   %-18s %-12s %-12.5f %-12s %.5f   (%.0f%% of its own native detail kept)"
          % (name, "native", nat, "533x400", sm, 100 * sm / max(1e-9, nat)))

print()
print("   === so, is the comparison unfair? ===")
photo = np.asarray(Image.open(REFS[3][1]).convert("RGB"), np.float32) / 255.0
real0 = np.asarray(Image.open(REFS[0][1]).convert("RGB"), np.float32) / 255.0
ph = hf_energy(np.asarray(Image.fromarray((np.clip(photo, 0, 1) * 255).astype(np.uint8)).resize((533, 400), Image.LANCZOS), np.float32) / 255.0)
rh = hf_energy(np.asarray(Image.fromarray((np.clip(real0, 0, 1) * 255).astype(np.uint8)).resize((533, 400), Image.LANCZOS), np.float32) / 255.0)
print("      the photograph at 533x400: hf %.5f" % ph)
print("      the footage at 533x400:    hf %.5f" % rh)
print("      ratio footage/photograph:  %.2f" % (rh / max(1e-9, ph)))
print()
if rh / max(1e-9, ph) > 1.5:
    print("   -> the footage carries markedly MORE fine detail at this size than the photograph does.  The model's input")
    print("      is the photograph, so a render compared against the footage is being asked to invent detail that its")
    print("      own reference does not contain.  **That is a real and unaccounted-for gap, and it is not the model's.**")
else:
    print("   -> the two carry comparable detail at this size, so the comparison is fair and the missing detail is the")
    print("      model's own.")
