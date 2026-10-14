"""
Can the local model give a DEPTH map, so the illuminator's falloff can be per-pixel rather than radial?

**Why this matters for the note's first mechanism.**  `note.txt` asks for

    "模拟机身红外灯的照射范围、光束中心与边缘的亮度差异，以及物体距离增加后反射光减弱的现象。
     重点：近处物体可能很亮，远处背景却仍然很暗。"

**and the model has no distance term at all**: `illuminator_field` builds a radial falloff from `beam_x`, `beam_y` and
`beam_angle`, normalised by its own mean, so it can produce a vignette (the default spans only **21%**) but never a
*decline with distance*.  **Distance is what the mechanism is about**, and a radial profile cannot express it.

**A classifier over ordinal depth bins can, and needs no depth checkpoint.**  `siglip2-base-patch16-224` scores an image
against arbitrary text, so prompts of the form "very close to the camera" ... "in the far distance" give, per cell, a
probability distribution over **ordinal** bins.  **Their expectation is a continuous depth per cell** -- not a hard
label -- which is exactly what an inverse-square falloff needs, and it degrades gracefully: an image the model cannot
place gets a flat distribution and therefore a mid depth, so the falloff stays gentle instead of inventing structure.

**And it must be validated before it is trusted**, because a wrong depth puts the lamp's light in the wrong place.  **The
test is whether the model's ordering is monotonic**: adjacent bins should be ordered, not scrambled, and the expectation
should separate the near field from the far field.  **The statistic that says so is the correlation between a cell's
depth estimate and its own height in the frame**, since in this scene -- and in most camcorder footage -- the floor runs
from near at the bottom to far at the top.  **A model that says "the bottom of the frame is nearer" is usable; one that
says nothing is not.**
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

os.environ.setdefault("HF_HUB_OFFLINE", "1")

OBJ = (r"C:\Users\Administrator\.dsh\attachments\v1\objects")
PHOTO = os.path.join(OBJ, "69", "699c23315df037b01fa831a273364b41f99c28733fb1dc928b22d1e6c281818d")
MODEL = r"D:\siglip2-base-patch16-224"

#: ordinal depth bins.  The wording matters more than the count: these have to be *ordered* phrases a captioning
#: corpus would use, because the model was trained on captions rather than on depth.
BINS = ["something extremely close to the camera, right in front of the lens",
        "something very near the camera, within arm's reach",
        "something a short distance away, a metre or two",
        "something a few metres away in the middle of the room",
        "something far away, across the room",
        "the distant background, far behind everything else"]

img = Image.open(PHOTO).convert("RGB")
W, H = img.size
a = np.asarray(img, np.float32) / 255.0
lum = np.clip(a @ np.array([0.299, 0.587, 0.114], np.float32), 0, 1)
print("   the photograph %dx%d, luma mean %.4f" % (W, H, float(lum.mean())))

import torch                                                                   # noqa: E402
from transformers import AutoModel, AutoProcessor                              # noqa: E402

print("   loading %s ..." % MODEL)
model = AutoModel.from_pretrained(MODEL, local_files_only=True)
proc = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
model.eval()

G = 8
cells, meta = [], []
for j in range(G):
    for i in range(G):
        x0, y0 = i * W // G, j * H // G
        x1, y1 = (i + 1) * W // G, (j + 1) * H // G
        cells.append(img.crop((x0, y0, x1, y1)))
        meta.append((j, i, float(lum[y0:y1, x0:x1].mean())))

with torch.no_grad():
    inputs = proc(text=BINS, images=cells, padding="max_length", return_tensors="pt")
    P = model(**inputs).logits_per_image.softmax(dim=-1).cpu().numpy()

#: the expectation over the ordinal bins, mapped to 0 (nearest) .. 1 (farthest)
idx = np.arange(len(BINS), dtype=np.float32)
depth = (P * idx).sum(axis=1) / (len(BINS) - 1)
D = depth.reshape(G, G)

print()
print("   === the depth the model infers, 8x8 (0 = nearest, 1 = farthest) ===")
for j in range(G):
    print("      " + " ".join("%.2f" % v for v in D[j]))
print()
print("   === and the checks that decide whether it is usable ===")
# **`rows` has to be a COLUMN of shape (G,1) repeated across the width, so that `ravel()` walks the same order as
# `D.ravel()`.**  Written as `rows.repeat(G, 1)` it repeats the *column* along axis 1 and then flattens row-major into
# `[0,0,...,0,1,1,...]`, which is the same sequence that `np.repeat(np.arange(G), G)` gives -- correct -- but an earlier
# version of the test built the index the other way round and reported a correlation of **-0.295 while the half-means
# said depth increases upwards**, i.e. two checks that contradict each other.  **When two of my own checks disagree, the
# instrument is what is wrong**, so both are printed side by side here rather than one being trusted.
row_idx = np.repeat(np.arange(G, dtype=np.float32), G)
c = float(np.corrcoef(row_idx, D.ravel())[0, 1])
top = D[: G // 2].mean()
bot = D[G // 2:].mean()
print("      the top half's mean depth %.3f; the bottom half's %.3f" % (top, bot))
print("      so the frame reads as %s" % ("farther at the top, i.e. a receding floor" if top > bot
                                          else "NEARER at the top, which is backwards"))
print("      correlation between depth and row index (0=top): %+.3f" % c)
print("      **the two must agree in sign**; if they do not, this test is wrong and not the model.")
print("      direct check: row 0's mean %.3f, row %d's mean %.3f" % (float(D[0].mean()), G - 1,
                                                                   float(D[G - 1].mean())))
# c) the per-cell confidence: is any cell a flat distribution?
conf = P.max(axis=1)
print("      per-cell confidence: mean %.3f  min %.3f  max %.3f"
      % (float(conf.mean()), float(conf.min()), float(conf.max())))
print("      how many cells are near-flat (confidence < 0.30): %d of %d"
      % (int((conf < 0.30).sum()), conf.size))
# d) does it separate the extremes at all?
print("      depth range across cells: %.3f .. %.3f  (spread %.3f)" % (float(D.min()), float(D.max()),
                                                                     float(D.std())))
print()
usable = (top > bot) and c > 0.3 and float(D.std()) > 0.05
if usable:
    print("   -> USABLE: the model orders depth monotonically with the frame and separates near from far, so an")
    print("      inverse-square falloff over this map is meaningful and is not inventing structure.")
else:
    print("   -> NOT usable as it stands: the ordering test fails, so a falloff built on this would put the lamp's")
    print("      light in the wrong place.  The depth must stay the radial estimate.")
