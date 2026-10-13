"""
Measure the two features the reference shows and the model does not, before changing anything.

**What was reported.** (a) A source that is orange-white wears a **blue light bar**, and (b) the white light has
**coloured streaks to its right**, whose shape differs from what the model produces. Both are specific and both are
measurable, and the previous several attempts all failed by proposing a mechanism and looking for it -- so the
features are described as numbers first.

The lamp in the attachment is at roughly (row 280, col 940) of a 1203x900 frame: a warm core, a pale pole below it,
and a blue vertical line down that pole. To the right of the core there is a fan of pale streaks.

Reported per region:

  * the **colour of the core** and of the **pole below it**, and the pole's blue-versus-the-core difference, which
    is what "a blue bar on an orange-white source" means numerically;
  * the **vertical profile** down the pole, because a "bar" is a run of rows rather than a point;
  * and the **angular profile** around the core, which is where the coloured streaks would show as a modulation of
    hue with angle -- the same statistic that was used for the starburst, but on the reference rather than on a render.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "tools", "ntscrs_ref")
COEF = np.array([0.299, 0.587, 0.114], np.float32)

SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    OUT, "..", "..", "..", "..", "..", "..", "Users", "Administrator", ".dsh", "attachments",
    "v1", "objects", "03", "039d9af408f07923375d9fae0a7fb6839668f20b2c0508face5b0a2a14b56eb9")
SRC = os.path.abspath(SRC)
if not os.path.exists(SRC):
    print("usage: python tools/measure_reference.py <image>")
    raise SystemExit(2)

img = np.asarray(Image.open(SRC).convert("RGB"), np.float32) / 255.0
H, W = img.shape[:2]
lum = img @ COEF
print("   the reference: %dx%d  mean luma %.4f  blown %.5f"
      % (W, H, float(lum.mean()), float((lum > 0.97).mean())))

# the lamp: the brightest patch, averaged so one hot pixel does not choose it
k = 31
box = np.lib.stride_tricks.sliding_window_view(lum, (k, k)).mean(axis=(2, 3))
cy, cx = np.unravel_index(int(np.argmax(box)), box.shape)
cy += k // 2
cx += k // 2
print("   the lamp core is at (row %d, col %d)" % (cy, cx))


def patch(r0, r1, c0, c1):
    r0, r1 = max(0, r0), min(H, r1)
    c0, c1 = max(0, c0), min(W, c1)
    return img[r0:r1, c0:c1].reshape(-1, 3).mean(axis=0)


core = patch(cy - 12, cy + 12, cx - 12, cx + 12)
print()
print("   (a) the blue bar over an orange-white source")
print("      the core          RGB %s   R-B %+.4f" % (np.round(core, 3), float(core[0] - core[2])))
print("      %-18s %-22s %-12s %s" % ("distance below", "RGB", "B-R", "bluer than the core?"))
for d in (20, 40, 60, 80, 100, 130, 170):
    p = patch(cy + d, cy + d + 14, cx - 6, cx + 6)
    print("      %-18d %-22s %-12.4f %s"
          % (d, np.round(p, 3), float(p[2] - p[0]),
             "yes" if float(p[2] - p[0]) > float(core[2] - core[0]) + 0.02 else "no"))

# the vertical profile, so the bar's extent and colour are explicit
print()
print("   the pole's colour down 200 rows, in 20-row steps (B-R and R-B):")
prof = []
for d in range(0, 200, 20):
    p = patch(cy + d, cy + d + 20, cx - 6, cx + 6)
    prof.append((d, float(p[2] - p[0]), float(p[0]), float(p[2])))
    print("      +%-4d B-R %+.4f  R %.3f  B %.3f" % (d, prof[-1][1], prof[-1][2], prof[-1][3]))
bluest = max(prof, key=lambda t: t[1])
print("      the bluest run is at +%d rows, B-R %+.4f" % (bluest[0], bluest[1]))

print()
print("   (b) the angular structure around the core, which is where coloured streaks would show")
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
ang = np.degrees(np.arctan2(-(yy - cy), xx - cx)) % 360.0     # 0 = right, 90 = up
for radius in (60, 100, 150, 200):
    m = (r >= radius - 8) & (r < radius + 8)
    if m.sum() < 40:
        continue
    bins = np.zeros(36)
    hue = np.zeros(36)
    idx = (ang[m] / 10.0).astype(int) % 36
    rb = (img[..., 0] - img[..., 2])[m]
    lum_m = lum[m]
    for b in range(36):
        sel = idx == b
        if sel.any():
            bins[b] = lum_m[sel].mean()
            hue[b] = rb[sel].mean()
    # the right half is 0..90 and 270..360 degrees
    right = np.r_[bins[0:9], bins[27:36]]
    left = bins[9:27]
    print("      r=%-4d luma: right half %.4f, left half %.4f (right/left %.2f) | R-B spread around the"
          " circle %.5f" % (radius, float(right.mean()), float(left.mean()),
                            float(right.mean()) / max(1e-9, float(left.mean())),
                            float(np.ptp(hue))))
print()
print("   a fan of streaks to the RIGHT shows as the right half being brighter than the left, and as R-B")
print("   varying with angle; a symmetric glare shows as neither.")
