"""
Compare the reference's pole profile against the render's, because the difference is now a measured shape.

**The reference's profile, read the same way a render will be.** Down from a blown lamp core, in 20-row steps:

    +0      white  [1.000 1.000 1.000]   B-R  +0.0000
    +40     white  [1.000 0.996 0.999]   B-R  -0.0008
    +60     warm   [0.994 0.990 0.835]   B-R  -0.1582
    +100    orange [0.957 0.767 0.452]   B-R  -0.5052   <- the warmest
    +160    cool   [0.606 0.788 0.815]   B-R  +0.1828   <- the turn
    +180    blue   [0.586 0.815 0.815]   B-R  +0.2290   <- the bluest

**So the blue is not "next to" the light: the source's own blown core is white, the near field is warm, and the
cool shift happens 160-180 rows down, in the dimmer part.** My render puts the teal trail immediately under the core,
which is the shape difference the user described. And this is a **gradient**, not two colours -- which means it is
something that grows with distance below the source and with distance from the core, not an additive tint.

**The angular structure says where the fan lives**: at r=60, 100 and 150 the right and left halves differ by 1-4%,
and at **r=200 the right half is 31% brighter** -- so the coloured streaks to the right are an **outer** feature.

This script renders the current settings and reports the same two profiles, so the gap is a pair of numbers rather
than an impression.
"""
from __future__ import annotations

import io
import json
import os
import urllib.request

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "tools", "ntscrs_ref")
BASE = "http://127.0.0.1:8941"
COEF = np.array([0.299, 0.587, 0.114], np.float32)
REF = (r"C:\Users\Administrator\.dsh\attachments\v1\objects\03"
       r"\039d9af408f07923375d9fae0a7fb6839668f20b2c0508face5b0a2a14b56eb9")


def get_raw(path, timeout=180):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return r.read()


def post(path, obj, timeout=900):
    req = urllib.request.Request(BASE + path, data=json.dumps(obj).encode(),
                                headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def profiles(img):
    """The pole profile below the brightest core, and the right/left balance, for any picture."""
    H, W = img.shape[:2]
    lum = img @ COEF
    k = 31
    if H < k + 4 or W < k + 4:
        return None, None
    box = np.lib.stride_tricks.sliding_window_view(lum, (k, k)).mean(axis=(2, 3))
    cy, cx = np.unravel_index(int(np.argmax(box)), box.shape)
    cy += k // 2
    cx += k // 2
    # the extent available below the core, at this picture's size relative to the reference's 900 rows
    scale = H / 900.0
    prof = []
    for d in (0, 40, 60, 100, 160, 180):
        dd = int(d * scale)
        r0, r1 = min(H - 1, cy + dd), min(H, cy + dd + max(4, int(20 * scale)))
        c0, c1 = max(0, cx - 6), min(W, cx + 7)
        if r1 <= r0 or c1 <= c0:
            continue
        p = img[r0:r1, c0:c1].reshape(-1, 3).mean(axis=0)
        prof.append((d, float(p[0]), float(p[1]), float(p[2]), float(p[2] - p[0])))
    # the right/left balance at two radii, scaled with the picture
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    ang = np.degrees(np.arctan2(-(yy - cy), xx - cx)) % 360.0
    bal = []
    for rad in (100, 200):
        rr = rad * scale
        m = (r >= rr - 8 * scale) & (r < rr + 8 * scale)
        if m.sum() < 40:
            continue
        a = ang[m]
        right = (a < 90) | (a >= 270)
        bal.append((rad, float(lum[m][right].mean()) / max(1e-9, float(lum[m][~right].mean()))))
    return prof, bal


ref = np.asarray(Image.open(REF).convert("RGB"), np.float32) / 255.0
print("   %-8s %-30s %-30s" % ("rows below", "the reference", "the render"))
rp, rb = profiles(ref)
seed = json.loads(get_raw("/api/export"))
print("   (using the project's current settings, which are the reproduction)")

im = None
with urllib.request.urlopen(BASE + "/api/frame?t=0.0&scale=0.5&q=96", timeout=900) as r:
    im = np.asarray(Image.open(io.BytesIO(r.read())).convert("RGB"), np.float32) / 255.0
gp, gb = profiles(im)

if rp and gp:
    for i in range(min(len(rp), len(gp))):
        d = rp[i][0]
        print("   %-8d %-30s %-30s"
              % (d,
                 "RGB %.3f %.3f %.3f  B-R %+.3f" % (rp[i][1], rp[i][2], rp[i][3], rp[i][4]),
                 "RGB %.3f %.3f %.3f  B-R %+.3f" % (gp[i][1], gp[i][2], gp[i][3], gp[i][4])))

print()
print("   %-8s %-30s %s" % ("radius", "the reference right/left", "the render right/left"))
if rb and gb:
    for i in range(min(len(rb), len(gb))):
        print("   %-8d %-30.3f %.3f" % (rb[i][0], rb[i][1], gb[i][1]))

print()
print("   what the pairing says:")
if rp and gp:
    ref_blue = max(rp, key=lambda t: t[4])
    ren_blue = max(gp, key=lambda t: t[4])
    print("      the reference's bluest run is at +%d rows (B-R %+.3f)" % (ref_blue[0], ref_blue[4]))
    print("      the render's bluest run is at    +%d rows (B-R %+.3f)" % (ren_blue[0], ren_blue[4]))
    print("      -> the reference's cool shift is %s the render's"
          % ("FURTHER DOWN than" if ref_blue[0] > ren_blue[0] else "higher up than"))
if rb and gb:
    print("      right/left at the outer radius: reference %.2f, render %.2f"
          % (rb[-1][1], gb[-1][1]))
