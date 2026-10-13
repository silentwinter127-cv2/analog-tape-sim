"""
Verify the boundaries the user gave, then measure each mode -- rather than taking the boundaries on trust.

**The boundaries, as given, at 30 fps:**

    0-690      0-23 s      normal colour
    690-1350  23-45 s      the first mode (NightShot)
    1350-1560 45-52 s      normal colour
    1560-1590 52-53 s      the first mode (NightShot), a short segment
    1620-2040 54-68 s      NIGHTSHOT 1
    2040-2490 68-83 s      NIGHTSHOT 2

**Two things are checked before anything is measured, and each can fail independently:**

  * **is there a discontinuity at each boundary?** A cut between recordings shows as a collapse in the consecutive-frame
    correlation. If the boundaries are real, each one should show it -- and if one does not, the honest reading is that
    the boundary is a *mode change within a take* rather than a cut, which is a different thing and matters for whether
    the segment's statistics are an average;
  * **are the segments actually different from each other?** If two named modes measure the same, then either the names
    denote something the statistics cannot see, or one of them is not doing what it is called.

**And the two NightShot segments are the ones with a claim attached** -- "NIGHTSHOT 1" and "NIGHTSHOT 2" should differ
in a way that is visible in the numbers, or the naming is not something this data supports.

**The 53-54 s gap the user did not name (frames 1590-1620) is left unclassified** rather than being assigned to a
neighbour, because assigning it would be an assumption of exactly the kind that has already gone wrong twice here.
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "tools", "calib")
CACHE = os.path.join(OUT, "scan30.json")
CUTS = os.path.join(OUT, "cuts.json")
VIDEO = r"C:\Users\Administrator\Downloads\10月13日.mp4"
FFMPEG = (r"E:\ffmpeg-6.1.1-essentials_build\ffmpeg-6.1.1-essentials_build\bin\ffmpeg.exe")
COEF = np.array([0.299, 0.587, 0.114], np.float32)

SEGS = [
    ("normal (0-23 s)", 0, 690),
    ("first mode / NightShot (23-45 s)", 690, 1350),
    ("normal (45-52 s)", 1350, 1560),
    ("first mode / NightShot (52-53 s)", 1560, 1590),
    ("unclassified (53-54 s)", 1590, 1620),
    ("NIGHTSHOT 1 (54-68 s)", 1620, 2040),
    ("NIGHTSHOT 2 (68-83 s)", 2040, 2490),
    ("after 83 s", 2490, None),
]

rows = json.load(open(CACHE, encoding="utf-8"))
n = len(rows)
corr = np.array(json.load(open(CUTS, encoding="utf-8"))["corr"], float)
print("   %d frames cached, %d correlation(s)" % (n, len(corr)))

print()
print("   === 1. is there a discontinuity at each boundary? ===")
print("   %-8s %-9s %-12s %-12s %s" % ("frame", "sec", "corr here", "median nearby", "reading"))
for _, a, b in SEGS[1:]:
    if b is None or b >= len(corr):
        continue
    here = float(corr[b - 1]) if b - 1 < len(corr) else float("nan")
    lo, hi = max(0, b - 60), min(len(corr), b + 60)
    nb = float(np.median(np.concatenate([corr[lo:b - 5], corr[b + 5:hi]]))) if hi - lo > 20 else float("nan")
    read = "a discontinuity" if here < 0.85 else ("NO discontinuity -- a mode change inside a take" if here > 0.97 else "weak")
    print("   %-8d %-9.1f %-12.4f %-12.4f %s" % (b, b / 30.0, here, nb, read))

print()
print("   === 2. each segment's own character, from the 30 fps scan ===")
print("   %-36s %-7s %-8s %-8s %-8s %-8s %-9s %s"
      % ("segment", "frames", "R", "G", "B", "hue", "luma", "hf"))
res = {}
for name, a, b in SEGS:
    bb = n if b is None else min(b, n)
    if bb - a < 10:
        print("   %-36s only %d frame(s), too short to characterise" % (name, bb - a))
        continue
    seg = rows[a:bb]
    R = float(np.mean([q["r"] for q in seg]))
    G = float(np.mean([q["g"] for q in seg]))
    B = float(np.mean([q["b"] for q in seg]))
    hue = float(np.nanmean([q["hue"] for q in seg]))
    luma = float(np.mean([q["mean"] for q in seg]))
    hf = float(np.mean([q["hf"] for q in seg]))
    res[name] = dict(R=R, G=G, B=B, hue=hue, luma=luma, hf=hf, a=a, b=bb)
    print("   %-36s %-7d %-8.4f %-8.4f %-8.4f %-8.4f %-9.4f %.5f"
          % (name, bb - a, R, G, B, hue, luma, hf))

print()
print("   === 3. are the named modes actually different? ===")
keys = list(res)
for i in range(len(keys)):
    for j in range(i + 1, len(keys)):
        a, b = res[keys[i]], res[keys[j]]
        dg = abs(a["G"] - b["G"])
        dl = abs(a["luma"] - b["luma"])
        dhf = abs(a["hf"] - b["hf"])
        # a rough separation: G share is the mode discriminator, luma and hf follow
        note = ""
        if dg < 0.01 and dl < 0.02:
            note = "  <- indistinguishable"
        print("   %-34s vs %-34s dG %.4f  dluma %.4f  dhf %.5f%s"
              % (keys[i][:34], keys[j][:34], dg, dl, dhf, note))
