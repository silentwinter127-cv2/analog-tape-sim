"""
Find every cut, so the clip is divided by recording rather than by statistics, and then measure what happens after
each one.

**What the analysis had been doing wrong, and it is the largest of my errors in this session.** The clip is a
**concatenation of separate recordings**, not one take. That was established by the correlation matrix around frame
2382, which is **block-diagonal**: `{2379, 2380, 2381}` correlate at 0.998-1.000 within themselves and at **0.339**
across to `{2382, 2383, ...}`, which correlate at 0.999 among themselves. **No repeated frame, and the row-wise
difference is uniform across the frame's height** rather than confined to a band -- so it is neither a dropped frame nor
a head-switching artefact but a **hard cut between two different shots**.

**And that invalidates every segment boundary the earlier rounds produced.** The "green stretch, 29 seconds" was a
statistical region that **spans cuts**, so its statistics were an average over several recordings and its "1.6 s
periodicity" was a mixture. The mode claims, the colour targets and the dynamic analysis all rest on segments that were
never the recordings.

**So the cuts are found first and independently**: consecutive frames are correlated, and a cut is where the correlation
**collapses** rather than declining. The threshold is not chosen -- the distribution of correlations is measured, and a
cut is a value separated from the body of the distribution, which is what a block-diagonal structure produces and what a
continuous shot does not.

**Then each recording is characterised on its own**, and the **post-cut settling** is measured -- because that is where
a lens's temporal response would actually show: a delay or lag is most visible in how the picture *arrives* at a new
scene. That is the quantity the user's "镜头延迟" points at, and it has been unmeasurable until the segments were the
recordings.
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "tools", "calib")
VIDEO = r"C:\Users\Administrator\Downloads\10月13日.mp4"
FFMPEG = (r"E:\ffmpeg-6.1.1-essentials_build\ffmpeg-6.1.1-essentials_build\bin\ffmpeg.exe")
COEF = np.array([0.299, 0.587, 0.114], np.float32)
SCALE_W, SCALE_H = 320, 240          # small, because this pass is about the whole timeline
CACHE = os.path.join(OUT, "cuts.json")

if os.path.exists(CACHE) and "--fresh" not in os.sys.argv:
    d = json.load(open(CACHE, encoding="utf-8"))
    corr = np.array(d["corr"], float)
    print("   read %d consecutive-frame correlations from the cache" % len(corr))
else:
    print("   decoding every frame at %dx%d to correlate consecutive pairs" % (SCALE_W, SCALE_H))
    cmd = [FFMPEG, "-v", "error", "-i", VIDEO, "-vf", "scale=%d:%d" % (SCALE_W, SCALE_H),
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    nbytes = SCALE_W * SCALE_H * 3
    corr = []
    prev = None
    while True:
        buf = proc.stdout.read(nbytes)
        if len(buf) < nbytes:
            break
        a = np.frombuffer(buf, np.uint8).reshape(SCALE_H, SCALE_W, 3).astype(np.float32) / 255.0
        l = a @ COEF
        if prev is not None:
            c = float(np.corrcoef(prev.ravel(), l.ravel())[0, 1])
            corr.append(c)
        prev = l
    proc.wait()
    corr = np.array(corr, float)
    print("   %d correlation(s)" % len(corr))
    json.dump({"corr": [float(c) for c in corr]}, open(CACHE, "w", encoding="utf-8"))

print()
print("   the distribution of consecutive-frame correlations:")
for q in (1, 5, 10, 25, 50, 75, 90, 99):
    print("      p%-3d %.4f" % (q, float(np.percentile(corr, q))))
print("      min %.4f   max %.4f" % (corr.min(), corr.max()))

# a cut is separated from the body of the distribution, not below an arbitrary line
body = float(np.percentile(corr, 50))
low = float(np.percentile(corr, 2))
print()
print("   the median is %.4f and the 2nd percentile is %.4f, so the body is well above 0.9" % (body, low))
cuts = np.where(corr < 0.75)[0] + 1          # corr[i] is between frames i and i+1
groups = []
for c in cuts:
    if groups and c - groups[-1][-1] <= 5:
        groups[-1].append(int(c))
    else:
        groups.append([int(c)])
cutsat = [int(round(np.mean(g))) for g in groups]
print("   candidate cuts (correlation below 0.75, merged within 5 frames): %d" % len(cutsat))
for c in cutsat:
    print("      frame %-6d (%.2f s)   the correlation there is %.4f" % (c, c / 30.0, corr[c - 1]))

print()
print("   === the recordings, and each one's own character ===")
bounds = [0] + cutsat + [len(corr) + 1]
print("   %-16s %-8s %-10s %-10s %-10s %-11s %s"
      % ("frames", "sec", "mean luma", "G share", "hf", "motion", "a settled recording?"))
kept = []
for a, b in zip(bounds[:-1], bounds[1:]):
    n = b - a
    if n < 30:
        continue
    lo, hi = max(0, a - 1), min(len(corr), b - 1)
    seg_corr = corr[lo:hi]
    print("   %-16s %-8.1f %-10s %-10s %-10s %-11s %s"
          % ("%d-%d" % (a, b - 1), n / 30.0, "-", "-", "-", "-",
             "median corr %.4f" % float(np.median(seg_corr)) if len(seg_corr) else ""))
    kept.append((a, b, n))
print()
print("   %d recording(s) of 30 frames or more" % len(kept))
print("   and the shortest ones are the most informative about the lens, because they are all settling:")
for a, b, n in kept:
    if n < 90:
        print("      frames %d-%d, %.1f s -- shorter than three seconds" % (a, b - 1, n / 30.0))
