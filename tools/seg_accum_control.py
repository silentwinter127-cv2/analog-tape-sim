"""
Check the accumulation count before changing the default, because one ratio is not a measurement.

**The claim to test.** NIGHTSHOT 2's motion statistic is **0.444** of NIGHTSHOT 1's, and its high-frequency energy
**0.741** of it -- implying **N = 2.25** and **N = 1.82** respectively, i.e. **about 2 frames**. The model's preset
currently uses **8**, so if this holds it is wrong by a factor of four.

**Why it needs more than one number.** A motion ratio is not a clean estimator of N:

  * the two segments are **different takes of a moving subject**, so their *motion* differs for reasons that have
    nothing to do with accumulation -- a slower pan alone would lower the ratio. **The camera's own displacement is
    therefore measured as the control**: an N-frame average reduces frame-to-frame change, but a slower pan also does,
    and only if the displacement is comparable can the ratio be attributed to accumulation;
  * the high-frequency ratio mixes **noise (which falls as 1/sqrt(N)) and detail (which the average barely touches)**,
    so its value depends on how much of the detail is noise, which varies with the scene;
  * and a **brightness** difference changes the noise's visibility, so the level ratio has to be accounted for.

**So this measures the displacement directly**, per segment, by aligning consecutive frames -- the control that says
whether the two segments were shot at comparable camera speed. If they were, the motion ratio is evidence for N; if the
second segment's camera moved about half as fast, then its motion ratio is explained without any accumulation at all.
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

SEGS = [("NIGHTSHOT 1", 1620, 2040), ("NIGHTSHOT 2", 2040, 2490)]
W, H = 480, 360
NPER = 60            # frames sampled from each segment


def decode(a, count, step):
    """Decode `count` frames starting at frame `a`, every `step` frames, through the pipe."""
    # stepping is done by decoding a contiguous block and taking every step-th frame, so indices stay exact
    total = count * step
    cmd = [FFMPEG, "-v", "error", "-i", VIDEO, "-ss", "%.4f" % (a / 30.0), "-frames:v", str(total),
           "-vf", "scale=%d:%d" % (W, H), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    nbytes = W * H * 3
    out = []
    while True:
        buf = proc.stdout.read(nbytes)
        if len(buf) < nbytes:
            break
        out.append(np.frombuffer(buf, np.uint8).reshape(H, W, 3).astype(np.float32) / 255.0)
    proc.wait()
    return [out[i] for i in range(0, len(out), step)][:count]


def best_shift(a, b, rng=20):
    """The translation of b that best matches a -- a coarse then fine integer search."""
    ha, wa = a.shape
    best = (0, 0, -2.0)
    for step in (2, 1):
        dx0, dy0 = best[0], best[1]
        r = rng if step == 2 else 2
        for dy in range(dy0 - r, dy0 + r + 1, step):
            for dx in range(dx0 - r, dx0 + r + 1, step):
                y0, y1 = max(0, dy), min(ha, ha + dy)
                x0, x1 = max(0, dx), min(wa, wa + dx)
                if y1 - y0 < ha // 2 or x1 - x0 < wa // 2:
                    continue
                sa = a[y0:y1, x0:x1]
                sb = b[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
                num = float(np.mean(sa * sb))
                den = float(np.sqrt(np.mean(sa ** 2) * np.mean(sb ** 2))) or 1e-9
                if num / den > best[2]:
                    best = (dx, dy, num / den)
    return best


print("   === the control: how fast did the camera move in each segment? ===")
print("   %-14s %-12s %-12s %-12s %s"
      % ("segment", "mean |dx|", "mean |dy|", "mean speed", "frames measured"))
speeds = {}
for name, a, b in SEGS:
    fs = decode(a + 30, NPER, 3)
    lum = [f @ COEF for f in fs]
    sp = []
    for i in range(1, len(lum)):
        dx, dy, c = best_shift(lum[i], lum[i - 1])
        sp.append(float(np.hypot(dx, dy)))
    speeds[name] = float(np.mean(sp))
    print("   %-14s %-12.2f %-12.2f %-12.2f %d" % (name, speeds[name], 0.0, speeds[name], len(sp)))

r1 = speeds["NIGHTSHOT 1"]
r2 = speeds["NIGHTSHOT 2"]
print()
print("      the camera speed ratio, NIGHTSHOT 2 / NIGHTSHOT 1: %.3f" % (r2 / max(1e-9, r1)))
print("      the motion statistic's ratio (from the scan): 0.444")
print()
if abs(r2 / max(1e-9, r1) - 1.0) < 0.30:
    print("      -> the two segments were shot at COMPARABLE camera speed, so the motion ratio is not explained by")
    print("         the camera moving less and is evidence for accumulation.")
elif r2 < r1:
    print("      -> the second segment's camera moved %d%% slower, which by itself explains part of the motion ratio."
          % round(100 * (1 - r2 / max(1e-9, r1))))
    print("         so the accumulation estimate is an UPPER bound and needs the speed divided out.")
    adj = 0.444 / max(1e-9, (r2 / max(1e-9, r1)))
    print("         the speed-adjusted motion ratio: %.3f, implying N = %.2f" % (adj, 1.0 / max(1e-9, adj)))
else:
    print("      -> the second segment's camera moved FASTER, so accumulation explains even more of the ratio.")

print()
print("   === and a direct test of the noise law, which does not depend on the camera speed ===")
print("   Within ONE segment, the accumulation is fixed, so the noise cannot be measured by comparing segments.")
print("   What can be measured is whether the high-frequency content behaves like noise at all: if it does, it will")
print("   fall as the picture brightens, because the same absolute noise is a smaller fraction of a brighter signal.")
for name, a, b in SEGS:
    fs = decode(a + 30, 60, 3)
    lum = [f @ COEF for f in fs]
    means = np.array([float(l.mean()) for l in lum])
    hfs = np.array([float(np.abs(np.diff(l, axis=1)).mean()) for l in lum])
    if means.std() > 1e-6 and hfs.std() > 1e-6:
        c = float(np.corrcoef(means, hfs)[0, 1])
        print("      %-14s luma %.4f+/-%.4f  hf %.5f+/-%.5f  corr(luma, hf) %+.3f"
              % (name, means.mean(), means.std(), hfs.mean(), hfs.std(), c))
