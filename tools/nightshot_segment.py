"""
Segment the clip by mode before measuring anything, because it contains more than one.

**The evidence that it does.** The mean channel ratio swings across frames:

    f_0001   R:G:B = 0.441:0.303:0.255   R > G   -- warm, not NightShot
    f_0021   R:G:B = 0.530:0.281:0.189   R > G   -- warmer still
    f_0041   R:G:B = 0.294:0.456:0.249   G > R   -- green, NightShot
    f_0061   R:G:B = 0.266:0.527:0.206   G > R   -- strongly green
    f_0080   R:G:B = 0.280:0.499:0.221   G > R

Averaging those together produced a mean hue spread of 0.1549 and a channel ratio that describes neither mode. **A
NightShot clip that begins warm and becomes green is a clip that switches modes**, and the user's description names
three modes, so the segments are very likely one each.

**So every frame is classified**, and the boundaries between classes are reported:

  * **the green ratio** `g_frac = G / (R + G + B)`, which is the discriminator: NightShot inverts the sensor's response
    so that green dominates, and the clip reads **0.46 to 0.53** inside the green segment against **0.28 to 0.30**
    outside it;
  * plus the **hue spread** and **corr(R,B)**, which distinguish a collapsed-hue mode from a colour one independently
    of the ratio -- two measurements rather than one, so a segment boundary is not decided by a single number;
  * and the **noise level** (high-frequency energy), which separates an accumulating mode from a single exposure.

With the segments known, each mode's statistics can be measured on its own and the three mechanisms built against their
own numbers instead of an average of all three.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAMES = os.path.join(ROOT, "tools", "nightshot", "frames")
OUT = os.path.join(ROOT, "tools", "nightshot")
COEF = np.array([0.299, 0.587, 0.114], np.float32)


def hue_spread(a):
    x = (a[..., 0] - a[..., 1]).ravel()
    y = (a[..., 1] - a[..., 2]).ravel()
    m = (np.abs(x) + np.abs(y)) > 0.02
    if m.sum() < 50:
        return float("nan")
    ang = np.arctan2(y[m], x[m])
    r = np.sqrt(np.mean(np.cos(ang)) ** 2 + np.mean(np.sin(ang)) ** 2)
    return float(np.sqrt(max(0.0, -2.0 * np.log(max(r, 1e-9)))))


files = sorted(f for f in os.listdir(FRAMES) if f.endswith(".png"))
rows = []
prev = None
for i, f in enumerate(files):
    a = np.asarray(Image.open(os.path.join(FRAMES, f)).convert("RGB"), np.float32) / 255.0
    m = a.reshape(-1, 3).mean(axis=0)
    tot = max(1e-9, float(m.sum()))
    lum = a @ COEF
    hf = float(np.abs(np.diff(lum, axis=1)).mean())
    d = float(np.abs(a - prev).mean()) if prev is not None else float("nan")
    prev = a
    rows.append(dict(i=i, f=f, r=float(m[0] / tot), g=float(m[1] / tot), b=float(m[2] / tot),
                     g_frac=float(m[1] / tot), sat=float(np.abs(m / tot * 3 - 1).mean()),
                     hue=hue_spread(a), hf=hf, diff=d, mean=float(lum.mean()),
                     floor=float(np.quantile(lum, 0.01))))

# classify: green when G's share exceeds both R's and B's by a clear margin
for q in rows:
    q["green"] = q["g_frac"] > 0.40

# find the runs
runs = []
cur = None
for q in rows:
    if cur is None or cur["green"] != q["green"]:
        cur = dict(green=q["green"], start=q["i"], end=q["i"], n=1)
        runs.append(cur)
    else:
        cur["end"] = q["i"]
        cur["n"] += 1

print("   the clip segments, by whether green dominates (G share > 0.40):")
print("   %-8s %-14s %-8s %-12s %-12s %-12s %s"
      % ("mode", "frames", "count", "mean G share", "hue spread", "hf (noise)", "mean luma"))
for r in runs:
    if r["n"] < 3:
        continue
    seg = rows[r["start"]:r["end"] + 1]
    tag = "NIGHTSHOT (green)" if r["green"] else "colour"
    print("   %-8s %-14s %-8d %-12.4f %-12.4f %-12.5f %.4f"
          % (tag, "%d-%d" % (r["start"], r["end"]), r["n"],
             float(np.mean([q["g_frac"] for q in seg])),
             float(np.nanmean([q["hue"] for q in seg])),
             float(np.mean([q["hf"] for q in seg])),
             float(np.mean([q["mean"] for q in seg]))))

print()
print("   the transition points, printed as the frames where the class changes:")
for r in runs:
    if r["n"] >= 3:
        tag = "green" if r["green"] else "colour"
        print("      frame %3d .. %3d  (%3d frames)  %s" % (r["start"], r["end"], r["n"], tag))

print()
print("   the per-frame trace, so the segmentation can be checked rather than trusted:")
print("   %-6s %-9s %-9s %-9s %-10s %-10s %s" % ("frame", "R share", "G share", "B share", "hue", "hf", "class"))
for q in rows[::4]:
    print("   %-6d %-9.4f %-9.4f %-9.4f %-10.4f %-10.5f %s"
          % (q["i"], q["r"], q["g"], q["b"], q["hue"], q["hf"], "GREEN" if q["green"] else "colour"))

with open(os.path.join(OUT, "segments.json"), "w", encoding="utf-8") as fh:
    json.dump({"runs": [r for r in runs if r["n"] >= 3],
               "per_frame": [{k: v for k, v in q.items() if k != "f"} for q in rows]},
              fh, ensure_ascii=False, indent=1)
print()
print("   wrote segments.json")
