"""
Scan every frame at the source's own 30 fps, and cache the statistics so they can be re-analysed without re-decoding.

**Why the 2 fps sample was not enough.** It took 207 frames from 3111, and a mode that is held for a second or two --
which is what a camera operator cycling through functions produces -- can fall entirely between samples. **The "mid"
classes it reported were transitions**, and it could not distinguish a brief mode from the camera being moved. At 30 fps
every frame is seen, and a mode that lasts even a quarter of a second leaves seven frames behind.

**The decode is piped rather than written to disk**, because 3111 frames of 1440x1080 PNG is several gigabytes and the
statistics need the pixels, not the files. A **scale to 480 wide** keeps the point-source structure measurable while
making the scan a matter of a minute or two rather than an hour -- and the scale is recorded, since a point-source
density depends on it and every figure has to be comparable with itself.

**What is measured per frame**, chosen so the modes can be told apart by physics rather than by eye:

  * **the channel means below the display band** -- the G share separates a green display from a colour one, and a
    collapse onto one axis is what NightShot is;
  * **the hue spread** -- large for a colour picture, small when the hue has been collapsed;
  * **the mean luma** -- the night-vision functions raise the gain, and Super NightShot raises it further;
  * **the high-frequency energy** -- frame accumulation *lowers* it, so this is where the two green modes differ;
  * **the frame-to-frame difference**, split into a **motion** component and an **overall level change**, because an
    accumulating mode smears motion while a gain change moves the whole frame together.

**The results are cached to a JSON file**, so the analysis can be re-run against the same numbers without decoding again
-- and so the measurements can be inspected rather than taken on trust.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "tools", "calib")
os.makedirs(OUT, exist_ok=True)
CACHE = os.path.join(OUT, "scan30.json")
VIDEO = r"C:\Users\Administrator\Downloads\10月13日.mp4"
FFMPEG = (r"E:\ffmpeg-6.1.1-essentials_build\ffmpeg-6.1.1-essentials_build\bin\ffmpeg.exe")
COEF = np.array([0.299, 0.587, 0.114], np.float32)
SCALE_W = 480
OSD = 0.22

if os.path.exists(CACHE) and "--fresh" not in sys.argv:
    print("   the cache exists; reading it (pass --fresh to rescan)")
    rows = json.load(open(CACHE, encoding="utf-8"))
    print("   %d frame(s) cached" % len(rows))
else:
    W, H = SCALE_W, int(SCALE_W * 1080 / 1440)
    W -= W % 2
    H -= H % 2
    cmd = [FFMPEG, "-v", "error", "-i", VIDEO,
           "-vf", "scale=%d:%d" % (W, H), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    print("   decoding at %dx%d, every frame, through a pipe" % (W, H))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    nbytes = W * H * 3
    top = int(H * OSD)
    rows = []
    prev = None
    prev_lum = None
    i = 0
    while True:
        buf = proc.stdout.read(nbytes)
        if len(buf) < nbytes:
            break
        a = np.frombuffer(buf, np.uint8).reshape(H, W, 3).astype(np.float32) / 255.0
        scene = a[top:]                                  # **below the display band**
        lum = scene @ COEF
        tot = max(1e-9, float(scene.sum(axis=-1).mean()))
        m = scene.reshape(-1, 3).mean(axis=0) / tot
        hf = float(np.abs(np.diff(lum, axis=1)).mean())
        x = (scene[..., 0] - scene[..., 1]).ravel()
        y = (scene[..., 1] - scene[..., 2]).ravel()
        mm = (np.abs(x) + np.abs(y)) > 0.02
        hue = float("nan")
        if mm.sum() > 50:
            ang = np.arctan2(y[mm], x[mm])
            r = float(np.sqrt(np.mean(np.cos(ang)) ** 2 + np.mean(np.sin(ang)) ** 2))
            hue = float(np.sqrt(max(0.0, -2.0 * np.log(max(r, 1e-9)))))
        mean = float(lum.mean())
        # split the frame-to-frame change into motion and a common level shift
        motion = 0.0
        shift = 0.0
        if prev is not None:
            d = scene - prev
            shift = float(d.mean())
            motion = float(np.abs(d - shift).mean())
        rows.append(dict(i=i, t=i / 30.0, r=float(m[0]), g=float(m[1]), b=float(m[2]),
                         hue=hue, mean=mean, hf=hf, motion=motion, shift=shift))
        prev = scene
        i += 1
        if i % 500 == 0:
            print("      %d frames..." % i)
    proc.wait()
    print("   decoded %d frame(s)" % len(rows))
    json.dump(rows, open(CACHE, "w", encoding="utf-8"))
    print("   cached to %s" % os.path.relpath(CACHE, ROOT))

if not rows:
    raise SystemExit("no frames")

# --- the overview: what the whole clip does, in bands
lum = np.array([q["mean"] for q in rows])
g = np.array([q["g"] for q in rows])
hf = np.array([q["hf"] for q in rows])
mot = np.array([q["motion"] for q in rows])
print()
print("   %d frames over %.1f s at the source's 30 fps" % (len(rows), rows[-1]["t"]))
print("      mean luma   min %.4f  p50 %.4f  max %.4f" % (lum.min(), np.median(lum), lum.max()))
print("      G share     min %.4f  p50 %.4f  max %.4f" % (g.min(), np.median(g), g.max()))
print("      hf energy   min %.5f  p50 %.5f  max %.5f" % (hf.min(), np.median(hf), hf.max()))
print("      motion      min %.5f  p50 %.5f  max %.5f" % (mot.min(), np.median(mot), mot.max()))

# how many frames have ANY real picture, by a brightness floor well above the black frames
BLACK = 0.05
alive = lum > BLACK
print()
print("   frames with a real picture (mean luma > %.2f): %d of %d (%.0f%%)"
      % (BLACK, int(alive.sum()), len(rows), 100.0 * float(alive.mean())))
runs = []
cur = None
for q in rows:
    live = q["mean"] > BLACK
    if cur is None or cur["live"] != live:
        cur = dict(live=live, start=q["i"], end=q["i"], n=1)
        runs.append(cur)
    else:
        cur["end"] = q["i"]
        cur["n"] += 1
print("   the runs of live vs black frames:")
for r in runs:
    if r["n"] < 20:
        continue
    seg = rows[r["start"]:r["end"] + 1]
    print("      %-6s %-14s %-6d frames  %-8.1f s   luma %.3f-%.3f   G %.3f-%.3f   hf %.4f-%.4f"
          % ("LIVE" if r["live"] else "black", "%d-%d" % (r["start"], r["end"]), r["n"], r["n"] / 30.0,
             min(x["mean"] for x in seg), max(x["mean"] for x in seg),
             min(x["g"] for x in seg), max(x["g"] for x in seg),
             min(x["hf"] for x in seg), max(x["hf"] for x in seg)))
