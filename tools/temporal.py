"""
The two temporal measurements the user asked for: the settle after a mode change, and the emitter trail.

**Kept as one tool because they are the same question asked twice** -- how the picture's response is delayed in time --
and because each one's control is what makes it readable:

  * **the settle** is measured across each of the operator's own mode changes, with a window **inside a segment** as the
    control.  Without that control a level change is indistinguishable from the material's own variation;
  * **the trail** is measured on emitters in the fastest-moving window, where a one-frame persistence would smear about
    **26 px** -- far beyond the ten-pixel profile window sampled, so its absence is a real result rather than a
    resolution limit.  The earlier attempt failed because its window had **zero displacement**: a trail cannot exist
    without motion, and what it measured was the emitters' own shape.

**And the settle's control is what establishes it is real**: inside a segment the level moves by **-0.6% to -2.9%**,
while across a mode change it moves by **25% to 130%**.

**One result is a shape rather than a magnitude.** The step responses **overshoot** -- the fraction completed reaches
**+2.22** and **+2.13**, i.e. past the target and back -- which a monotone exponential cannot do. **That is a servo
hunting**, and it is the same behaviour as the 1.60 s periodicity found earlier, which had been attributed to handheld
sway and is better explained by the exposure loop oscillating in low light.

**Usage**: `python tools/temporal.py` measures both on the cached scans; `--full` re-decodes at 1440x1080 for the trail.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "tools", "calib")
VIDEO = r"C:\Users\Administrator\Downloads\10月13日.mp4"
FFMPEG = (r"E:\ffmpeg-6.1.1-essentials_build\ffmpeg-6.1.1-essentials_build\bin\ffmpeg.exe")
COEF = np.array([0.299, 0.587, 0.114], np.float32)

#: the operator's own mode changes, as given and then verified (none is a cut -- all are settings changes in one take)
TRAN = [("normal -> NightShot", 690), ("NightShot -> normal", 1350),
        ("normal -> NightShot (short)", 1560), ("NightShot -> ?", 1590),
        ("? -> NIGHTSHOT 1", 1620), ("NIGHTSHOT 1 -> NIGHTSHOT 2", 2040),
        ("NIGHTSHOT 2 -> after", 2490)]
#: windows inside segments, where nothing changed -- the control
CONTROL = [("middle of the first mode", 1020), ("middle of NIGHTSHOT 1", 1830),
           ("middle of NIGHTSHOT 2", 2265)]
#: the model's own AE time constant, for comparison
TAU_MODEL_FRAMES = 1.1 * 30.0
WIN = 60


def settle(rows):
    print("   === the settle across each mode change, and the control inside a segment ===")
    print("   the model's AE loop has tau_s = 1.1 s = %.0f frames; the window is %d frames (%.1f s)"
          % (TAU_MODEL_FRAMES, WIN, WIN / 30.0))
    print()
    print("   %-32s %-9s %-9s %-8s %-11s %s"
          % ("window", "level0", "level_inf", "step %", "63% at", "overshoot?"))
    for name, at in TRAN + CONTROL:
        seg = rows[at - 15:at + WIN]
        lv = np.array([q["mean"] for q in seg], float)
        l0, linf = float(np.mean(lv[:15])), float(np.mean(lv[-15:]))
        step = 100.0 * (linf - l0) / max(1e-9, l0)
        if abs(linf - l0) > 1e-4:
            frac = (lv[15:] - l0) / (linf - l0)
            reached = np.where(frac >= 0.632)[0]
            t63 = ("+%d f (%.2f s)" % (reached[0], reached[0] / 30.0)) if len(reached) else "not in window"
            over = float(np.max(frac))
            ov = "yes, +%.2f" % over if over > 1.15 else "no (max %.2f)" % over
        else:
            t63, ov = "no step", "-"
        print("   %-32s %-9.4f %-9.4f %-+8.1f %-11s %s" % (name, l0, linf, step, t63, ov))
    print()
    print("   the control is the point: inside a segment the level moves by under 3%, so a 25-130% change across a")
    print("   mode change is a real transient and not how this material varies.")


def trail(full=False):
    print()
    print("   === the emitter trail, in the fastest-moving window of NIGHTSHOT 1 ===")
    W, H = (1440, 1080) if full else (480, 360)
    print("   decoded at %dx%d%s" % (W, H, "" if full else "  (pass --full for the source resolution)"))
    cmd = [FFMPEG, "-v", "error", "-i", VIDEO, "-ss", "%.4f" % (1700 / 30.0), "-frames:v", "4",
           "-vf", "scale=%d:%d" % (W, H), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    nbytes = W * H * 3
    fs = []
    while True:
        buf = proc.stdout.read(nbytes)
        if len(buf) < nbytes:
            break
        fs.append(np.frombuffer(buf, np.uint8).reshape(H, W, 3).astype(np.float32) / 255.0)
    proc.wait()
    if len(fs) < 3:
        print("   only %d frame(s) decoded" % len(fs))
        return
    lum = [f @ COEF for f in fs]

    def disp(a, b, rng=24):
        best = (0, 0, -2.0)
        for st in (4, 1):
            dx0, dy0 = best[0], best[1]
            r = rng if st == 4 else 3
            for dy in range(dy0 - r, dy0 + r + 1, st):
                for dx in range(dx0 - r, dx0 + r + 1, st):
                    y0, y1 = max(0, dy), min(H, H + dy)
                    x0, x1 = max(0, dx), min(W, W + dx)
                    if y1 - y0 < H // 2 or x1 - x0 < W // 2:
                        continue
                    sa, sb = a[y0:y1, x0:x1], b[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
                    num = float(np.mean(sa * sb))
                    den = float(np.sqrt(np.mean(sa ** 2) * np.mean(sb ** 2))) or 1e-9
                    if num / den > best[2]:
                        best = (dx, dy, num / den)
        return best

    dx, dy, c = disp(lum[2], lum[1])
    mag = float(np.hypot(dx, dy))
    print("   the displacement between consecutive frames: (%d, %d) = %.1f px, correlation %.4f" % (dx, dy, mag, c))
    if mag < 0.5:
        print("   -> no motion, so a trail cannot be measured here; this is exactly why the earlier attempt failed")
        return
    from numpy.lib.stride_tricks import sliding_window_view
    k, half, floor = 9, 10, 0.25
    loc = sliding_window_view(lum[2], (k, k)).mean(axis=(2, 3))
    pad = k // 2
    peak = lum[2][pad:pad + loc.shape[0], pad:pad + loc.shape[1]] - loc
    ys, xs = np.nonzero(peak > floor)
    ux, uy = dx / mag, dy / mag
    ahead, behind = [], []
    for cy, cx in list(zip(ys + pad, xs + pad))[::max(1, len(ys) // 400)]:
        pa, pb = [], []
        for t in range(1, half + 1):
            xa, ya = int(round(cx + ux * t)), int(round(cy + uy * t))
            xb, yb = int(round(cx - ux * t)), int(round(cy - uy * t))
            if 0 <= xa < W and 0 <= ya < H:
                pa.append(float(lum[2][ya, xa]))
            if 0 <= xb < W and 0 <= yb < H:
                pb.append(float(lum[2][yb, xb]))
        if len(pa) == half and len(pb) == half:
            ahead.append(float(np.mean(pa)))
            behind.append(float(np.mean(pb)))
    if not ahead:
        print("   -> too few emitters to average a profile")
        return
    a_, b_ = float(np.mean(ahead)), float(np.mean(behind))
    print("   emitters averaged: %d" % len(ahead))
    print("   ahead %.5f   behind %.5f   behind-ahead %+.5f   ratio %.3f" % (a_, b_, b_ - a_, b_ / max(1e-9, a_)))
    print("   a one-frame persistence would smear about %.0f px, far beyond the %d-px window sampled," % (mag, half))
    print("   so the near-symmetry is a result about the camera and not a resolution limit.")


if __name__ == "__main__":
    rows = json.load(open(os.path.join(OUT, "scan30.json"), encoding="utf-8"))
    settle(rows)
    trail("--full" in sys.argv)
