"""
Extract frames from the Sony NightShot reference clip and measure what makes it what it is.

**Why measurements first.** The same discipline that caught the head-switching band and the `source_glow` magnitude
applies here: a night-vision mode is a set of *specific* physical changes, and guessing which ones produces a filter
that looks plausible and reproduces none of them. So the reference clip is measured for the quantities each candidate
mechanism would move, and the mechanisms are then built against those numbers.

What a NightShot-family mode changes, physically, and therefore what to measure:

  * **the IR-cut filter is removed**, so the sensor sees near-infrared. That shows up as a **spectral** change -- the
    same scene rendered with a different channel weighting -- and the clip's own colour tells us what the camera did
    with it;
  * **the IR illuminator is on**, so brightness falls off with distance from the camera and is *independent of the
    scene's visible lighting* -- measurable as the ratio of near-field to far-field brightness against a visible-light
    frame;
  * **the display is monochrome green** in NightShot, so saturation should be near zero with a green cast;
  * **gain is raised and the shutter extended**, so noise rises. **Super NightShot** accumulates frames, so noise
    *falls* relative to a single frame while motion smears -- measurable as the temporal noise against the spatial
    noise;
  * **Color Slow Shutter** keeps colour and extends the shutter, so saturation is *not* removed and the smear is long.

So the report is: per-frame luma statistics, the chroma statistics, the noise level (high-frequency energy), the
near/far brightness profile, and the frame-to-frame difference -- the last being what separates an accumulating mode
from a single long exposure.
"""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "tools", "nightshot")
VIDEO = (r"C:\Users\Administrator\Downloads\索尼F717夜视测试.MOV00014.409916754.mp4")
FFMPEG = (r"E:\ffmpeg-6.1.1-essentials_build\ffmpeg-6.1.1-essentials_build\bin\ffmpeg.exe")
FFPROBE = (r"E:\ffmpeg-6.1.1-essentials_build\ffmpeg-6.1.1-essentials_build\bin\ffprobe.exe")
os.makedirs(OUT, exist_ok=True)
COEF = np.array([0.299, 0.587, 0.114], np.float32)

print("   the clip: %s" % os.path.basename(VIDEO))
print("      exists: %s, %.1f MB" % (os.path.exists(VIDEO),
                                     os.path.getsize(VIDEO) / 1e6 if os.path.exists(VIDEO) else 0))

if os.path.exists(VIDEO):
    r = subprocess.run([FFPROBE, "-v", "error", "-show_entries",
                        "stream=width,height,r_frame_rate,nb_frames,codec_name:format=duration",
                        "-of", "default=nw=1", VIDEO], capture_output=True, text=True, timeout=120)
    for line in r.stdout.strip().split("\n"):
        print("      %s" % line)

# extract a spread of frames as PNG so nothing is measured through a decoder twice
frame_dir = os.path.join(OUT, "frames")
os.makedirs(frame_dir, exist_ok=True)
if not os.listdir(frame_dir):
    subprocess.run([FFMPEG, "-v", "error", "-y", "-i", VIDEO, "-vf", "fps=4",
                    os.path.join(frame_dir, "f_%04d.png")], check=True, timeout=900)
frames = sorted(f for f in os.listdir(frame_dir) if f.endswith(".png"))
print("   extracted %d frame(s) at 4 fps into tools/nightshot/frames/" % len(frames))

if not frames:
    raise SystemExit("no frames extracted")

from PIL import Image                                                            # noqa: E402

imgs = []
for f in frames[:60]:
    a = np.asarray(Image.open(os.path.join(frame_dir, f)).convert("RGB"), np.float32) / 255.0
    imgs.append(a)
H, W = imgs[0].shape[:2]
print("   frame size %dx%d" % (W, H))


def stats(a):
    lum = a @ COEF
    mx = a.max(axis=-1)
    mn = a.min(axis=-1)
    sat = np.where(mx > 1e-5, (mx - mn) / np.maximum(mx, 1e-5), 0.0)
    # high-frequency energy along rows: the noise proxy, since a night-vision gain stage raises it
    hf = float(np.abs(np.diff(lum, axis=1)).mean())
    return dict(mean=float(lum.mean()), p50=float(np.median(lum)),
                p99=float(np.quantile(lum, 0.99)), blown=float((lum > 0.97).mean()),
                sat=float(sat.mean()), rb=float(a[..., 0].mean() - a[..., 2].mean()),
                gb=float(a[..., 1].mean() - a[..., 2].mean()), hf=hf)


s = [stats(a) for a in imgs]
print()
print("   %-8s %-9s %-9s %-9s %-9s %-9s %-9s %-9s %s"
      % ("frame", "mean", "p50", "p99", "blown", "sat", "R-B", "G-B", "hf(noise)"))
for i in (0, len(s) // 4, len(s) // 2, 3 * len(s) // 4, len(s) - 1):
    q = s[i]
    print("   %-8d %-9.4f %-9.4f %-9.4f %-9.5f %-9.4f %-+9.4f %-+9.4f %.5f"
          % (i, q["mean"], q["p50"], q["p99"], q["blown"], q["sat"], q["rb"], q["gb"], q["hf"]))

agg = {k: float(np.mean([q[k] for q in s])) for k in s[0]}
print()
print("   the clip's averages over %d frames:" % len(s))
for k, v in agg.items():
    print("      %-8s %.5f" % (k, v))

# frame-to-frame difference: what separates an accumulating mode from a single exposure
if len(imgs) > 1:
    d = [float(np.abs(imgs[i] - imgs[i - 1]).mean()) for i in range(1, min(len(imgs), 40))]
    print()
    print("   frame-to-frame difference: mean %.5f, min %.5f, max %.5f  (motion + noise, not separated yet)"
          % (float(np.mean(d)), float(np.min(d)), float(np.max(d))))
