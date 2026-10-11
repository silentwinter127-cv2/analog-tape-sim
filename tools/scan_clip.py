"""
Locate the unstable stretches in a long capture.

Sampling frames on a grid and measuring how much each frame's row-mean profile jumps
finds the sections where the picture is unstable, which is where the TBC was being
toggled. Looking at a few arbitrary frames would be luck; this says where to look.

The metric is deliberately about *rows*: horizontal displacement changes what each row
contains, so a frame being displaced relative to its neighbours shows up as a large
change in the row-mean profile, while scene motion changes it much less abruptly.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atsim.io.imagecodec import load_image  # noqa: E402

LUMA = np.array([0.299, 0.587, 0.114], np.float32)
FFMPEG = "ffmpeg"


def sample_frames(path: str, step: float, out_dir: str, scale: int = 360,
                  count: int | None = None) -> list[tuple[float, str]]:
    """Pull one frame every `step` seconds, downscaled, into `out_dir`."""
    os.makedirs(out_dir, exist_ok=True)
    dur = float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", path],
        capture_output=True, text=True).stdout.strip() or 0.0)
    n = int(dur / step) if count is None else count
    made = []
    for k in range(n):
        t = k * step
        if t >= dur:
            break
        out = os.path.join(out_dir, "t%07.1f.png" % t)
        if not os.path.exists(out):
            subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-ss", "%.2f" % t, "-i", path,
                 "-frames:v", "1", "-vf", "scale=%d:-2" % scale, out],
                capture_output=True)
        if os.path.exists(out):
            made.append((t, out))
    return made


def instability(img: np.ndarray) -> dict:
    g = np.clip(np.asarray(img, np.float32), 0, 1) @ LUMA
    rm = g.mean(axis=1)
    return {
        "row_jump": float(np.median(np.abs(np.diff(rm)))),
        "row_var": float(rm.std()),
        "hf": float(np.abs(np.diff(g, 2, axis=1)).mean()),
        "flat": float(rm.std() < 0.01),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="scan a capture for unstable stretches")
    ap.add_argument("clip")
    ap.add_argument("--step", type=float, default=60.0, help="seconds between samples")
    ap.add_argument("--out", default="tools/scan")
    ap.add_argument("--scale", type=int, default=360)
    ap.add_argument("--top", type=int, default=14)
    args = ap.parse_args(argv)

    frames = sample_frames(args.clip, args.step, args.out, args.scale)
    print("sampled %d frames from %s" % (len(frames), args.clip))
    rows = []
    for t, path in frames:
        m = instability(load_image(path))
        rows.append((t, m))
    # consecutive-frame change in the row profile is what "unstable" means here
    print()
    print("%9s %10s %10s %9s %6s" % ("time", "row_jump", "row_var", "hf", "flat"))
    ordered = sorted(rows, key=lambda r: -r[1]["row_jump"])
    shown = {t for t, _ in ordered[:args.top]}
    for t, m in rows:
        mark = "  <- unstable" if t in shown else ""
        print("%8.0fs %10.4f %10.4f %9.4f %6s%s"
              % (t, m["row_jump"], m["row_var"], m["hf"], m["flat"], mark))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
