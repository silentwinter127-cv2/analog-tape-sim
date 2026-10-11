"""
Measure per-row horizontal displacement on a real capture.

This is the quantity the per-line time-base mechanisms model, so measuring it on real
material is what says whether their amplitudes mean anything.

Three decisions that the earlier, wrong version of this measurement got wrong:

  * **rows, not row means.**  A scan based on how much a frame's row-mean profile
    changed found the karaoke sections, but those move because *scrolling lyrics*
    change the content. Content motion is not time-base error, and the two have to be
    separated. Horizontal displacement of an edge is insensitive to what the scene is
    doing.
  * **fields, not frames.** A time-base corrector works field by field, and an
    interlaced frame interleaves two fields that were displaced independently. A
    frame-level measurement averages the two and hides the very effect being looked
    for.
  * **sub-pixel.** Displacement error is often well under one pixel, and rounding to
    integers would report zero for most rows.

The estimate per row comes from the position of the strongest vertical edge, found by
parabolic interpolation of the gradient peak, tracked along the row.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LUMA = np.array([0.299, 0.587, 0.114], np.float32)


def extract_field(path: str, t: float, out_path: str, width: int = 720) -> bool:
    """One field at native width, so no resampling sits between us and the signal."""
    r = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-ss", "%.3f" % t, "-i", path,
         "-frames:v", "1", "-vf", "format=gray,scale=%d:-2,separatefields" % width,
         "-frames:v", "1", out_path],
        capture_output=True, text=True)
    return os.path.exists(out_path)


def load_gray(path: str) -> np.ndarray:
    from atsim.io.imagecodec import load_image
    img = load_image(path)
    if img.ndim == 3:
        return (img @ LUMA).astype(np.float32)
    return img.astype(np.float32)


def row_shifts_correlation(gray: np.ndarray, max_shift: int = 12
                           ) -> tuple[np.ndarray, np.ndarray]:
    """
    Displacement of each row relative to the field's own average, by correlation.

    Tracking a single edge fails on real material: the strongest gradient in one row
    is a different object from the strongest gradient in the next, so the "track" jumps
    across the frame.  Measured that way the spread came out at 235 px against a frame
    715 px wide, which is the tracker reporting scene content, not displacement.

    Correlating each row against the average of its neighbours is robust to that: the
    question is only how far this row has moved relative to what surrounds it, and the
    answer is the lag that maximises the match, refined by a parabolic fit.
    """
    from scipy.ndimage import gaussian_filter1d
    g = gaussian_filter1d(gray, 0.7, axis=1)
    h, w = g.shape
    ref = gaussian_filter1d(g, 4.0, axis=0)          # local vertical average
    lags = np.arange(-max_shift, max_shift + 1)
    shifts = np.full(h, np.nan, np.float32)
    quality = np.zeros(h, np.float32)
    for r in range(h):
        row = g[r] - g[r].mean()
        tgt = ref[r] - ref[r].mean()
        if row.std() < 0.01:                          # flat row: nothing to align
            continue
        scores = np.array([float(np.dot(row, np.roll(tgt, int(s)))) for s in lags])
        k = int(np.argmax(scores))
        quality[r] = scores[k] / max(1e-9, np.abs(scores).mean())
        best = float(lags[k])
        if 0 < k < len(lags) - 1:
            y0, y1, y2 = scores[k - 1], scores[k], scores[k + 1]
            den = y0 - 2 * y1 + y2
            if abs(den) > 1e-9:
                best -= 0.5 * (y2 - y0) / den
        shifts[r] = best
    return shifts, quality


def displacement_stats(gray: np.ndarray) -> dict:
    """
    Spread of the per-row displacement, and how it varies along the field.

    `std` is the whole-field spread: how far the rows are displaced from each other,
    which is the quantity the row-time-base mechanism's amplitude should correspond to.
    `line_step` is the median change between adjacent rows: a small value with a large
    spread means the displacement varies *smoothly*, which is what bends a vertical
    edge, whereas a large value means it is ragged.
    """
    shifts, quality = row_shifts_correlation(gray)
    ok = np.isfinite(shifts) & (quality > 1.15)
    if ok.sum() < 16:
        return {"rows": int(ok.sum()), "std": float("nan"), "line_step": float("nan"),
                "range": float("nan"), "curvature": float("nan")}
    p = shifts[ok]
    d = np.diff(p)
    d2 = np.diff(p, 2) if p.size > 3 else np.array([0.0])
    return {
        "rows": int(ok.sum()),
        "std": float(p.std()),
        "line_step": float(np.median(np.abs(d))),
        "range": float(p.max() - p.min()),
        "curvature": float(np.median(np.abs(d2))),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="per-row displacement on a real capture")
    ap.add_argument("clip")
    ap.add_argument("--times", default="120,600,1000,1380,2000")
    ap.add_argument("--out", default="tools/scan/fields")
    ap.add_argument("--width", type=int, default=720)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    print("%9s %6s %9s %10s %8s %11s" % ("time", "rows", "std(px)", "line_step",
                                         "range", "curvature"))
    for part in args.times.split(","):
        t = float(part)
        p = os.path.join(args.out, "f%07.1f.png" % t)
        if not extract_field(args.clip, t, p, args.width):
            print("%8.0fs  field extraction failed" % t)
            continue
        st = displacement_stats(load_gray(p))
        print("%8.0fs %6d %9.3f %10.3f %8.2f %11.4f"
              % (t, st["rows"], st["std"], st["line_step"], st["range"],
                 st["curvature"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
