"""
Measure the head-switching band on real VHS material.

The band exists because a VHS machine has two video heads on a rotating drum and they
cannot both be reading at once. At the end of each field the active head runs off the end of
its track and the machine switches to the other one, and during the changeover neither head
is reading the track it should be. The result is a strip at the bottom of the picture where
the signal is not the picture: it is noise, or the opposite field, or the wrong part of the
track.

What the strip *looks* like is therefore specific and worth measuring rather than guessing:
its height in lines, whether its edges are hard or soft, how its level and noise compare
with the picture above it, and whether there is a step -- a horizontal discontinuity at the
boundary, because the misread lines are not horizontally aligned with the correct ones.

Measured per frame on several frames, and reported separately for the lower few rows so a
graded band can be distinguished from a hard-edged one.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atsim.io.imagecodec import load_image  # noqa: E402

LUMA = np.array([0.299, 0.587, 0.114], np.float32)


def profile(img: np.ndarray) -> dict:
    g = np.clip(np.asarray(img, np.float32), 0, 1)
    if g.ndim == 3:
        g = g @ LUMA
    h = g.shape[0]
    # Per-row scalars.  `np.diff` along an axis returns a 2-D array, so the along-row
    # statistic has to be reduced to one number per row before anything can be compared
    # with another row -- broadcasting a (rows, cols) array against a (rows,) one is what
    # made the first version fail with "operands could not be broadcast together".
    row_mean = g.mean(axis=1)
    row_std = g.std(axis=1)
    noise = np.abs(np.diff(g, 2, axis=1)).mean(axis=1)
    # Differencing along the *row* axis gives one fewer value than there are rows.  That
    # mismatch is what actually raised "operands could not be broadcast together": the two
    # per-row arrays were combined with `|` while their lengths differed by one.  Pad it
    # back to the row count, with the first entry copied, so the arrays align.
    step_raw = np.abs(np.diff(g, axis=0)).mean(axis=1)
    step = np.concatenate([step_raw[:1], step_raw]) if step_raw.size else \
        np.zeros(g.shape[0], np.float32)

    # The band's extent: compare each row's noise with the picture's own median. Rows in
    # the band are markedly noisier or markedly offset.
    med_noise = float(np.median(noise))
    med_step = float(np.median(step))
    hot = (noise > med_noise * 1.8) | (step > med_step * 3.0)
    # look only at the bottom third, where the band lives
    tail = hot[int(h * 0.66):]
    band_rows = int(tail.sum())
    return {
        "height": h,
        "band_rows": band_rows,
        "band_frac": band_rows / max(1, h),
        "median_noise": med_noise,
        "band_noise": float(np.median(noise[int(h * 0.9):])) if h > 10 else 0.0,
        "median_step": med_step,
        "boundary_step": float(np.max(step[int(h * 0.8):])) if h > 10 else 0.0,
        "row_mean_tail": float(np.mean(row_mean[int(h * 0.94):])),
        "row_mean_mid": float(np.mean(row_mean[int(h * 0.5):int(h * 0.6)])),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="head-switching band on real material")
    ap.add_argument("paths", nargs="*", default=None)
    args = ap.parse_args(argv)
    paths = args.paths or sorted(glob.glob("tools/ref13/*.png")) \
        + sorted(glob.glob("tools/ref12/*.png"))
    if not paths:
        print("no frames found")
        return 1

    print("%-24s %6s %9s %10s %11s %10s %10s"
          % ("frame", "rows", "band rows", "band %", "tail noise", "mid noise", "boundary"))
    print("-" * 92)
    for p in paths:
        try:
            m = profile(load_image(p))
        except Exception as exc:                      # noqa: BLE001 - reported
            print("%-24s could not read: %s" % (os.path.basename(p), str(exc)[:40]))
            continue
        print("%-24s %6d %9d %9.1f%% %11.5f %10.5f %10.5f"
              % (os.path.basename(p), m["height"], m["band_rows"], 100 * m["band_frac"],
                 m["band_noise"], m["median_noise"], m["boundary_step"]))

    print()
    print("Interpretation: a hard-edged band shows a small number of large `boundary`")
    print("steps and a clean separation in noise; a graded one shows noise rising")
    print("gradually with no single step.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
