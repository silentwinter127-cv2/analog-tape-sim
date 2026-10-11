"""
Measure field-to-field wander, which the existing tracker is structurally blind to.

Every tracker so far subtracted a local vertical average, because a uniform shift of a
field is not visible as a shift. That makes them blind to exactly the component the
time-base corrector has to be removing: on the real capture it cuts the per-row spread
about 8.8x, and the within-field fast component there is small, so the shared term is the
only candidate left.

This measures the shared term instead. Each field is cross-correlated against a reference
over a window of shifts, giving one number per field: how far the whole field sits from
where it was. Tracked across time that is the wander; compared between stretches it says
where the corrector was engaged.

Three things had to be right, each learned from a failed attempt:

  * rows, not row means -- a row-mean scan found the karaoke sections, which move because
    scrolling lyrics change the content;
  * correlation against a *reference*, not against a local average -- the local average is
    what removes the shared term;
  * fields, not frames -- a frame interleaves two fields displaced independently, and
    averaging them hides the effect being measured.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.measure_timebase import extract_field, load_gray  # noqa: E402


def field_shift(field: np.ndarray, ref: np.ndarray, max_shift: int = 24
                ) -> tuple[float, float]:
    """
    (shift, quality) of one field against a reference, by correlation of gradients.

    Gradients rather than intensities: a field's overall brightness varies for reasons
    that have nothing to do with time base, and differentiating removes that while
    keeping the structure that a horizontal displacement moves.
    """
    a = np.diff(field, axis=1)
    b = np.diff(ref, axis=1)
    h = min(a.shape[0], b.shape[0])
    a, b = a[:h], b[:h]
    a = a - a.mean()
    b = b - b.mean()
    if a.std() < 1e-6 or b.std() < 1e-6:
        return float("nan"), 0.0
    lags = np.arange(-max_shift, max_shift + 1)
    scores = np.array([float(np.sum(a * np.roll(b, int(s)))) for s in lags])
    k = int(np.argmax(scores))
    quality = scores[k] / max(1e-9, float(np.mean(np.abs(scores))))
    shift = float(lags[k])
    if 0 < k < len(lags) - 1:
        y0, y1, y2 = scores[k - 1], scores[k], scores[k + 1]
        den = y0 - 2 * y1 + y2
        if abs(den) > 1e-9:
            shift -= 0.5 * (y2 - y0) / den
    return shift, float(quality)


def within_field_spread(field: np.ndarray, block: int = 32) -> float:
    """
    How much the displacement varies *down* a single field.

    Split the field into horizontal bands, measure each band's shift against the first
    band, and take the spread of those shifts. This is the quantity a corrector acts on:
    it pulls lines towards a common reference, so a field whose bands disagree is one it
    can improve.
    """
    g = np.diff(field, axis=1)
    h = g.shape[0]
    ref_band = g[:block]
    ref_band = ref_band - ref_band.mean()
    if ref_band.std() < 1e-6:
        return float("nan")
    lags = np.arange(-16, 17)
    shifts = []
    for y in range(0, h - block + 1, block):
        band = g[y:y + block]
        band = band - band.mean()
        if band.std() < 1e-6:
            continue
        sc = np.array([float(np.sum(band * np.roll(ref_band, int(s)))) for s in lags])
        shifts.append(float(lags[int(np.argmax(sc))]))
    return float(np.std(shifts)) if len(shifts) > 3 else float("nan")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="track field-to-field wander")
    ap.add_argument("clip")
    ap.add_argument("--from", dest="start", type=float, default=960.0)
    ap.add_argument("--to", dest="stop", type=float, default=1200.0)
    ap.add_argument("--step", type=float, default=10.0)
    ap.add_argument("--out", default="tools/scan/wander")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    fields, times = [], []
    t = args.start
    while t < args.stop:
        p = os.path.join(args.out, "f%07.1f.png" % t)
        if extract_field(args.clip, t, p, 720):
            fields.append(load_gray(p))
            times.append(t)
        t += args.step
    if len(fields) < 6:
        print("too few fields extracted (%d)" % len(fields))
        return 0

    ref = fields[len(fields) // 2]
    print("%9s %10s %8s %14s" % ("time", "shift(px)", "quality", "within-field"))
    rows = []
    for t, f in zip(times, fields):
        s, q = field_shift(f, ref)
        w = within_field_spread(f)
        rows.append((t, s, q, w))
        flag = "  <- flat" if q < 1.3 else ""
        print("%8.0fs %10.2f %8.2f %14.3f%s" % (t, s, q, w, flag))

    ok = [(t, s, w) for t, s, q, w in rows if np.isfinite(s) and np.isfinite(w)]
    if len(ok) < 6:
        print("\ntoo few usable fields")
        return 0
    shifts = np.array([s for _t, s, _w in ok])
    within = np.array([w for _t, _s, w in ok])
    print()
    print("field-to-field wander (shared displacement):")
    print("  spread %.3f px, range %.1f px" % (shifts.std(), shifts.max() - shifts.min()))
    print("within-field variation:")
    print("  median %.3f px, mean %.3f px" % (float(np.median(within)), float(within.mean())))
    print()
    print("If within-field variation is far below the field-to-field wander, the corrector's")
    print("effect cannot be about the within-field component, and the wander is what to model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
