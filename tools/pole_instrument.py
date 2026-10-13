"""
A better instrument: locate the pole by position and measure every pixel of it, not the brightest column per row.

**Why the old statistic could not answer the question.** It took, for each row below the core, the **brightest column**
and read its hue. That column is a mixture of the pole, the wall behind it and the sky, and which of them dominates
changes with brightness -- so the figure moved for reasons unrelated to the pole's colour. The evidence that it was
the wrong instrument, rather than the mechanisms being inert: **three independent levers** (the glow amount,
`halation_source_colour`, `shadow_depth`) all left it at **exactly +0.0000**, and the chain's stage-by-stage figures
show the pole well below white, so the pixels carrying the measurement are not the highlights those mechanisms act on.

**What this does instead, and it is a different kind of measurement:**

  * **the pole is located as an object**, as the strongest **vertical** structure below the core -- a column whose
    brightness stands above its horizontal neighbours **consistently over many rows**, which is what a pole is and
    what a wall is not;
  * **its width is measured**, so the band is the pole and not a guess;
  * the **hue is then taken over all the pixels inside that band**, as the **median of the per-row medians**, so a few
    mixed pixels at the edges cannot set it;
  * and **the same measurement is taken on the surround** at a fixed offset, so "the pole is blue relative to
    something" becomes explicit -- a hue on its own says nothing without the local reference.

The reference scene is measured with it first, because the instrument must be able to state what the pole's colour
actually *is* before any mechanism can be judged on whether it preserves it.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "tools", "ntscrs_ref")
COEF = np.array([0.299, 0.587, 0.114], np.float32)


def locate_pole(img):
    """
    Find the pole as a vertical structure: the column band that stands above its neighbours over many rows.

    A pole is not the brightest thing in a row -- a bulb or a window can be -- it is the column that is consistently
    brighter than the columns beside it, all the way down. So the score is the **per-row contrast against the local
    neighbourhood, accumulated down the frame**, and the band that wins is the pole.
    """
    H, W = img.shape[:2]
    lum = img @ COEF
    # where the brightest core is, so "below" is measured from it
    k = max(5, H // 40)
    box = np.lib.stride_tricks.sliding_window_view(lum, (k, k)).mean(axis=(2, 3))
    cy, cx = np.unravel_index(int(np.argmax(box)), box.shape)
    cy += k // 2
    cx += k // 2
    # score each column by its consistent excess over a wide local mean, using the rows below the core
    rows = slice(min(H - 2, cy + 8), H)
    band = lum[rows]
    if band.shape[0] < 20:
        return cy, cx, 1, 1, np.zeros(band.shape[0], np.float32)
    wide = np.apply_along_axis(lambda r: np.convolve(r, np.ones(41) / 41, mode="same"), 1, band)
    excess = band - wide
    # a pole is a narrow column or a few: take the best single column, then grow while the excess stays positive
    score = excess.mean(axis=0)
    # restrict to near the core horizontally, so a window elsewhere in the frame cannot win
    lo, hi = max(0, cx - W // 3), min(W, cx + W // 3)
    if hi - lo < 5:
        lo, hi = 0, W
    j = lo + int(np.argmax(score[lo:hi]))
    left, right = j, j
    while left - 1 >= 0 and score[left - 1] > 0.3 * score[j]:
        left -= 1
    while right + 1 < W and score[right + 1] > 0.3 * score[j]:
        right += 1
    return cy, j, left, right, band[:, left:right + 1].mean(axis=1)


def pole_colour(img):
    """
    The pole's colour, measured over the whole band and over the rows where the band is actually visible.

    Returned as: the band's columns, the hue of the pole's own pixels, the hue of the surround at a fixed offset, and
    the number of rows used -- because a hue without its row count and its local reference is not a measurement.
    """
    H, W = img.shape[:2]
    cy, j, left, right, prof = locate_pole(img)
    lum = img @ COEF
    rows = np.arange(min(H - 1, cy + 8), H)
    if len(rows) < 20:
        return dict(cols=(left, right), pole=float("nan"), surround=float("nan"), rows=0)
    # only the rows where the band is brighter than the surrounding wall, which is where the pole is unambiguous
    band = lum[rows][:, left:right + 1].mean(axis=1)
    off = max(12, (right - left) * 4)
    c0, c1 = max(0, left - off), max(1, left - 3)
    d0, d1 = min(W - 1, right + 3), min(W, right + off)
    sur = np.concatenate([lum[rows][:, c0:c1], lum[rows][:, d0:d1]], axis=1).mean(axis=1)
    keep = band > sur + 0.02
    if keep.sum() < 10:
        keep = np.ones_like(band, dtype=bool)
    sel = rows[keep]
    pole_hue = np.median([float((img[r, left:right + 1, 2] - img[r, left:right + 1, 0]).mean())
                          for r in sel])
    sur_hue = np.median([float(np.concatenate([
        img[r, c0:c1, 2] - img[r, c0:c1, 0], img[r, d0:d1, 2] - img[r, d0:d1, 0]]).mean())
        for r in sel])
    pole_con = float(np.median([float(lum[r, left:right + 1].mean()
                                     - np.concatenate([lum[r, c0:c1], lum[r, d0:d1]]).mean())
                                for r in sel]))
    return dict(cols=(left, right), pole=float(pole_hue), surround=float(sur_hue),
                excess=float(pole_hue - sur_hue), contrast=pole_con, rows=int(keep.sum()))


src = np.asarray(Image.open(os.path.join(OUT, "refscene.png")).convert("RGB"),
                 np.float32) / 255.0
print("   THE INSTRUMENT ON THE REFERENCE SCENE -- what the pole's colour actually is")
r = pole_colour(src)
print("      the pole's band is columns %d..%d (%d px wide)" % (r["cols"][0], r["cols"][1],
                                                                r["cols"][1] - r["cols"][0] + 1))
print("      measured over %d rows below the core" % r["rows"])
print("      the pole's own hue      B-R %+.4f" % r["pole"])
print("      the surround's hue      B-R %+.4f" % r["surround"])
print("      the pole's EXCESS over its surround  %+.4f   <- this is the quantity a mechanism must preserve" % r["excess"])
print("      the pole's contrast over its surround %.4f" % r["contrast"])
print()
print("   and the same on the earlier candidates, to show the instrument can tell subjects apart:")
for name in ("refscene.png", "raw_lamp_arena.jpg", "clean_source.jpg"):
    p = os.path.join(OUT, name)
    if not os.path.exists(p):
        continue
    a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0
    q = pole_colour(a)
    print("      %-22s band %3d..%3d, pole hue %+.4f, surround %+.4f, excess %+.4f, %d rows"
          % (name, q["cols"][0], q["cols"][1], q["pole"], q["surround"], q["excess"], q["rows"]))
