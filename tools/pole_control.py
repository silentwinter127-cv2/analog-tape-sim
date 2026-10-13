"""
A positive control for the pole detector, before it is pointed at any photograph.

**Why this comes first.** The detector has failed twice and neither failure was noticed until later: once it read a
diluted statistic and reported a line too faint to see, once it locked onto a bright vertical edge and reported that as
the pole. **Both produced confident numbers.** The discipline that caught the head-switching band and the `source_glow`
magnitude earlier in this work -- *check the instrument against a case where the answer is known* -- was not applied
here, and this is that check.

**The control is a synthetic frame whose pole is known in every respect:**

    a wall of known brightness and colour, with a vertical pole of known width and known hue, running down the frame

so the detector's **location**, **width** and **colour** can each be compared against an answer rather than against
another measurement. Two further cases are included because they are what the detector got wrong:

  * **a bright vertical edge** on the wall, of the same width as the pole and far from it -- the feature the detector
    locked onto on the real renders. A detector that cannot tell an edge from a pole will pick whichever scores
    higher, and the control says which it picked;
  * **no pole at all** -- a wall with nothing on it. The correct answer there is **"not found"**, and a detector that
    invents a pole is worse than one that misses one.

The detector must pass all three before any figure it produces about a render is treated as evidence.
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from atsim.core.signal import lowpass                                      # noqa: E402

COEF = np.array([0.299, 0.587, 0.114], np.float32)
H, W = 384, 452

# --- the scene builder.  A wall, a pole, and optionally a bright vertical edge elsewhere on the wall.
WALL_RGB = (0.10, 0.14, 0.20)      # a dark blue-grey wall: hue B-R +0.10
POLE_RGB = (0.55, 0.62, 0.78)      # a blue-white pole:        hue B-R +0.23
EDGE_RGB = (0.70, 0.70, 0.70)      # a neutral bright edge


def scene(pole_col, pole_w, edge_col=None, edge_w=0):
    a = np.zeros((H, W, 3), np.float32)
    a[..., :] = WALL_RGB
    if pole_w > 0:
        a[:, pole_col:pole_col + pole_w] = POLE_RGB
    if edge_col is not None and edge_w > 0:
        a[:, edge_col:edge_col + edge_w] = EDGE_RGB
    # a bright core at the top, which the detector uses as its origin for "below"
    a[0:26, W // 2 - 22:W // 2 + 22] = (0.98, 0.96, 0.90)
    return np.clip(a, 0, 1).astype(np.float32)


def find_pole(img, min_rows=40, search=None, min_score=0.01):
    """
    Locate a pole: the column band whose brightness exceeds its horizontal neighbours **consistently down the frame**.

    `search` optionally restricts the columns examined, which is the constraint that stops a bright edge elsewhere on
    the wall from winning.  It is a parameter rather than a hard-coded assumption so that the control can show what
    happens both with and without it -- which is the difference between a detector and a guess.
    """
    lum = img @ COEF
    k = max(5, H // 40)
    box = np.lib.stride_tricks.sliding_window_view(lum, (k, k)).mean(axis=(2, 3))
    cy, cx = np.unravel_index(int(np.argmax(box)), box.shape)
    cy += k // 2
    cx += k // 2
    start = min(H - 2, cy + 8)
    band = lum[start:H]
    if band.shape[0] < min_rows:
        return None
    wide = np.apply_along_axis(lambda r: np.convolve(r, np.ones(41) / 41, mode="same"), 1, band)
    score = (band - wide).mean(axis=0)
    if search is None:
        lo, hi = max(0, cx - W // 3), min(W, cx + W // 3)
        if hi - lo < 5:
            lo, hi = 0, W
    else:
        lo, hi = search
    j = lo + int(np.argmax(score[lo:hi]))
    if score[j] <= min_score:
        return None
    left = right = j
    while left - 1 >= 0 and score[left - 1] > 0.3 * score[j]:
        left -= 1
    while right + 1 < W and score[right + 1] > 0.3 * score[j]:
        right += 1
    off = max(12, (right - left) * 4)
    c0, c1 = max(0, left - off), max(1, left - 3)
    d0, d1 = min(W - 1, right + 3), min(W, right + off)
    pole_lum = band[:, left:right + 1].mean(axis=1)
    sur_lum = np.concatenate([band[:, c0:c1], band[:, d0:d1]], axis=1).mean(axis=1)
    keep = pole_lum > sur_lum + 0.02
    if keep.sum() < min_rows:
        return None
    # `keep` is built from `band`, whose first row is `start`; the caller indexes it with `rows`, so the two are
    # returned together and cannot fall out of step.
    return dict(left=int(left), right=int(right), width=int(right - left + 1), keep=keep,
                c0=int(c0), c1=int(c1), d0=int(d0), d1=int(d1), start=int(start))


def measure(img, **kw):
    r = find_pole(img, **kw)
    if r is None:
        return None
    rows = np.arange(r["start"], r["start"] + len(r["keep"]))
    sel = rows[r["keep"]]
    if len(sel) < 20:
        return None
    lum = img @ COEF
    pole = float(np.median([float((img[q, r["left"]:r["right"] + 1, 2]
                                   - img[q, r["left"]:r["right"] + 1, 0]).mean()) for q in sel]))
    sur = float(np.median([float(np.concatenate([
        img[q, r["c0"]:r["c1"], 2] - img[q, r["c0"]:r["c1"], 0],
        img[q, r["d0"]:r["d1"], 2] - img[q, r["d0"]:r["d1"], 0]]).mean()) for q in sel]))
    con = float(np.median([float(lum[q, r["left"]:r["right"] + 1].mean()
                                 - np.concatenate([lum[q, r["c0"]:r["c1"]],
                                                   lum[q, r["d0"]:r["d1"]]]).mean()) for q in sel]))
    return dict(band=(r["left"], r["right"]), width=r["width"], n=int(len(sel)),
                pole=pole, surround=sur, excess=pole - sur, contrast=con)


POLE_COL, POLE_W = 200, 6
print("   THE POSITIVE CONTROL -- the answers are known by construction")
print("      the wall's hue is B-R %+.4f;  the pole's is %+.4f" % (WALL_RGB[2] - WALL_RGB[0],
                                                                 POLE_RGB[2] - POLE_RGB[0]))
print("      the pole is at columns %d..%d, width %d" % (POLE_COL, POLE_COL + POLE_W - 1, POLE_W))
print()

CASES = [
    ("a pole on a plain wall", scene(POLE_COL, POLE_W), None),
    ("a pole, plus a bright edge at column 330", scene(POLE_COL, POLE_W, 330, POLE_W), None),
    ("a wall with no pole at all", scene(POLE_COL, 0), "expect NOT FOUND"),
]
for label, img, note in CASES:
    print("   --- %s%s ---" % (label, ("  (%s)" % note) if note else ""))
    for tag, kw in (("unconstrained", {}), ("constrained to +/-40 of the pole", {"search": (POLE_COL - 40, POLE_COL + 40)})):
        q = measure(img, **kw)
        if q is None:
            print("      %-30s NOT FOUND" % tag)
            continue
        ok_w = "OK" if q["width"] == POLE_W else "WRONG (expected %d)" % POLE_W
        ok_c = "OK" if abs(q["band"][0] - POLE_COL) <= 2 else "WRONG (expected %d)" % POLE_COL
        ok_h = "OK" if abs(q["pole"] - (POLE_RGB[2] - POLE_RGB[0])) < 0.05 else \
            "WRONG (expected %+.3f)" % (POLE_RGB[2] - POLE_RGB[0])
        print("      %-30s band %3d..%3d  width %2d %-22s column %-22s hue %+.4f %s"
              % (tag, q["band"][0], q["band"][1], q["width"], ok_w, ok_c, q["pole"], ok_h))
    print()

print("   what the control decides:")
print("      if the unconstrained detector picks the edge in case 2, it cannot be used on a photograph that has one;")
print("      if it invents a pole in case 3, it is worse than useless;")
print("      and the constrained form is only acceptable if it still finds a pole that moved.")
