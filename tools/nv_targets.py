"""
Native-resolution targets for the DV NightShot work.

**Why this file exists rather than reusing `tools/native_resolution.py`.**  That file's real references are the two
frames in the attachment store, and both are **691x520** -- which is why every figure in this work was taken at
533x400 and why REVIEW.md section 4.1 had to be written.  The clip itself is **1440x1080**, so the targets can be
taken at the size the effect survives; this file does that and prints the resolution beside every number.

**It measures only.**  Nothing is retuned here: the point is to state, at native resolution and on this round's
inputs, what the camcorder actually produced, so that whatever is changed next is aimed at what is there.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

COEF = np.array([0.299, 0.587, 0.114], np.float32)
FRAMES = os.path.join(ROOT, "tools", "nightshot", "frames")
NEW_REF = os.path.join(ROOT, "tools", "nightshot", "ref_osd_1440.png")
NEW_CASE = os.path.join(ROOT, "tools", "nightshot", "case_new_2048.png")


def native(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), np.float32) / 255.0


def crop_osd(a: np.ndarray, frac: float = 0.20) -> np.ndarray:
    """The camcorder's own overlay sits in the top fifth; cropping it keeps text out of every statistic."""
    return a[int(a.shape[0] * frac):]


def measure(x: np.ndarray) -> dict:
    """
    The quantities the footage's own segments were characterised by, plus the ones the highlight work needed.

    `hue` is the circular spread of the chroma angle, which is what "the colour is not uniform" means numerically;
    `g` is the green channel's share of the total, which is what "it reads green" means; and the bands are reported
    with their pixel counts so a band measured on almost nothing cannot be mistaken for a measurement.
    """
    lum = np.clip(x @ COEF, 0.0, 1.0)
    r, g, b = (float(x[..., i].mean()) for i in range(3))
    u, v = x[..., 0] - x[..., 1], x[..., 2] - x[..., 1]
    m = np.sqrt(u * u + v * v)
    ang = np.arctan2(v, u)
    spreads, counts = [], []
    for lo, hi in ((0.0, 0.25), (0.25, 0.6), (0.6, 0.85), (0.85, 1.01)):
        k = (lum >= lo) & (lum < hi) & (m > 0.02)
        counts.append(int(k.sum()))
        if k.sum() < 50:
            spreads.append(float("nan"))
            continue
        aa = ang[k]
        spreads.append(float(np.angle(np.exp(1j * (aa - np.angle(np.exp(1j * aa).mean())))).std()))
    return dict(
        res="%dx%d" % (x.shape[1], x.shape[0]),
        primary="#%02X%02X%02X" % tuple(int(round(c * 255)) for c in (r, g, b)),
        mean=float(lum.mean()), std=float(lum.std()),
        p1=float(np.quantile(lum, 0.01)), p99=float(np.quantile(lum, 0.99)),
        below05=float((lum < 0.05).mean()), blown=float((lum >= 0.98).mean()),
        #: **`g_share` is `mean(G) / (mean(R)+mean(G)+mean(B))`, and the first two versions were both wrong.**
        #: The first divided by `x.sum()` -- the *entire array's* total -- and the second divided by the same thing
        #: again after being told the denominator was wrong, so it still read **0.0000** while looking like a
        #: considered figure.  `x.sum()` scales with the pixel count, so the quantity was 1/npixels and not a share
        #: at all; the third version is a ratio of channel means and lands where the calibration's own target does
        #: (nv1 -> 0.43 against the documented 0.44-0.58).  **A figure that is wrong by a constant factor is
        #: invisible in a column of its own**, which is the whole reason this file prints it beside a known target.
        g_share=float(g / max(1e-9, r + g + b)),
        rg=r / max(1e-9, g), bg=b / max(1e-9, g),
        hue=float(np.angle(np.exp(1j * (ang[m > 0.02] -
                                       np.angle(np.exp(1j * ang[m > 0.02]).mean())))).std()) if (m > 0.02).sum() else 0.0,
        spreads=spreads, counts=counts,
    )


def line(tag: str, s: dict, note: str = "") -> None:
    print("   %-30s %-10s %-8s %-8.4f %-8.4f %-8.4f %-8.5f %-8.4f %-7.4f %-7.4f %s"
          % (tag, s["res"], s["primary"], s["mean"], s["std"], s["p1"], s["blown"], s["g_share"],
             s["rg"], s["bg"], note))


HEAD = ("   %-30s %-10s %-8s %-8s %-8s %-8s %-8s %-8s %-7s %-7s %s"
        % ("", "resolution", "primary", "mean", "std", "p1", "blown", "G share", "R/G", "B/G", "note"))


def main() -> int:
    print("=" * 132)
    print("  DV NIGHTSHOT -- the targets, at the clip's own 1440x1080")
    print("=" * 132)
    print(HEAD)

    groups: dict[str, list[dict]] = {}
    for path in sorted(glob.glob(os.path.join(FRAMES, "nv*.png"))):
        name = os.path.basename(path)[:-4]
        seg = name.rsplit("_", 1)[0]
        s = measure(crop_osd(native(path)))
        groups.setdefault(seg, []).append(s)
        line(name, s, "OSD cropped")

    print()
    print("   --- per segment (mean of the frames in it) ---")
    for seg, rows in sorted(groups.items()):
        agg = {k: float(np.mean([r[k] for r in rows])) for k in
               ("mean", "std", "p1", "blown", "g_share", "rg", "bg", "hue", "below05")}
        sp = np.array([r["spreads"] for r in rows], np.float64)
        cnt = np.array([r["counts"] for r in rows], np.int64)
        print("   %-14s n=%d  mean %.4f  std %.4f  p1 %.4f  <0.05 %.4f  blown %.5f  G %.4f  R/G %.4f  B/G %.4f"
              % (seg, len(rows), agg["mean"], agg["std"], agg["p1"], agg["below05"], agg["blown"],
                 agg["g_share"], agg["rg"], agg["bg"]))
        print("   %-14s      hue spread (pixels): %s"
              % ("", "  ".join("%.3f(%d)" % (v, c) for v, c in zip(sp.mean(axis=0), cnt.mean(axis=0).astype(int)))))

    print()
    print("   --- this round's two inputs ---")
    for path, tag, frac in ((NEW_REF, "the new 1440x1080 reference", 0.20),
                            (NEW_CASE, "the new case photograph", 0.0)):
        a = native(path)
        line(tag + " (input)", measure(crop_osd(a, frac) if frac else a),
             "the camcorder's own output, OSD cropped" if frac else "a visible-light photograph")
    print()
    print("   **every hue-spread figure above is meaningless without the resolution beside it** -- REVIEW.md section 4.1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
