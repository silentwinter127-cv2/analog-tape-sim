"""
Measure a degradation fingerprint, so simulation and real material can be compared.

Five quantities, chosen because each one separates failure modes that look similar in
a still:

  * `hf`      -- high-frequency energy, normalised by local contrast so it survives a
                 change of scene.  Separates "buried in noise" from "merely soft".
  * `sat`     -- mean saturation.  Separates a colour *loss* (falls) from a colour
                 *error* (stays, while hue moves).
  * `hue_spread` -- circular spread of hue weighted by saturation.  A mechanism that
                 rotates hue on some lines raises this; one that desaturates does not.
  * `row_step` -- median absolute difference in row-mean luma between adjacent rows,
                 which is where a whole-line fault shows up and per-pixel noise does
                 not.
  * `along/across curvature` -- as used for the stripe work: within-row against
                 across-row second derivative of a demodulated grating phase.  On
                 material with no grating this falls back to a broad-band measure of
                 how much structure varies sideways within a row.

Reported for a real clip and for a simulated render, side by side, so the difference is
a table rather than an impression.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atsim.io.imagecodec import load_image  # noqa: E402

LUMA = np.array([0.299, 0.587, 0.114], np.float32)


def fingerprint(img: np.ndarray) -> dict:
    img = np.clip(np.asarray(img, np.float32), 0.0, 1.0)
    g = img @ LUMA
    mx, mn = img.max(axis=2), img.min(axis=2)
    sat_map = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)

    # high frequency normalised by local contrast: dividing by the local standard
    # deviation keeps this comparable between a bright logo and a dark scene
    hf_raw = np.abs(np.diff(g, 2, axis=1)).mean() + np.abs(np.diff(g, 2, axis=0)).mean()
    local = float(g.std()) + 1e-6
    hf = float(hf_raw / local)

    # circular hue spread, weighted by how colourful each pixel is
    r, gg, b = img[..., 0], img[..., 1], img[..., 2]
    hue = np.arctan2(np.sqrt(3.0) * (gg - b), 2.0 * r - gg - b)
    w = sat_map
    if w.sum() > 1e-6:
        c = np.cos(hue) * w
        s = np.sin(hue) * w
        rbar = np.sqrt((c.mean()) ** 2 + (s.mean()) ** 2)
        hue_spread = float(np.sqrt(max(0.0, -2.0 * np.log(max(1e-9, rbar)))))
    else:
        hue_spread = 0.0

    # whole-line structure: how much the row-mean level jumps between neighbours
    rm = g.mean(axis=1)
    row_step = float(np.median(np.abs(np.diff(rm)))) if rm.size > 2 else 0.0

    # within-row vs across-row structure, the curvature pair from the stripe work
    along = float(np.median(np.abs(np.diff(g, 2, axis=1))))
    across = float(np.median(np.abs(np.diff(rm, 2)))) if rm.size > 3 else 0.0

    return {
        "mean": float(img.mean()),
        "sat": float(sat_map.mean()),
        "hf": hf,
        "hue_spread": hue_spread,
        "row_step": row_step,
        "along": along,
        "across": across,
        "dark": float((g < 0.12).mean()),
    }


def report(name: str, img: np.ndarray) -> dict:
    fp = fingerprint(img)
    print("%-34s mean %.3f  sat %.3f  hf %6.3f  hue_spread %.3f  row_step %.4f  "
          "along %.5f  across %.5f  dark %.1f%%"
          % (name, fp["mean"], fp["sat"], fp["hf"], fp["hue_spread"], fp["row_step"],
             fp["along"], fp["across"], 100 * fp["dark"]))
    return fp


if __name__ == "__main__":
    print("=== reference material ===")
    fps = {}
    for d, prefix in (("tools/ref11", "vcd"), ("tools/ref9", "svhs"), ("tools/ref10", "clip")):
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".png"):
                fps[fn] = report("%s/%s" % (prefix, fn), load_image(os.path.join(d, fn)))
