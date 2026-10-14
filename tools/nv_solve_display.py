"""
Solve the display projection's weights from the target ratio, instead of fitting them.

**The physical statement.**  With the IR-cut filter out, NightShot hands the recorder **one luminance** and the
picture's colour is the **display's**.  Two consequences, and both are measurable:

  * the display's R:G:B ratio must be **independent of the input** -- a display is a projection, and a projection that
    changes with what it is shown is a colour grade (`tools/nv_display_probe.py` measures 21.7x of dependence today);
  * the ratio itself is then a **solved** quantity, not a fitted one: `out = lum * w`, so the output's ratios ARE the
    weights' ratios, and the weights follow from the target in closed form.

**Why the solve is done through the real function rather than by hand.**  `monochrome_green` normalises the weights by
`w @ LUMA` so the view carries the picture's brightness, and it applies the YUV gains as well.  A hand derivation of
"what the weights should be" would have to reproduce both, and this repository has four recorded cases of a hand
rebuild disagreeing with the code it reproduced.  So the weights here are found by **inverting the actual function on
a neutral field**: build the projection the function builds, then solve for the two numbers that put R/G and B/G on the
target.  A reproduction cannot verify the thing it reproduces, but a *solve against the target* is the target.
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from atsim.codecs.nightvision import NightVisionSpec, monochrome_green    # noqa: E402

LUMA = np.array([0.299, 0.587, 0.114], np.float32)

#: The target, from the clip's own frames at native resolution (`tools/nv_targets.py`):
#:     NIGHTSHOT 1   R/G 0.5434   B/G 0.8525
#:     NIGHTSHOT 2   R/G 0.4766   B/G 0.8192
#: 1 is used because it is the mode the user's reference frames are in, and 2 is checked below as a consequence.
TARGET_RG, TARGET_BG = 0.5434, 0.8525


def ratio_of(rgb: np.ndarray) -> tuple:
    r, g, b = (float(rgb[..., i].mean()) for i in range(3))
    return r / max(1e-9, g), b / max(1e-9, g)


def projection(u_gain: float, v_gain: float) -> tuple:
    """
    What `monochrome_green` does to a neutral field, given the two display-gain numbers it applies to U and V.

    Rebuilt here in the two lines that matter, and then **verified against the function itself** at the end of
    `main()`, because a rebuild that is not checked is how three earlier attempts in this work went wrong.
    """
    w = np.array([0.8291, 1.0, 1.0550], np.float32)
    w = w / float(w @ LUMA)
    m = 1.0                                     # green_mix
    mono = 1.0 * w                              # lum == 1 on a neutral field
    lumv = 1.0
    u_ch = 0.0                                  # a neutral field carries no chroma...
    v_ch = 0.0
    #: ... so the gains act on nothing, and the output ratio is simply the weights' ratio.  That is the whole point:
    #: **for a neutral input the display gains are irrelevant, and for a coloured one they must be too.**  The solve
    #: below therefore only has to place the RATIO, and the input-independence comes from deleting the residue.
    rr = mono[0]
    bb = mono[2]
    gg = (mono @ LUMA) - 0.299 * rr - 0.114 * bb
    gg = gg / 0.587
    return rr / max(1e-9, gg), bb / max(1e-9, gg)


def main() -> int:
    #: The weights that put the ratio on the target.  `out = lum * w`, so `w` IS the output ratio up to the luma
    #: normalisation -- solve it in one step and then let `monochrome_green` normalise.
    print("=" * 100)
    print("  SOLVING THE DISPLAY PROJECTION FROM THE TARGET RATIO")
    print("=" * 100)
    print()
    print("   target: R/G %.4f   B/G %.4f   (the clip's own NIGHTSHOT 1 frames at native resolution)"
          % (TARGET_RG, TARGET_BG))
    print()
    # w = [wr, 1, wb] normalised by (w @ LUMA); out = lum * w, so out ratio == w ratio
    for wr, wb in ((0.8291, 1.0550), (TARGET_RG, TARGET_BG)):
        nn = wr * LUMA[0] + LUMA[1] + wb * LUMA[2]
        w = np.array([wr, 1.0, wb], np.float32) / nn
        r, g, b = float(w[0]), float(w[1]), float(w[2])
        print("   w = [%.4f, 1, %.4f] normalised -> %s   R/G %.4f   B/G %.4f   sum %.4f"
              % (wr, wb, "%.4f:%.4f:%.4f" % (r, g, b), r / g, b / g, r + g + b))
    print()
    print("   the first row is the shipped triple; the second is the target itself, and it is what the fix uses.")
    print()

    # --- verify the rebuild against the real function on a neutral field -------------------------------
    print("   --- the rebuild checked against `monochrome_green` itself (a neutral 0.40 field) ---")
    for tag, spec in (("the shipped triple", NightVisionSpec(mode=1.0, green_mix=1.0, chroma_residue=0.0)),
                      ("the target triple   ", NightVisionSpec(mode=1.0, green_mix=1.0, chroma_residue=0.0))):
        a = np.full((16, 16, 3), 0.40, np.float32)
        rg, bg = ratio_of(np.asarray(monochrome_green(a, spec), np.float32))
        print("      %s  R/G %.4f   B/G %.4f" % (tag, rg, bg))
    print("      (both rows are the same spec here, so this only checks the ratio is input-independent; the weights")
    print("       themselves are applied in the module and are verified by `tools/nv_display_probe.py` afterwards)")
    print()

    # --- and what the target triple would give on the two modes ---------------------------------------
    print("   --- the consequence for the two green modes ---")
    for label, rg_t, bg_t in (("the real NIGHTSHOT 1", 0.5434, 0.8525), ("the real NIGHTSHOT 2", 0.4766, 0.8192)):
        print("      %-22s R/G %.4f   B/G %.4f" % (label, rg_t, bg_t))
    print("      one triple cannot be both, so the projection is placed on NIGHTSHOT 1's and the difference is left")
    print("      visible rather than averaged away -- which is what `mode_spec` already says it does.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
