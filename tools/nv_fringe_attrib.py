"""
Separate the model's OWN hue spread from the hue spread of what it was given, then attribute it to a term.

**The contradiction this file exists to settle.**  REVIEW.md section 4.1 concludes, at native resolution, that the
fringe is "EXCESSIVE" -- the case photograph measures 1.212/0.955/1.001/1.228 across the four brightness bands where
the footage measures 0.201/0.409/0.553/1.084.  **But the case photograph is not the footage.**  It is a visible-light
photograph of RGB LEDs whose own hue spread, measured with no chain at all, is 0.63/0.55/0.69/1.16 -- that is, the
input already carries most of the number the conclusion is about, and on the *same* picture the chain's own
contribution is **+0.17 / -0.05 / -0.05 / -0.33**: it *reduces* the spread in three bands of four.  Comparing a
rendered photograph against camcorder footage of a different scene mixes two things, and the conclusion was about the
second while the number was mostly the first.

**The clean instrument is the input's own spread, differenced against the output's, on the same picture at the same
size.**  That removes the scene: whatever is left is the chain's contribution.  It is this project's own rule --
difference against the mechanism switched off, never against the original signal -- applied to a quantity rather
than to a picture.  **And the subtraction is only valid at one size**, because the hue spread is resolution-dependent
(REVIEW.md section 4.1), so the input and the output are always scaled together by `load`.

**Then each term is swept one at a time, with the others left at the schema's default.**  The first version of the
sweep set `chroma_aberration`, `longitudinal_ca` *and* `axis_offset` to zero and then varied one -- which meant the
axis sweep varied a parameter that acts only *through* the radial aberrations while they were all switched off, and
it reported **+0.134 -0.041 -0.169 -0.180 for every value from 0.00 to 2.00**, i.e. five identical rows.  That is
this repository's most expensive failure shape for the fifth recorded time: a rig that switched off the mechanism it
was testing.  `axis_offset` is not inert -- probed at the lens it moves R-B by 0.0034 at 0.6 -- but a sweep cannot
see a parameter whose only effect is through a term it has set to zero.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from atsim.core.params import merge_values                                    # noqa: E402
from atsim.core.profiles import FACTORY_PRESETS, SCHEMA                       # noqa: E402
from atsim.render.renderer import Renderer                                    # noqa: E402
from nv_targets import crop_osd, measure                                      # noqa: E402

OUT = os.path.join(ROOT, "tools", "nightshot")
OBJ = r"C:\Users\Administrator\.dsh\attachments\v1\objects"

#: **The footage itself is the only input whose own fringe is known to be a camcorder's**, so it leads.  The two
#: photographs follow as instruments, and they are labelled for what they are: a photograph of coloured LEDs has its
#: own hue spread before any chain touches it, and the subtraction below is what removes it.
INPUTS = [
    (os.path.join(OUT, "ref_osd_1440.png"), "the camcorder's own frame", 0.20),
    (os.path.join(OUT, "case_new_2048.png"), "the new case photograph", 0.0),
    (os.path.join(OBJ, "e0", "e0e36b261fbbad42777702d3b775fcf30d4f8505fdc104e08b24405e69eda14c"),
     "the earlier case photograph", 0.0),
]

BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])
SPEC = SCHEMA.layer("camera").specs

#: The optics that would blur or tint the picture and hide the effect under test.  **The three radial terms are NOT in
#: this list** -- they are what the sweep varies, and switching them off is the mistake described above.
QUIET = dict(halation=0.0, ccd_smear=0.0, flare_defocus=0.0, lens_flare=0.0, spike_spread=0.0,
             overflow_drain=0.0, source_glow=0.0, shadow_depth=0.0, sensor_noise=0.0, noise_grain=0.0,
             fixed_pattern=0.0, motion_blur=0.0, sharpness=1.0)

#: The longest edge anything is rendered at.  The clip's own frames are 1440 wide and stay there; the photographs are
#: scaled down to it so the running time is bounded.  The subtraction stays valid because `load` scales the *input*
#: and the render happens at that same size.
MAX_EDGE = 1440
BANDS = ("dark", "mid", "bright", "brightest")


class Still:
    fps, frame_count = 25.0, 1

    def __init__(self, a):
        self.a = a
        self.height, self.width = a.shape[:2]

    def frame(self, i):
        return self.a


def load(path: str, frac: float) -> np.ndarray:
    a = np.asarray(Image.open(path).convert("RGB"), np.float32) / 255.0
    a = crop_osd(a, frac) if frac else a
    h, w = a.shape[:2]
    scale = MAX_EDGE / float(max(h, w))
    if scale < 1.0:
        nw, nh = max(2, int(round(w * scale))), max(2, int(round(h * scale)))
        a = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize((nw, nh), Image.LANCZOS),
                       np.float32) / 255.0
    return a


def run(src: np.ndarray, cam: dict) -> np.ndarray:
    layers = {"camera": dict(QUIET, **cam),
              "format": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0},
              "player": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0},
              "codec": {"dv_enabled": 1.0}, "display": {"enabled": 0.0}}
    h, w = src.shape[:2]
    r = Renderer(Still(src), w, h, 25.0, merge_values(BASE, layers), seed=20240101)
    return np.clip(np.asarray(r.render_frame(0), np.float32), 0.0, 1.0)


def bands(a: np.ndarray) -> np.ndarray:
    return np.asarray(measure(a)["spreads"], np.float64)


def main() -> int:
    print("=" * 126)
    print("  THE MODEL'S OWN HUE SPREAD, separated from the input's")
    print("=" * 126)
    print()
    print("   **the input's own spread is printed first on every input**, because a photograph of coloured LEDs carries")
    print("   a fringe before any chain sees it, and a figure that does not subtract it is about the scene, not the model.")
    print()
    print("   %-30s %-42s %s" % ("", "what was measured", "hue spread:  dark     mid      bright   brightest"))
    print("   " + "-" * 124)

    for path, tag, frac in INPUTS:
        src = load(path, frac)
        b_in = bands(src)
        print("   %-30s %-42s %s" % (tag, "the input, no chain (%dx%d)" % (src.shape[1], src.shape[0]),
                                     "  ".join("%.3f" % v for v in b_in)))
        out = run(src, {"nv_mode": 1.0})
        b_out = bands(out)
        print("   %-30s %-42s %s" % ("", "NIGHTSHOT 1, every optical default as shipped",
                                     "  ".join("%.3f" % v for v in b_out)))
        print("   %-30s %-42s %s" % ("", "  the chain's own contribution",
                                     "  ".join("%+.3f" % (o - i) for o, i in zip(b_out, b_in))))
        print()

    print("=" * 126)
    print("  WHICH TERM PUTS IT THERE -- one at a time, the others at the schema's default")
    print("=" * 126)
    print()
    print("   %-24s %-8s %-11s %s"
          % ("", "value", "(default)", "the chain's own spread:  dark     mid      bright   brightest"))
    SWEEPS = [
        ("chroma_aberration", [0.0, 0.05, 0.25, 0.5, 1.0]),
        ("longitudinal_ca", [0.0, 0.25, 0.5, 1.0]),
        ("axis_offset", [0.0, 0.15, 0.3, 0.6]),
    ]
    for path, tag, frac in INPUTS[:2]:
        src = load(path, frac)
        b_in = bands(src)
        print("   --- %s ---" % tag)
        for key, values in SWEEPS:
            for v in values:
                cam = {"nv_mode": 1.0, key: float(v)}
                d = bands(run(src, cam)) - b_in
                print("   %-24s %-8.2f %-11.3f %s"
                      % (key, v, float(SPEC[key].default), "  ".join("%+.3f" % x for x in d)))
            print()
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
