"""
Find which player parameter lifts the blue channel, by sweeping them one at a time on a clean input.

The stage-by-stage attribution showed the whole colour drift happens in `player`: on a sodium-lit scene the blue
channel goes **0.0222 -> 0.1199**, a factor of 5.4. The candidates inside the stage are all decoder settings, and
guessing between them has been wrong repeatedly in this work, so each is swept across its own range with everything
else held still.

The subject is a **flat field of the source's own colour**, not the whole picture, because a decoder's behaviour on
a saturated single colour is what the fault is about and a flat field removes the picture's own structure from the
measurement.
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from PIL import Image                                                        # noqa: E402
from atsim.core.params import merge_values                                   # noqa: E402
from atsim.core.profiles import FACTORY_PRESETS, SCHEMA                      # noqa: E402
from atsim.render.renderer import Renderer                                   # noqa: E402

OUT = os.path.join(ROOT, "tools", "ntscrs_ref")
full = np.asarray(Image.open(os.path.join(OUT, "clean_source.jpg")).convert("RGB"),
                  np.float32) / 255.0
H, W = 180, 240
im = Image.fromarray((full * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
PIC = np.asarray(im, np.float32) / 255.0
# a flat field at the source's own mean colour, so the decoder is exercised on the colour and nothing else
FLAT = np.zeros_like(PIC)
FLAT[:] = PIC.reshape(-1, 3).mean(axis=0)
print("   the flat field: %s" % np.round(FLAT[0, 0], 4))


class Still:
    def __init__(self, img):
        self.width, self.height, self.fps, self.frame_count = W, H, 25.0, 3
        self._img = img

    def frame(self, index):
        return self._img


BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])
# a clean path: no tape, no display, both composite ends wide, so the player's own settings are the only variable
CLEAN = {
    "camera": {"exposure": 0.0, "color_cast": 0.0, "halation": 0.0, "ccd_smear": 0.0,
               "chroma_aberration": 0.0, "lens_flare": 0.0, "spike_spread": 0.0,
               "flare_defocus": 0.0, "longitudinal_ca": 0.0, "axis_offset": 0.0},
    "format": {"chroma_bandwidth": 3.0, "luma_bandwidth": 6.0, "chroma_gain": 1.0,
               "color_noise": 0.0, "cross_color": 0.0, "cross_luma": 0.0, "comb_depth": 0.0},
    "codec": {"dv_enabled": 0.0},
    "tape": {},
    "display": {"enabled": 0.0},
}


def run(**player):
    layers = {k: dict(v) for k, v in CLEAN.items()}
    layers["player"] = dict(player)
    p = merge_values(BASE, layers)
    r = Renderer(Still(FLAT), W, H, 25.0, p, seed=5)
    return np.clip(r.render_frame(0), 0, 1)


def means(a):
    return a.reshape(-1, 3).mean(axis=0)


base = run()
print("   with the player at its defaults: %s" % np.round(means(base), 4))
print()
print("   %-30s %-26s %s" % ("player parameter", "channel means", "blue"))
SWEEPS = [
    ("chroma_bandwidth", [0.4, 0.75, 1.5, 3.0]),
    ("chroma_vertical_averaging", [0.0, 0.4, 0.85]),
    ("luma_bandwidth", [1.0, 3.0, 6.0]),
    ("separation_kind", [0.0, 1.0, 2.0, 3.0]),
    ("comb_depth", [0.0, 0.55]),
    ("cross_color", [0.0, 0.45]),
    ("cross_luma", [0.0, 0.3]),
    ("color_noise", [0.0, 0.25]),
    ("output_gain", [0.0, 0.5]),
]
for key, vals in SWEEPS:
    for v in vals:
        img = run(**{key: float(v)})
        m = means(img)
        print("   %-30s %-26s %.4f" % ("%s=%.2f" % (key, v), np.round(m, 4), m[2]))
    print()

print("   the source's own means for comparison: %s" % np.round(FLAT.reshape(-1, 3).mean(axis=0), 4))
