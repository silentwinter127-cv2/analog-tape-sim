"""
Find out why `chroma_residue` is inert, by printing the value the term actually receives.

**Measured**: with `cast_variation` fixed, setting `nv_chroma_residue` to 0.35, 0.50, 0.70 and 0.95 gives **bit-identical
output** -- hue spread 0.1891 and saturation 0.1950 in every case.  **So the term does not act on the picture at all**,
which is a mechanism that is declared, reachable and inert -- the fault class this work has met repeatedly.

**And the cause is visible in the source**: `monochrome_green` computes the residue from
`getattr(spec, "chroma_residue", 0.0)`, while the *value* comes from `cfg["nv_chroma_residue"]` through `from_params`
-- and **`mode_spec` rebuilds the spec from the mode's preset on every frame**.  The presets carry their own
`chroma_residue` (`nightshot` 0.35, `super_nightshot` 0.90, `colour_slow_shutter` 1.0), so **a user's value is
overwritten by the preset every frame**, exactly as `accumulate` was.

**That is testable in one line**: print `spec.chroma_residue` as `monochrome_green` receives it, at two different user
settings, and see whether they differ.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from atsim.core.params import merge_values                                    # noqa: E402
from atsim.core.profiles import FACTORY_PRESETS, SCHEMA                       # noqa: E402
from atsim.render.renderer import Renderer                                    # noqa: E402
import atsim.codecs.nightvision as NV                                         # noqa: E402

W, H = 320, 240
_a = np.asarray(Image.open(os.path.join(r"C:\Users\Administrator\.dsh\attachments\v1\objects", "69",
                                        "699c23315df037b01fa831a273364b41f99c28733fb1dc928b22d1e6c281818d")
                             ).convert("RGB"), np.float32) / 255.0
SRC = np.asarray(Image.fromarray((np.clip(_a, 0, 1) * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS),
                 np.float32) / 255.0


class One:
    height, width = H, W
    fps = 25.0
    frame_count = 1

    def frame(self, i):
        return SRC


BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])
OFF = {"halation": 0.0, "ccd_smear": 0.0, "flare_defocus": 0.0, "lens_flare": 0.0,
       "overflow_drain": 0.0, "chroma_aberration": 0.0, "sensor_noise": 0.0, "noise_grain": 0.0,
       "fixed_pattern": 0.0, "spike_spread": 0.0, "motion_blur": 0.0, "stuck_fraction": 0.0,
       "bloom_threshold": 0.0, "overflow_width": 0.0, "sharpness": 1.0}

seen = []
o = NV.monochrome_green


def probe(rgb, spec):
    seen.append((float(getattr(spec, "chroma_residue", -1)), float(getattr(spec, "green_mix", -1)),
                 float(getattr(spec, "cast_variation", -1))))
    return o(rgb, spec)


NV.monochrome_green = probe
import atsim.stages.camera as CAM                                             # noqa: E402
CAM.monochrome_green = probe

print("   === the value `monochrome_green` actually receives ===")
print("   %-18s %-14s %s" % ("configured", "residue seen", "green_mix / cast"))
for cfg_val in (0.0, 0.35, 0.95):
    seen.clear()
    layers = {"camera": dict(OFF, nv_mode=1.0, nv_chroma_residue=cfg_val),
              "format": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0},
              "player": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0},
              "codec": {"dv_enabled": 0.0}, "display": {"enabled": 0.0}}
    rr = Renderer(One(), W, H, 25.0, merge_values(BASE, layers), seed=1)
    rr.render_frame(0)
    if seen:
        r, gm, cv = seen[-1]
        print("   %-18.2f %-14.3f %.2f / %.2f" % (cfg_val, r, gm, cv))
    else:
        print("   %-18.2f %-14s (never called)" % (cfg_val, "-"))

NV.monochrome_green = o
CAM.monochrome_green = o
print()
print("   if the middle column never changes, the preset is overwriting the user's value every frame -- and the")
print("   residue's own presets are 0.35 / 0.90 / 1.0 for the three modes, so the setting is unreachable.")
