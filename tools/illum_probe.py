"""
Probe whether `illuminator` reaches the term, the same way `chroma_residue` was proved unreachable.

**Measured**: `nv_illuminator` set to 0.15, 0.30, 0.50, 0.80, 1.20, 2.00 and 3.00 gives **bit-identical output** --
0.2298 / 0.5394 / 0.8011 / 0.8662 / 0.00000 in every case.  **And the illuminator is the dominant term in a night-vision
picture**, which is what the mode is named for.

**So the value is printed where it is used**, and the two-stage resolution is followed -- `NightVisionSpec.from_params`
sets it from the config, and `mode_spec` rebuilds the spec from the mode's preset on every frame -- because that is the
mechanism that swallowed `chroma_residue`, `accumulate` and `ir_response` before it.

**And the answer is checked against the code as well as the measurement**, since a probe can itself be wrong: which
preset carries `illuminator`, and does `add_illuminator` see the value.
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

print("   === what the presets carry ===")
for k, v in NV.MODE_PRESETS.items():
    print("      %-22s illuminator in the preset: %s" % (k, v.get("illuminator", "(absent)")))
print()
print("   === the value `add_illuminator` receives ===")
seen = []
o = NV.add_illuminator


def probe(rgb, spec):
    seen.append((float(spec.illuminator), float(getattr(spec, "ir_response", -1)),
                 float(getattr(spec, "ir_albedo_mix", -1))))
    return o(rgb, spec)


NV.add_illuminator = probe
import atsim.stages.camera as CAM                                             # noqa: E402
CAM.add_illuminator = probe

print("   %-18s %-18s %s" % ("configured", "illuminator seen", "ir_response / mix"))
for v in (0.0, 0.15, 0.8, 3.0):
    seen.clear()
    layers = {"camera": dict(OFF, nv_mode=1.0, nv_illuminator=v),
              "format": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0},
              "player": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0},
              "codec": {"dv_enabled": 0.0}, "display": {"enabled": 0.0}}
    rr = Renderer(One(), W, H, 25.0, merge_values(BASE, layers), seed=1)
    rr.render_frame(0)
    if seen:
        il, ir, am = seen[-1]
        print("   %-18.2f %-18.3f %.2f / %.2f" % (v, il, ir, am))
    else:
        print("   %-18.2f %-18s (never called)" % (v, "-"))

NV.add_illuminator = o
CAM.add_illuminator = o
print()
print("   and the schema's own default for nv_illuminator:")
print("      %s" % BASE["camera"].get("nv_illuminator"))
print("   the mode presets do NOT carry an illuminator except for colour_slow_shutter (0.0), so the user's value")
print("   should survive `mode_spec` -- which means the cause is elsewhere and the probe above names it.")
