"""
Confirm that `ir_highlights` is the 3x, with the mix at the schema's default rather than at the zero I tested with.

**The earlier isolation test gated the term off by accident.**  It built its spec with `ir_albedo_mix = 0.0`, and
`ir_highlights` returns the picture unchanged when `ir_albedo_mix * ir_response < 1e-4` -- so the test measured a
disabled mechanism and reported "not the excess".  **The schema's default for `nv_ir_albedo` is 0.70**, and the
attribution table's "the mode on, nothing else" row used the schema's defaults, which is where the **0.3019** came from.

**And the mechanism is visible in one line**:

    ir = np.power(lum, 0.55)
    return rgb + ir[..., None] * m

**`lum^0.55 >= lum` for every value in [0, 1]** -- 0.2^0.55 = 0.41, a factor of 2.05 -- so the term **adds light, and
adds most where the picture is darkest**.  At the schema's `m = 0.70` that is a large brightening of exactly the kind
the attribution attributes to it.

**So the test is the whole scene branch at the schema's defaults, and then with only `ir_albedo_mix` zeroed**, which
isolates the term without touching anything else.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from atsim.codecs import nightvision as nv                                    # noqa: E402
from atsim.core.params import merge_values                                    # noqa: E402
from atsim.core.profiles import FACTORY_PRESETS, SCHEMA                       # noqa: E402
from atsim.render.renderer import Renderer                                    # noqa: E402

COEF = np.array([0.299, 0.587, 0.114], np.float32)
OBJ = (r"C:\Users\Administrator\.dsh\attachments\v1\objects")
PHOTO = os.path.join(OBJ, "69", "699c23315df037b01fa831a273364b41f99c28733fb1dc928b22d1e6c281818d")
W, H = 533, 400
OSD = 0.22
src = np.asarray(Image.open(PHOTO).convert("RGB"), np.float32) / 255.0
src = np.asarray(Image.fromarray((np.clip(src, 0, 1) * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS),
                 np.float32) / 255.0
L0 = float((src[int(H * OSD):] @ COEF).mean())

BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])
print("   the schema's own defaults for the IR terms:")
for k in ("nv_ir_response", "nv_ir_albedo", "nv_illuminator", "nv_exposure_steps", "nv_green_mix"):
    print("      %-20s %s" % (k, BASE["camera"].get(k)))
print("   and the product that gates the term: ir_response * ir_albedo = %.4f"
      % (float(BASE["camera"]["nv_ir_response"]) * float(BASE["camera"]["nv_ir_albedo"])))
print()
print("   the source's luma: %.4f     the real NIGHTSHOT 0: 0.2349" % L0)


class One:
    height, width = H, W
    fps = 25.0
    frame_count = 1

    def frame(self, i):
        return src


OFFDEF = {"halation": 0.0, "ccd_smear": 0.0, "flare_defocus": 0.0, "lens_flare": 0.0,
          "overflow_drain": 0.0, "chroma_aberration": 0.0}
DISPLAY = {"enabled": 1.0, "gamma": 1.0, "scanlines": 0.0, "mask_depth": 0.0, "saturation": 0.62}


def luma_of(**cam):
    layers = {"camera": dict(OFFDEF, **cam),
              "format": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0},
              "player": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0},
              "codec": {"dv_enabled": 1.0}, "tape": {"head_switching": 0.0},
              "display": dict(DISPLAY)}
    p = merge_values(BASE, layers)
    rr = Renderer(One(), W, H, 25.0, p, seed=20240101)
    a = np.clip(np.asarray(rr.render_frame(0), np.float32), 0, 1)
    return float((a[int(H * OSD):] @ COEF).mean())


print()
print("   === the decisive test, changing ONLY the IR term ===")
print("   %-50s %-10s %-9s %s" % ("setting", "luma", "factor", "verdict"))
rows = [
    ("the mode on, schema defaults", dict(nv_mode=1.0)),
    ("...with nv_ir_albedo = 0 (the IR term off)", dict(nv_mode=1.0, nv_ir_albedo=0.0)),
    ("...with nv_ir_response = 0 (the filter in)", dict(nv_mode=1.0, nv_ir_response=0.0)),
]
for name, cam in rows:
    lm = luma_of(**cam)
    print("   %-50s %-10.4f %-9.4f %s" % (name, lm, lm / max(1e-9, L0),
                                          "the IR term IS the excess" if lm > L0 * 1.6 else "not the excess"))

print()
print("   === and the term's own arithmetic, on the picture's real values ===")
lum = np.clip(src @ COEF, 0, 1)
for q in (1, 25, 50, 75, 99):
    v = float(np.percentile(lum, q))
    print("      a pixel at the %2dth percentile: lum %.4f -> lum^0.55 %.4f   (factor %.3f)"
          % (q, v, v ** 0.55, (v ** 0.55) / max(1e-9, v)))
print()
print("   `lum^0.55 >= lum` everywhere in [0,1] and most strongly in the shadows, so the term brightens darkest")
print("   regions the most.  At the schema's mix of 0.70 it adds 0.70 * lum^0.55 to every pixel.")
