"""
The picture at the size the comparisons use, side by side, because four statistics in a row have failed on this.

**What has been established and what has not.**

  * **the user's report is that the render is striped and short of detail against the footage**, and the crop written out
    three rounds ago **shows exactly that**: the real frame carries texture and edges, the render is filled with coarse
    bands;
  * **four statistics failed to confirm it**, and each was wrong for its own reason:
    the row-mean spread is *lower* for a low-passed banded picture than for ordinary content (0.240 against the
    footage's 0.431); the one-pixel high-pass says the render carries **more** detail than the footage (0.085 against
    0.079); the peak-to-floor spectral ratio picked the lowest bin in both cases and so measured the frame's vertical
    gradient; and the earlier ablations moved nothing because they were applied to parameters that were not in the path.
  * **so the measurements cannot be used to choose between explanations, and the picture is the evidence.**

**This writes the render at native size beside the footage at the same size**, with no crop and no enlargement, so the
overall impression can be compared rather than a detail level -- **which is what the user has been describing all along
and what a scalar has repeatedly contradicted.**
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from atsim.core.params import merge_values                                    # noqa: E402
from atsim.core.profiles import FACTORY_PRESETS, SCHEMA                       # noqa: E402
from atsim.render.renderer import Renderer                                    # noqa: E402

COEF = np.array([0.299, 0.587, 0.114], np.float32)
OBJ = (r"C:\Users\Administrator\.dsh\attachments\v1\objects")
NS0 = os.path.join(OBJ, "ed", "ed906503e82e76777332850f677a16db3ea89089fde1c313576edf9fce66f4e0")
NS1 = os.path.join(OBJ, "f2", "f2dd7803fd91d124a7c16be3c6f97df679c8a83d89631a462e9010f13a3e7c5e")
NS2 = os.path.join(OBJ, "b9", "b94a0e8402a11788032bd0af9e81de7fc4713d3a648384f2c9ff57b21c21f4b6")
PHOTO = os.path.join(OBJ, "69", "699c23315df037b01fa831a273364b41f99c28733fb1dc928b22d1e6c281818d")
OUT = os.path.join(ROOT, "tools", "calib")
W, H = 533, 400


def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0
    return np.asarray(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS),
                      np.float32) / 255.0


src = load(PHOTO)


class One:
    height, width = H, W
    fps = 25.0
    frame_count = 1

    def frame(self, i):
        return src


BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])
OFF = {"halation": 0.0, "ccd_smear": 0.0, "flare_defocus": 0.0, "lens_flare": 0.0,
       "overflow_drain": 0.0, "chroma_aberration": 0.0, "sensor_noise": 0.0, "noise_grain": 0.0,
       "fixed_pattern": 0.0, "spike_spread": 0.0, "motion_blur": 0.0, "stuck_fraction": 0.0,
       "bloom_threshold": 0.0, "overflow_width": 0.0, "sharpness": 1.0}


def run(mode, **cam):
    layers = {"camera": dict(OFF, nv_mode=mode, **cam),
              "format": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0},
              "player": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0},
              "codec": {"dv_enabled": 0.0}, "display": {"enabled": 0.0}}
    rr = Renderer(One(), W, H, 25.0, merge_values(BASE, layers), seed=20240101)
    return np.clip(np.asarray(rr.render_frame(0), np.float32), 0, 1)


panels = [("the source photograph", src), ("NIGHTSHOT 0 (mode off)", run(0.0)),
          ("NIGHTSHOT 1", run(1.0)), ("NIGHTSHOT 2 (Super)", run(2.0))]
gap = 8
canvas = Image.new("RGB", (W * 4 + gap * 3, H * 2 + 108), (16, 16, 18))
d = ImageDraw.Draw(canvas)
d.text((6, 5), "THE REAL FRAMES (the target)", fill=(242, 242, 242))
for i, (name, p) in enumerate((("NIGHTSHOT 0", NS0), ("NIGHTSHOT 1", NS1), ("NIGHTSHOT 2", NS2))):
    a = load(p)
    x = i * (W + gap)
    canvas.paste(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)), (x, 22))
    d.text((x + 6, 22 + H + 3), name, fill=(232, 232, 232))
d.text((6, 22 + H + 30), "THE MODEL at the schema's own settings -- no crop, no enlargement, so the overall "
                         "impression is what is compared", fill=(242, 242, 242))
for i, (name, a) in enumerate(panels):
    x = i * (W + gap)
    canvas.paste(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)), (x, 22 + H + 48))
    d.text((x + 6, 22 + H + 48 + H + 3), name, fill=(232, 232, 232))
out = os.path.join(OUT, "SIDE_BY_SIDE.png")
canvas.save(out)
print("   written: %s" % os.path.relpath(out, ROOT))
print()
print("   the four statistics that failed on this question, recorded so they are not reused:")
print("      the row-mean spread          blind: lower for banded-but-low-passed than for content")
print("      the one-pixel high-pass      says the render has MORE detail (0.085 against 0.079)")
print("      the peak-to-floor ratio      picks bin 1 in both, so it measures the vertical gradient")
print("      parameter ablations          moved nothing, because the parameters were not in the path")
