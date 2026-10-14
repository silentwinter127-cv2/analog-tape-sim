"""
The final test figure: the new photograph, the settings that reach the footage's highlights, and the footage.

**What is established.**  The highlight mechanism now works and can be asked to blow out:

    the footage                       mean 0.2346   p99 0.9511   max 1.0000   clipped 0.00657
    the default (sat 0.0, gain 2.0)   mean 0.2543   p99 0.7963   max 0.9104   clipped **0.00000**
    sat 0.8 + exposure -0.15          mean 0.2573   p99 0.9202   max 0.9973   clipped **0.00226**
    sat 1.0 + exposure -0.30          mean 0.2373   p99 0.8705   max 0.9920   clipped **0.00064**

**so the engine can produce a blown highlight -- 0.00064 to 0.00226 of the frame against the footage's 0.00657 -- where
before it produced none at all.**  And the setting that keeps the mean nearest the footage's (sat 1.0 with exposure -0.30)
is not the default, **because making it the default broke four checks**: the exposure loop's own behaviour, the scanline
structure costing no light, and the overflow drain's dark ring.  **Those four are properties verified earlier in this work,
and trading them for a highlight is what the 163 checks exist to prevent** -- so the mechanism is shipped reachable and
the calibration is recorded rather than imposed.

**This figure shows both**, so the difference is visible rather than asserted, and every panel carries its own figures.
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
NEW = os.path.join(OBJ, "e0", "e0e36b261fbbad42777702d3b775fcf30d4f8505fdc104e08b24405e69eda14c")
REF = [("NIGHTSHOT 0", os.path.join(OBJ, "ed",
                                    "ed906503e82e76777332850f677a16db3ea89089fde1c313576edf9fce66f4e0")),
       ("NIGHTSHOT 2", os.path.join(OBJ, "b9",
                                    "b94a0e8402a11788032bd0af9e81de7fc4713d3a648384f2c9ff57b21c21f4b6"))]
OUT = os.path.join(ROOT, "tools", "calib")
W, H, OSD = 533, 400, 0.22
sl = slice(int(H * OSD), H)


def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0
    return np.asarray(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS),
                      np.float32) / 255.0


def fig(a):
    g = np.clip(np.asarray(a, np.float32)[sl] @ COEF, 0, 1)
    return dict(mean=float(g.mean()), p90=float(np.quantile(g, 0.90)), p99=float(np.quantile(g, 0.99)),
                mx=float(g.max()), clip=float((g >= 0.98).mean()))


NEW_IMG = load(NEW)


class One:
    height, width = H, W
    fps = 25.0
    frame_count = 1

    def frame(self, i):
        return NEW_IMG


BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])
OFF = {"halation": 0.0, "ccd_smear": 0.0, "flare_defocus": 0.0, "lens_flare": 0.0,
       "overflow_drain": 0.0, "chroma_aberration": 0.0, "sensor_noise": 0.0, "noise_grain": 0.0,
       "fixed_pattern": 0.0, "spike_spread": 0.0, "motion_blur": 0.0, "stuck_fraction": 0.0,
       "bloom_threshold": 0.0, "overflow_width": 0.0, "sharpness": 1.0}


def run(**cam):
    cam = dict(cam)
    cam.setdefault("nv_mode", 1.0)
    layers = {"camera": dict(OFF, **cam),
              "format": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0},
              "player": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0},
              "codec": {"dv_enabled": 0.0}, "display": {"enabled": 0.0}}
    rr = Renderer(One(), W, H, 25.0, merge_values(BASE, layers), seed=20240101)
    return np.clip(np.asarray(rr.render_frame(0), np.float32), 0, 1)


panels = [("the real NIGHTSHOT 0", load(REF[0][1])), ("the real NIGHTSHOT 2", load(REF[1][1])),
          ("the new photograph (the input)", NEW_IMG),
          ("the model, as it ships", run()),
          ("the model, sat 0.8 / exposure -0.15", run(saturation_stops=0.8, exposure=-0.15,
                                                     nv_highlight_saturation=4.0)),
          ("the model, sat 1.0 / exposure -0.30", run(saturation_stops=1.0, exposure=-0.30,
                                                     nv_highlight_saturation=4.0))]
print("   %-40s %-8s %-8s %-8s %-8s %s" % ("", "mean", "p90", "p99", "max", "clipped"))
for tag, arr in panels:
    f = fig(arr)
    print("   %-40s %-8.4f %-8.4f %-8.4f %-8.4f %.5f"
          % (tag, f["mean"], f["p90"], f["p99"], f["mx"], f["clip"]))

gap = 7
cw, ch = W // 3, (H - int(H * OSD)) // 3
canvas = Image.new("RGB", ((cw + gap) * 3 + gap, (ch + 46) * 2 + 28), (16, 16, 18))
d = ImageDraw.Draw(canvas)
d.text((6, 5), "HIGHLIGHTS -- the new input has an emitter in frame, and the engine can now blow it out",
       fill=(242, 242, 242))
for i, (tag, arr) in enumerate(panels):
    x = gap + (i % 3) * (cw + gap)
    y = 26 + (i // 3) * (ch + 46)
    canvas.paste(Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).resize((cw, ch), Image.LANCZOS), (x, y))
    f = fig(arr)
    d.text((x, y + ch + 3), tag, fill=(238, 238, 238))
    d.text((x, y + ch + 16), "mean %.3f  p90 %.3f  p99 %.3f" % (f["mean"], f["p90"], f["p99"]), fill=(214, 214, 214))
    d.text((x, y + ch + 29), "max %.3f  clipped %.4f" % (f["mx"], f["clip"]), fill=(214, 214, 214))
out = os.path.join(OUT, "HIGHLIGHT_FINAL.png")
canvas.save(out)
print()
print("   written: %s" % os.path.relpath(out, ROOT))
