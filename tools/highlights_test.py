"""
Measure the HIGHLIGHTS, which is where the model is reported to depart from the footage.

**Why the measurements so far could not see it.**  Everything taken has been a whole-frame quantity -- mean luma, overall
contrast, hue spread, polarisation.  **A picture can match on all of them and still have its highlights in the wrong
place**, because the top of the range is a small fraction of the pixels and contributes little to an average.  The
footage's own figures show how much room there is: the three NightShot segments read luma 0.2346 / 0.3348 / 0.3405, so the
mode does move a long way up the range.

**The quantities that describe a highlight:**

  * **the upper percentiles** -- p90, p99 and the maximum, which say how bright the brightest parts actually get;
  * **the clipped fraction** -- how much of the frame sits at or near 1.0, because a real night-vision frame blows out
    under the lamp and a model that does not will look flat exactly where the footage is brightest;
  * **the shape of the top decile** -- the mean and spread of the brightest tenth, which separates "a few bright points"
    from "a large bright region".

**And the model has a specific candidate**: `ir_highlights` is the term named for bright surfaces, and `note.txt` says
"高亮区域的饱和" -- highlight saturation -- is one of the mechanisms it asks for.  `_roll_off` is the other, and it was
measured to **raise** contrast.

**So the same figure is taken on the footage, the input, and each mode**, and the departures are named before any fix is
attempted.
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
REF = [("NIGHTSHOT 0", os.path.join(OBJ, "ed",
                                    "ed906503e82e76777332850f677a16db3ea89089fde1c313576edf9fce66f4e0")),
       ("NIGHTSHOT 1", os.path.join(OBJ, "f2",
                                    "f2dd7803fd91d124a7c16be3c6f97df679c8a83d89631a462e9010f13a3e7c5e")),
       ("NIGHTSHOT 2", os.path.join(OBJ, "b9",
                                    "b94a0e8402a11788032bd0af9e81de7fc4713d3a648384f2c9ff57b21c21f4b6"))]
PHOTO = os.path.join(OBJ, "69", "699c23315df037b01fa831a273364b41f99c28733fb1dc928b22d1e6c281818d")
OUT = os.path.join(ROOT, "tools", "calib")
W, H, OSD = 533, 400, 0.22
sl = slice(int(H * OSD), H)


def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0
    return np.asarray(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS),
                      np.float32) / 255.0


def highlights(a):
    """
    The top of the range.  Returned as a tuple so every figure can be printed the same way.

    `clip` is the fraction at or above 0.98, which is where a real frame blows out under the lamp; `top` is the mean of
    the brightest tenth and `topsp` its spread -- a few blown points against a large bright region.
    """
    g = np.clip(np.asarray(a, np.float32)[sl] @ COEF, 0, 1)
    p50, p90, p99 = (float(np.quantile(g, q)) for q in (0.50, 0.90, 0.99))
    top = g >= np.quantile(g, 0.90)
    return dict(mean=float(g.mean()), p50=p50, p90=p90, p99=p99, mx=float(g.max()),
                clip=float((g >= 0.98).mean()), top=float(g[top].mean()), topsp=float(g[top].std()))


COLS = ["mean", "p50", "p90", "p99", "mx", "clip", "top", "topsp"]


def show(tag, a):
    d = highlights(a)
    print("   %-26s %s" % (tag, "  ".join("%s %.4f" % (c, d[c]) for c in COLS)))
    return d


print("   === the footage, the input, and then the model ===")
print("   %-26s %s" % ("", "  ".join("%-9s" % c for c in COLS)))
show("the source photograph", load(PHOTO))
refs = {}
for name, p in REF:
    refs[name] = show(name, load(p))
print()
r0 = refs["NIGHTSHOT 0"]
print("   the footage to match, NIGHTSHOT 0: p90 %.4f  p99 %.4f  max %.4f  clipped %.5f"
      % (r0["p90"], r0["p99"], r0["mx"], r0["clip"]))


class One:
    height, width = H, W
    fps = 25.0
    frame_count = 1

    def frame(self, i):
        return load(PHOTO)


BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])
OFF = {"halation": 0.0, "ccd_smear": 0.0, "flare_defocus": 0.0, "lens_flare": 0.0,
       "overflow_drain": 0.0, "chroma_aberration": 0.0, "sensor_noise": 0.0, "noise_grain": 0.0,
       "fixed_pattern": 0.0, "spike_spread": 0.0, "motion_blur": 0.0, "stuck_fraction": 0.0,
       "bloom_threshold": 0.0, "overflow_width": 0.0, "sharpness": 1.0}


def run(mode=None, **cam):
    #: `nv_mode` travels inside `cam` when the caller passes it that way, so it is only added if absent
    cam = dict(cam)
    if mode is not None:
        cam["nv_mode"] = mode
    layers = {"camera": dict(OFF, **cam),
              "format": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0},
              "player": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0},
              "codec": {"dv_enabled": 0.0}, "display": {"enabled": 0.0}}
    rr = Renderer(One(), W, H, 25.0, merge_values(BASE, layers), seed=20240101)
    return np.clip(np.asarray(rr.render_frame(0), np.float32), 0, 1)


print()
print("   === and the model ===")
print("   %-26s %s" % ("", "  ".join("%-9s" % c for c in COLS)))
panels = []
for tag, kw in (("mode 1 as it ships", dict(nv_mode=1.0)),
                ("mode 2 (Super)", dict(nv_mode=2.0)),
                ("mode 1, ir_response = 0", dict(nv_mode=1.0, nv_ir_response=0.0)),
                ("mode 1, ir_albedo = 0", dict(nv_mode=1.0, nv_ir_albedo=0.0)),
                ("mode 1, illuminator = 0", dict(nv_mode=1.0, nv_illuminator=0.0)),
                ("mode 1, exposure bias = 0", dict(nv_mode=1.0, nv_exposure_steps=0.0)),
                ("mode 0", dict(nv_mode=0.0))):
    a = run(**kw)
    show(tag, a)
    panels.append((tag, a))

print()
d = highlights(panels[0][1])
print("   === where mode 1 departs from the footage ===")
for c in ("p50", "p90", "p99", "mx", "clip", "top"):
    print("      %-6s model %.4f   footage %.4f   %+.4f" % (c, d[c], r0[c], d[c] - r0[c]))

gap = 7
cw, ch = W // 3, (H - int(H * OSD)) // 3
canvas = Image.new("RGB", ((cw + gap) * 3 + gap, (ch + 30) * 2 + 26), (16, 16, 18)
                   if False else ((cw + gap) * 3 + gap, (ch + 30) * 2 + 26), (16, 16, 18))
dr = ImageDraw.Draw(canvas)
dr.text((6, 5), "HIGHLIGHTS -- the footage, the input, and the model, with each night-vision term removed",
        fill=(242, 242, 242))
six = [("the real NIGHTSHOT 0", load(REF[0][1])), ("the source photograph", load(PHOTO))] + panels[:4]
for i, (tag, a) in enumerate(six):
    x = gap + (i % 3) * (cw + gap)
    y = 26 + (i // 3) * (ch + 30)
    canvas.paste(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).resize((cw, ch), Image.LANCZOS), (x, y))
    hh = highlights(a)
    dr.text((x, y + ch + 3), tag, fill=(238, 238, 238))
    dr.text((x, y + ch + 16), "p90 %.3f  p99 %.3f  max %.3f  clip %.4f" % (hh["p90"], hh["p99"], hh["mx"], hh["clip"]),
            fill=(214, 214, 214))
out = os.path.join(OUT, "HIGHLIGHTS.png")
canvas.save(out)
print()
print("   written: %s" % os.path.relpath(out, ROOT))
