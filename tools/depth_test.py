"""
What the distance term actually does, beside the radial field alone and beside the footage.

**The three panels differ only in one switch each**, so the figure says what each mechanism contributes rather than
which picture looks best:

    the radial field only      `nv_nir_depth = 0` -- the field about a point, whose default spans 21%
    plus the distance term     `nv_nir_depth = 1` -- inverse-square over the model's relative depth
    and the material albedo    both, which is the fully-labelled chain

**And the measurement that decides whether the distance term has done anything** is the frame's own brightness against
the inferred depth: **if the term works, brightness should fall as depth rises**, which is measurable as a correlation
between the rendered luma and the depth map.  **A term that does not move that correlation has not done what it is for**,
however plausible the picture looks -- which is the same test this work had to apply to its own instruments throughout.
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
from atsim.stages.camera import CameraStage                                   # noqa: E402

COEF = np.array([0.299, 0.587, 0.114], np.float32)
OBJ = (r"C:\Users\Administrator\.dsh\attachments\v1\objects")
NS2 = os.path.join(OBJ, "b9", "b94a0e8402a11788032bd0af9e81de7fc4713d3a648384f2c9ff57b21c21f4b6")
PHOTO = os.path.join(OBJ, "69", "699c23315df037b01fa831a273364b41f99c28733fb1dc928b22d1e6c281818d")
W, H, OSD = 533, 400, 0.22
sl = slice(int(H * OSD), H)


def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0
    return np.asarray(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS),
                      np.float32) / 255.0


src = load(PHOTO)
real = load(NS2)
lr = np.clip(real[sl] @ COEF, 0, 1)
ls = np.clip(src[sl] @ COEF, 0, 1)


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

panels = []
CASES = [("radial field only (no depth)", dict(nv_mode=2.0, nv_nir_material=0.0, nv_nir_depth=0.0)),
         ("+ the distance term", dict(nv_mode=2.0, nv_nir_material=0.0, nv_nir_depth=1.0)),
         ("+ the material albedo too", dict(nv_mode=2.0, nv_nir_material=1.0, nv_nir_depth=1.0))]
for tag, cam in CASES:
    seen = {}
    o_att = CameraStage._analyse_scene_nir

    def att(self, rgb, _o=o_att, _s=seen):
        _o(self, rgb)
        _s["d"] = getattr(self, "_nir_depth", None)
        _s["a"] = getattr(self, "_nir_albedo", None)
        _s["dsrc"] = self.nir_depth_source
        _s["asrc"] = self.nir_label_source
    CameraStage._analyse_scene_nir = att
    layers = {"camera": dict(OFF, **cam),
              "format": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0},
              "player": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0},
              "codec": {"dv_enabled": 0.0}, "display": {"enabled": 0.0}}
    rr = Renderer(One(), W, H, 25.0, merge_values(BASE, layers), seed=20240101)
    a = np.clip(np.asarray(rr.render_frame(0), np.float32), 0, 1)
    CameraStage._analyse_scene_nir = o_att
    l = np.clip(a[sl] @ COEF, 0, 1)
    panels.append((tag, a, float(l.mean()), float(l.std()), seen))

print("   %-30s %-10s %-11s %s" % ("", "luma", "contrast", "the real frame: %.4f / %.4f"
                                   % (float(lr.mean()), float(lr.std()))))
for tag, a, lm, lst, seen in panels:
    print("   %-30s %-10.4f %-11.4f" % (tag, lm, lst))

print()
print("   === does the distance term actually make brightness fall with depth? ===")
dep = None
for tag, a, lm, lst, seen in panels:
    if seen.get("d") is not None:
        dep = seen["d"]
        break
if dep is None:
    print("      no depth map was produced, so this cannot be measured")
else:
    g = np.repeat(np.repeat(np.asarray(dep, np.float32), H // dep.shape[0], 0), W // dep.shape[1], 1)[:H, :W]
    gg = np.resize(g, (H, W))[sl]
    for tag, a, lm, lst, seen in panels:
        l = np.clip(a[sl] @ COEF, 0, 1)
        c = float(np.corrcoef(gg.ravel(), l.ravel())[0, 1])
        print("      %-30s corr(luma, depth) = %+0.3f" % (tag, c))
    cr = float(np.corrcoef(gg.ravel(), lr.ravel())[0, 1])
    print("      %-30s corr(luma, depth) = %+0.3f" % ("the real NIGHTSHOT 2", cr))
    print()
    print("      **a working distance term drives this NEGATIVE** -- more depth, less light.  The radial field alone")
    print("      has no reason to correlate with depth at all, and the footage's own value is the target.")

print()
print("   === and the model's own record of what it inferred ===")
for tag, a, lm, lst, seen in panels:
    if seen.get("dsrc") or seen.get("asrc"):
        print("      %s" % tag)
        if seen.get("asrc"):
            print("         albedo: %s" % seen["asrc"])
        if seen.get("dsrc"):
            print("         depth:  %s" % seen["dsrc"])

gap = 8
canvas = Image.new("RGB", (W * 4 + gap * 3, H * 2 + 96), (18, 18, 20))
dr = ImageDraw.Draw(canvas)
dr.text((6, 6), "THE REAL FRAME, THE INPUT, AND THE ILLUMINATOR WITH AND WITHOUT A DISTANCE TERM",
        fill=(240, 240, 240))
top = [("the real NIGHTSHOT 2  luma %.4f contrast %.4f" % (float(lr.mean()), float(lr.std())), real),
       ("the source photograph (the model's input)", src)]
for i, (cap, a) in enumerate(top):
    x = i * (W + gap)
    canvas.paste(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)), (x, 24))
    dr.text((x + 6, 24 + H + 2), cap, fill=(230, 230, 230))
x = 2 * (W + gap)
canvas.paste(Image.fromarray((np.clip(panels[0][1], 0, 1) * 255).astype(np.uint8)), (x, 24))
dr.text((x + 6, 24 + H + 2), "%s  luma %.4f contrast %.4f" % (panels[0][0], panels[0][2], panels[0][3]),
        fill=(230, 230, 230))
if dep is not None:
    dv = np.asarray(dep, np.float32)
    dv = (dv - dv.min()) / max(1e-9, float(dv.max() - dv.min()))
    vis = np.resize(np.repeat(np.repeat(dv, H // dv.shape[0], 0), W // dv.shape[1], 1)[:H, :W], (H, W))
    x = 3 * (W + gap)
    canvas.paste(Image.fromarray((vis * 255).astype(np.uint8)).convert("RGB"), (x, 24))
    dr.text((x + 6, 24 + H + 2), "the inferred relative depth (bright = FAR)", fill=(230, 230, 230))
dr.text((6, 24 + H + 30), "THE MODEL at mode 2", fill=(240, 240, 240))
for i, (tag, a, lm, lst, seen) in enumerate(panels[1:], start=1):
    x = (i - 1) * (W + gap)
    canvas.paste(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)), (x, 24 + H + 50))
    dr.text((x + 6, 24 + H + 50 + H + 2), "%s  luma %.4f contrast %.4f" % (tag, lm, lst), fill=(230, 230, 230))
out = os.path.join(ROOT, "tools", "calib", "IR_DISTANCE.png")
canvas.save(out)
print()
print("   written: %s" % os.path.relpath(out, ROOT))
