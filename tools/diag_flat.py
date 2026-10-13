"""
Diagnose the inversion: the field has become almost uniform, so the illuminator is a flat lift that flattens contrast.

**What the render shows and the numbers did not catch.**  The model's output is pale and inverted-looking -- bright where
the real frame is dark -- while its **mean luma matches the footage (0.2371 against 0.2346)**.  A correct *level* with a
reversed *appearance* means the fault is in the **distribution**, and the field's own range says why:

    the field's min, mean and max are all within a few percent of each other

**so `illuminator * field` is a nearly uniform addition to every pixel.**  A uniform lift raises the shadows and
compresses everything toward the middle, which is exactly what "pale and flat" looks like -- and against a real frame
whose subject is a **dark case in front of a lit wall**, a lift destroys the contrast that carries the image.

**And the cause is the shape function itself**: `L^2 / (L^2 + r^2)` with `L = 1/beam_angle` spans only about
**0.88 to 1.07** across a frame whose `r` runs to roughly 1, because `L` is large next to `r`.  **So the "beam" never
varies much within the picture**, whatever `beam_angle` is set to -- the parameter changes `L`, but every value of `L`
that the clamp allows is still large compared with the frame's own radius.

**The quantity that decides this is contrast**, not level: the standard deviation of the luminance, against the real
frame's own.  It is measured here for the source, the model and the footage, so the fault is stated as a number.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from atsim.codecs import nightvision as nv                                    # noqa: E402
from atsim.core.params import merge_values                                    # noqa: E402
from atsim.core.profiles import FACTORY_PRESETS, SCHEMA                       # noqa: E402
from atsim.render.renderer import Renderer                                    # noqa: E402

OUT = os.path.join(ROOT, "tools", "calib")
COEF = np.array([0.299, 0.587, 0.114], np.float32)
OBJ = (r"C:\Users\Administrator\.dsh\attachments\v1\objects")
W, H, OSD = 533, 400, 0.22
sl = slice(int(H * OSD), H)


def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0
    return np.asarray(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS),
                      np.float32) / 255.0


REAL = [("NIGHTSHOT 0", os.path.join(OBJ, "ed",
                                     "ed906503e82e76777332850f677a16db3ea89089fde1c313576edf9fce66f4e0")),
        ("NIGHTSHOT 1", os.path.join(OBJ, "f2",
                                     "f2dd7803fd91d124a7c16be3c6f97df679c8a83d89631a462e9010f13a3e7c5e")),
        ("NIGHTSHOT 2", os.path.join(OBJ, "b9",
                                     "b94a0e8402a11788032bd0af9e81de7fc4713d3a648384f2c9ff57b21c21f4b6"))]
src = load(os.path.join(OBJ, "69", "699c23315df037b01fa831a273364b41f99c28733fb1dc928b22d1e6c281818d"))

print("   === the field's own range, which is the fault ===")
for ang in (0.25, 0.55, 1.0, 2.0):
    f = nv.illuminator_field(H, W, nv.NightVisionSpec(beam_x=0.5, beam_y=0.5, beam_angle=ang))
    print("      beam_angle %.2f -> min %.5f  mean %.5f  max %.5f   spread/mean %.4f"
          % (ang, float(f.min()), float(f.mean()), float(f.max()),
             float(f.max() - f.min()) / max(1e-9, float(f.mean()))))
print("   a beam that varies by a few percent across the picture is not a beam; it is a flat lift.")

print()
print("   === contrast, which is what a flat lift destroys ===")
print("   %-26s %-10s %-12s %s" % ("", "luma", "contrast (std)", "vs the real frame"))


class One:
    height, width = H, W
    fps = 25.0
    frame_count = 1

    def frame(self, i):
        return src


BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])


def render(ill):
    layers = {"camera": {"nv_mode": 1.0, "nv_green_mix": 1.0, "nv_illuminator": ill},
              "format": {"chroma_bandwidth": 3.0, "luma_bandwidth": 6.0},
              "player": {"chroma_bandwidth": 3.0, "luma_bandwidth": 6.0}}
    rr = Renderer(One(), W, H, 25.0, merge_values(BASE, layers), seed=1)
    return np.clip(np.asarray(rr.render_frame(0), np.float32), 0, 1)


real0 = load(REAL[0][1])
r_std = float(np.clip(real0[sl] @ COEF, 0, 1).std())
for tag, a in (("the source photograph", src), ("the real NIGHTSHOT 0", real0)):
    lum = np.clip(a[sl] @ COEF, 0, 1)
    print("   %-26s %-10.4f %-12.4f %.3f" % (tag, float(lum.mean()), float(lum.std()), float(lum.std()) / r_std))
for ill in (0.0, 0.15, 0.30):
    a = render(ill)
    lum = np.clip(a[sl] @ COEF, 0, 1)
    print("   %-26s %-10.4f %-12.4f %.3f"
          % ("the model, lamp %.2f" % ill, float(lum.mean()), float(lum.std()), float(lum.std()) / r_std))

gap = 10
canvas = Image.new("RGB", (W * 3 + gap * 2, H * 2 + 96), (18, 18, 20))
d = ImageDraw.Draw(canvas)
d.text((6, 6), "THE REAL FRAMES", fill=(240, 240, 240))
for i, (n, p) in enumerate(REAL):
    a = load(p)
    canvas.paste(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)), (i * (W + gap), 24))
    lum = np.clip(a[sl] @ COEF, 0, 1)
    d.text((i * (W + gap) + 6, 24 + H + 2), "%s  luma %.4f  contrast %.4f" % (n, float(lum.mean()), float(lum.std())),
           fill=(230, 230, 230))
d.text((6, 24 + H + 26), "THE MODEL (the field is flat, so this is a level lift and not a beam)", fill=(240, 240, 240))
for i, ill in enumerate((0.0, 0.15, 0.30)):
    a = render(ill)
    canvas.paste(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)), (i * (W + gap), 24 + H + 44))
    lum = np.clip(a[sl] @ COEF, 0, 1)
    d.text((i * (W + gap) + 6, 24 + H + 44 + H + 2),
           "lamp %.2f  luma %.4f  contrast %.4f" % (ill, float(lum.mean()), float(lum.std())),
           fill=(230, 230, 230))
p = os.path.join(OUT, "DIAG.png")
canvas.save(p)
print()
print("   written: %s" % os.path.relpath(p, ROOT))
