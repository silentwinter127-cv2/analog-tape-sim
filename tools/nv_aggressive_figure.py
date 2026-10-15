"""
The aggressive strategy: brightness from the curve instead of from the exposure.

The previous round raised the texture from 55% to 69% by moving `nv_exposure_trim` from -0.55 to -0.25, and the
measurement that came out of it was that **`_expose` costs 39% of the picture's texture** -- because a global
multiplication takes the local contrast down with the level.  This round takes that seriously:

    brightness is set by the CURVE (dsp_contrast), not by the exposure (nv_exposure_trim)

which is what the two controls are for.  Exposure decides where the mid-tone sits; a tone curve decides how the ends
are distributed.  Using exposure for the dark end means paying for it in texture everywhere.

    [DV frame]                         texture  %    p1      p5      p50     >.90
    the frame itself                   0.03947  100  0.0353  0.0565  0.1833   6.71
    before this round (trim -.25 .40)  0.02728   69  0.0327  0.0402  0.1909  15.28
    trim -0.10, contrast 0.70          0.02993   76  0.0231  0.0295  0.1822  18.48
    + nv_ir_albedo 0.35                0.03159   80  0.0250  0.0315  0.1727  18.73

    [photograph]                       0.01086   27 -> 0.02315  57

**p50 lands on the target -- 0.1822 against 0.1833 -- with the texture at 80%**, and the photograph more than doubles.
The cost is stated rather than hidden: **18.7% of the frame is above 0.90 where the footage has 6.71%**, because the
curve pushes the bright end up as well as the dark end down.

`nv_ir_albedo` moves from 0.70 to 0.35: the IR term had 70% of the picture's appearance, and the measurement is that
lowering it leaves more of the visible-light structure intact while the reflection ratio still drives the shadows.

**And `nv_ir_focus_shift` 0.5 -> 0.2 changed nothing at all** (0.02940 either way), which is recorded as the next thing
to look at rather than a tuning result.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from atsim.core.params import merge_values                                    # noqa: E402
from atsim.core.profiles import FACTORY_PRESETS, SCHEMA                       # noqa: E402
from atsim.render.renderer import Renderer                                    # noqa: E402

OUT = os.path.join(ROOT, "tools", "nightshot")
W, H = 720, 540
BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])
LUMA = np.array([0.299, 0.587, 0.114], np.float32)
BEFORE = {"nv_exposure_trim": -0.25, "dsp_contrast": 0.40, "nv_ir_albedo": 0.70}


class Still:
    fps, frame_count = 25.0, 1

    def __init__(self, a):
        self.a = a
        self.height, self.width = a.shape[:2]

    def frame(self, i):
        return self.a


def load(p, frac=0.0):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0
    if frac:
        a = a[int(a.shape[0] * frac):]
    return np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS),
                      np.float32) / 255.0


photo = load(os.path.join(OUT, "new_photo.jpg"))
dv = load(os.path.join(OUT, "sc1515", "d_nv3.png"), 0.20)


def run(src, cam=None):
    lay = {"camera": dict(nv_mode=1.0, **(cam or {})),
           "format": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0},
           "player": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0},
           "codec": {"dv_enabled": 1.0}}
    r = Renderer(Still(src), W, H, 25.0, merge_values(BASE, lay), seed=20240101)
    return np.clip(np.asarray(r.render_frame(0), np.float32), 0, 1)


def texture(x):
    l = np.clip(np.asarray(x, np.float32) @ LUMA, 0, 1)
    img = Image.fromarray((l * 255).astype(np.uint8))
    mean = np.asarray(img.filter(ImageFilter.GaussianBlur(2.0)), np.float32) / 255.0
    sq = np.asarray(Image.fromarray(((l * l) * 255).clip(0, 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(2.0)), np.float32) / 255.0
    return float(np.sqrt(np.maximum(0.0, sq - mean * mean)).mean())


def stat(x):
    l = np.clip(np.asarray(x, np.float32) @ LUMA, 0, 1)
    return dict(t=texture(x), p1=float(np.quantile(l, 0.01)), p5=float(np.quantile(l, 0.05)),
                p50=float(np.quantile(l, 0.50)), p99=float(np.quantile(l, 0.99)),
                std=float(l.std()), b90=100.0 * float((l > 0.90).mean()))


def main() -> int:
    nv = run(dv)
    panels = [
        ("THE DV FRAME\nthe target: texture 0.03947  p50 0.1833", dv, None),
        ("the model BEFORE this round\ntrim -0.25 / contrast 0.40", run(dv, BEFORE), run(dv, BEFORE)),
        ("the model NOW\n** trim -0.10 / contrast 0.70 / ir 0.35 **", nv, nv),
        ("the model now, sharpness 1.0\n(what the check forbids as a default)", run(dv, {"sharpness": 1.0}), None),
        ("THE PHOTOGRAPH\nthe visible-light input", photo, None),
        ("the model on it BEFORE\n(27% of the input's texture)", run(photo, BEFORE), run(photo, BEFORE)),
        ("the model on it NOW\n(57%)", run(photo), run(photo)),
        ("the model on it now\ncontrast 0.50 (a gentler curve)", run(photo, {"dsp_contrast": 0.50}), None),
    ]
    tp, td = texture(photo), texture(dv)
    print("=" * 124)
    print("  BRIGHTNESS FROM THE CURVE, NOT FROM THE EXPOSURE -- texture is the local standard deviation")
    print("=" * 124)
    print()
    print("   %-52s %-9s %-6s %-8s %-8s %-8s %-8s %-7s"
          % ("panel", "texture", "%", "p1", "p5", "p50", "std", ">.90"))
    print("   " + "-" * 120)
    for tag, img, _ in panels:
        s = stat(img)
        base = td if img is not photo and "PHOTOGRAPH" not in tag.upper() else tp
        print("   %-52s %-9.5f %-6.0f %-8.4f %-8.4f %-8.4f %-8.4f %-7.2f"
              % (tag.split(chr(10))[0][:51], s["t"], 100 * s["t"] / base, s["p1"], s["p5"], s["p50"], s["std"],
                 s["b90"]))
    print()
    s0, s1 = stat(run(dv, BEFORE)), stat(nv)
    print("   texture   %.5f -> %.5f  (%.0f%% -> %.0f%% of the frame's)"
          % (s0["t"], s1["t"], 100 * s0["t"] / td, 100 * s1["t"] / td))
    print("   p50       %.4f -> %.4f   against the frame's %.4f" % (s0["p50"], s1["p50"], stat(dv)["p50"]))
    print("   and the cost: >0.90 goes %.2f%% -> %.2f%% against the frame's %.2f%%"
          % (s0["b90"], s1["b90"], stat(dv)["b90"]))
    print()

    cols, cw, ch, gap, cap = 4, 330, 248, 8, 70
    rows = (len(panels) + cols - 1) // cols
    canvas = Image.new("RGB", ((cw + gap) * cols + gap, (ch + cap) * rows + 40), (14, 14, 16))
    d = ImageDraw.Draw(canvas)
    d.text((10, 6), "BRIGHTNESS FROM THE CURVE, NOT FROM THE EXPOSURE -- texture 55%% -> 80%% of the DV frame's, and "
                    "p50 on target", fill=(240, 240, 242))
    d.text((10, 22), "the cost is stated: 18.7%% of the frame above 0.90 where the footage has 6.71%%",
           fill=(220, 200, 150))
    for i, (tag, img, _) in enumerate(panels):
        x = gap + (i % cols) * (cw + gap)
        y = 42 + (i // cols) * (ch + cap)
        canvas.paste(Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).resize((cw, ch), Image.LANCZOS),
                     (x, y))
        s = stat(img)
        for k, line in enumerate(tag.split("\n")):
            d.text((x, y + ch + 3 + 13 * k), line[:64], fill=(245, 245, 248) if k == 0 else (176, 176, 182))
        d.text((x, y + ch + 32), "texture %.5f   p1 %.4f  p50 %.4f  p99 %.4f" % (s["t"], s["p1"], s["p50"], s["p99"]),
               fill=(212, 212, 218))
        d.text((x, y + ch + 45), "std %.4f   >0.90 %.2f%%" % (s["std"], s["b90"]), fill=(212, 212, 218))
    path = os.path.join(OUT, "AGGRESSIVE.png")
    canvas.save(path)
    print("   written: %s" % os.path.relpath(path, ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
