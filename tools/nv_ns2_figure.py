"""
NIGHTSHOT 2 against the real frames: the hue, and what was wrong with it.

The user supplied two native 1440x1080 NIGHTSHOT 2 frames and said the render "still differs from the real NIGHTSHOT 2
picture".  It did, and the difference had a single cause that was **not** the projection triple:

    real frame (a)   R/G 0.4113   B/G 0.7898   hue 155
    real frame (b)   R/G 0.3732   B/G 0.7960   hue 153
    the model before R/G ~0.55    B/G ~0.87    hue 161     <- yellow-green
    the model now    R/G 0.3928   B/G 0.8456   hue 157

**`nv_highlight_saturation` was 4.0, which put the saturation gate inside the picture's own range.**  With the triple
re-solved the normalised weights are `[0.4896, 1.2585, 1.0078]`, so green pins at luminance 0.795 and blue at 0.992 --
and since the display drives `lum * w` and clips per channel, the frame was being **dragged toward white across most of
its range**.  A picture pushed toward white reads as a paler, yellower green, which is exactly what "still differs"
described.

Two corrections came out of chasing it, and both are recorded where they belong:

  * **the OSD has to be masked out of any measurement of these frames.**  `STBY`, the battery indicator, the mode glyph
    and the clock are neutral white and saturated green; including them pulled the measured R/G from 0.389 to 0.454 on
    the same two frames, i.e. the ratio was being solved partly against the instrument's own overlay;
  * **the gate and the triple are coupled**, because changing the triple changes the normalised weights and therefore
    where the gate fires.  An attempt to bring blue down on its own moved R/G from 0.4514 to 0.4592 -- the wrong way --
    so the gate is set from the gate sweep and **blue's remaining 0.05 is stated rather than forced.**
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image, ImageDraw

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


class Still:
    fps, frame_count = 25.0, 1

    def __init__(self, a):
        self.a = a
        self.height, self.width = a.shape[:2]

    def frame(self, i):
        return self.a


def load(p, frac, size=(W, H)):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0
    a = a[int(a.shape[0] * frac):]
    return np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize(size, Image.LANCZOS),
                      np.float32) / 255.0


def run(src, gate, mode=2.0):
    lay = {"camera": {"nv_mode": mode, "nv_highlight_saturation": gate},
           "format": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0},
           "player": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0},
           "codec": {"dv_enabled": 1.0}}
    r = Renderer(Still(src), W, H, 25.0, merge_values(BASE, lay), seed=20240101)
    return np.clip(np.asarray(r.render_frame(0), np.float32), 0, 1)


def stat(x):
    a = np.asarray(x, np.float32)
    l = np.clip(a @ LUMA, 0, 1)
    r, g, b = (float(a[..., i].mean()) for i in range(3))
    import colorsys
    hues = []
    for i in range(0, a.shape[0], 6):
        for j in range(0, a.shape[1], 6):
            p = a[i, j]
            if max(p) > 0.08:
                hues.append(colorsys.rgb_to_hsv(float(p[0]), float(p[1]), float(p[2]))[0] * 360)
    return dict(rg=r / max(1e-9, g), bg=b / max(1e-9, g), mean=float(l.mean()),
                hue=float(np.median(hues)) if hues else float("nan"))


def main() -> int:
    a_src = load(os.path.join(OUT, "ref_ns2_a.png"), 0.18)
    b_src = load(os.path.join(OUT, "ref_ns2_b.png"), 0.18)
    old_src = load(os.path.join(OUT, "frames", "nv2_1.png"), 0.20)

    panels = [
        ("THE REAL NIGHTSHOT 2 (a)\nnative 1440x1080, OSD visible", load(os.path.join(OUT, "ref_ns2_a.png"), 0.0,
                                                                        (W, int(W * 1080 / 1440)))),
        ("THE REAL NIGHTSHOT 2 (b)", load(os.path.join(OUT, "ref_ns2_b.png"), 0.0, (W, int(W * 1080 / 1440)))),
        ("the model, gate 4.0\n(what shipped)  hue 161", run(a_src, 4.0)),
        ("the model, ** gate 2.0 **\nhue 157  -  the frames are 153-155", run(a_src, 2.0)),
        ("the model, gate 1.0\nhue 155 but R/G 0.299, too far", run(a_src, 1.0)),
        ("the model on the 10月13 NS2 frame\ngate 2.0", run(old_src, 2.0)),
    ]
    print("=" * 118)
    print("  NIGHTSHOT 2 -- the colour, and the gate that was set inside the picture")
    print("=" * 118)
    print()
    print("   %-46s %-9s %-9s %-9s %-7s" % ("panel", "R/G", "B/G", "mean", "hue"))
    print("   " + "-" * 112)
    for tag, img in panels:
        s = stat(img)
        print("   %-46s %-9.4f %-9.4f %-9.4f %-7.0f"
              % (tag.split(chr(10))[0][:45], s["rg"], s["bg"], s["mean"], s["hue"]))
    print()
    sa, sb = stat(a_src), stat(b_src)
    print("   the two real frames give hue %.0f and %.0f; the model now reads %.0f."
          % (sa["hue"], sb["hue"], stat(run(a_src, 2.0))["hue"]))
    print("   R/G: real %.4f / %.4f, model %.4f   -- on target" % (sa["rg"], sb["rg"], stat(run(a_src, 2.0))["rg"]))
    print("   B/G: real %.4f / %.4f, model %.4f   -- ** still about 0.05 high, stated not hidden **"
          % (sa["bg"], sb["bg"], stat(run(a_src, 2.0))["bg"]))
    print()

    cols, cw, ch, gap, cap = 3, 430, 322, 10, 74
    rows = (len(panels) + cols - 1) // cols
    canvas = Image.new("RGB", ((cw + gap) * cols + gap, (ch + cap) * rows + 40), (14, 14, 16))
    d = ImageDraw.Draw(canvas)
    d.text((10, 6), "NIGHTSHOT 2 -- the saturation gate was set inside the picture's own range, which pulled the frame "
                    "toward white", fill=(240, 240, 242))
    d.text((10, 22), "hue 161 (yellow-green) -> 157, against the real frames' 153-155; R/G 0.55 -> 0.393 against "
                     "0.411 / 0.373", fill=(220, 200, 150))
    for i, (tag, img) in enumerate(panels):
        x = gap + (i % cols) * (cw + gap)
        y = 42 + (i // cols) * (ch + cap)
        canvas.paste(Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).resize((cw, ch), Image.LANCZOS),
                     (x, y))
        s = stat(img)
        for k, line in enumerate(tag.split("\n")):
            d.text((x, y + ch + 3 + 13 * k), line[:76], fill=(245, 245, 248) if k == 0 else (176, 176, 182))
        d.text((x, y + ch + 32), "R/G %.4f   B/G %.4f   mean %.4f" % (s["rg"], s["bg"], s["mean"]),
               fill=(212, 212, 218))
        d.text((x, y + ch + 45), "hue %.0f degrees" % s["hue"], fill=(212, 212, 218))
    path = os.path.join(OUT, "NS2_HUE.png")
    canvas.save(path)
    print("   written: %s" % os.path.relpath(path, ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
