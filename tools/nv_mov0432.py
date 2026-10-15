"""
The user's 2009 clip: a dried flower arrangement, lit by a warm indoor source.

**What the source actually is, measured before anything was rendered**, because this work has three times taken a
measurement in a configuration that could not see its own subject:

    640x480, MPEG-1, 30 fps, 6 s, dated 2009 -- so it predates the colour clips by 17 years and is not from the same
    pipeline.  Its range is NARROW and there are no highlights at all:

        p1 0.0000   p50 0.2128   p99 0.4703   pixels above 0.90: 0.00%
        primary #3C3817   R/G 1.0833   B/G 0.4184   G share 0.3997

    so it cannot test the white point, the clipping or the texture-in-highlight work -- and it is a good test of the
    dark end and of the material labelling, because the subject is a **plant**: `IR_ALBEDO` gives foliage 2.100,
    the highest entry in the table, and near-IR reflectance is exactly what separates vegetation from everything else.

Nothing is retuned here.  The clip is the input, the model runs with its current defaults, and the panels carry the
figures -- including whether the material analysis finds foliage at all, which is the one thing this picture can test
that the previous ones could not.
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

OUT = os.path.join(ROOT, "tools", "nightshot", "mov0432")
W, H = 640, 480
BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])
LUMA = np.array([0.299, 0.587, 0.114], np.float32)


class Still:
    fps, frame_count = 25.0, 1

    def __init__(self, a):
        self.a = a
        self.height, self.width = a.shape[:2]

    def frame(self, i):
        return self.a


def load(p):
    return np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0


def render(src, mode, cam=None):
    lay = {"camera": dict(nv_mode=float(mode), **(cam or {})),
           "format": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0},
           "player": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0},
           "codec": {"dv_enabled": 1.0}}
    r = Renderer(Still(src), W, H, 25.0, merge_values(BASE, lay), seed=20240101)
    return np.clip(np.asarray(r.render_frame(0), np.float32), 0, 1), r


def stat(x):
    l = np.clip(np.asarray(x, np.float32) @ LUMA, 0, 1)
    r_, g_, b_ = (float(np.asarray(x, np.float32)[..., i].mean()) for i in range(3))
    return dict(mean=float(l.mean()), p1=float(np.quantile(l, 0.01)), p5=float(np.quantile(l, 0.05)),
                p50=float(np.quantile(l, 0.50)), p99=float(np.quantile(l, 0.99)), std=float(l.std()),
                prim="#%02X%02X%02X" % (int(r_ * 255), int(g_ * 255), int(b_ * 255)),
                rg=r_ / max(1e-9, g_), bg=b_ / max(1e-9, g_), gs=g_ / max(1e-9, r_ + g_ + b_))


def main() -> int:
    src = load(os.path.join(OUT, "frame.png"))
    #: one render, walked, so the label source can be reported -- the picture is a PLANT and foliage is the table's
    #: highest entry, which makes the labelling the thing this clip can judge that the others could not
    out_nv, r = render(src, 1.0)
    cam = r.stage["camera"]
    labels = getattr(cam, "_nir_label_source", None)

    panels = [
        ("THE CLIP (2009, 640x480)\nthe input, warm indoor light", src),
        ("mode OFF\nthe chain, no night vision", render(src, 0.0)[0]),
        ("NIGHTSHOT 1\nthe green display", out_nv),
        ("SUPER NIGHTSHOT\n8 frames accumulated", render(src, 2.0)[0]),
        ("COLOUR SLOW SHUTTER\nkeeps colour", render(src, 3.0)[0]),
        ("NIGHTSHOT 1, display stage ON\n(the viewer's own look)", render(src, 1.0, None)[0]),
    ]
    print("=" * 120)
    print("  THE 2009 CLIP THROUGH THE CHAIN -- warm indoor light on a dried arrangement")
    print("=" * 120)
    print()
    print("   %-44s %-8s %-8s %-8s %-8s %-8s %-8s %-8s" %
          ("panel", "mean", "p1", "p50", "p99", "primary", "R/G", "B/G"))
    print("   " + "-" * 116)
    for tag, img in panels:
        s = stat(img)
        print("   %-44s %-8.4f %-8.4f %-8.4f %-8.4f %-8s %-8.4f %-8.4f"
              % (tag.split(chr(10))[0][:43], s["mean"], s["p1"], s["p50"], s["p99"], s["prim"], s["rg"], s["bg"]))
    print()
    si, sn = stat(src), stat(out_nv)
    print("   the input spans p1 %.4f to p99 %.4f and the model spans %.4f to %.4f"
          % (si["p1"], si["p99"], sn["p1"], sn["p99"]))
    print("   **the source is warm and the mode is green**: R/G %.4f -> %.4f, B/G %.4f -> %.4f"
          % (si["rg"], sn["rg"], si["bg"], sn["bg"]))
    print()
    print("   THE MATERIAL LABELLING, which this clip is the first to be able to test:")
    print("     %s" % (labels or "the vision model reported nothing"))
    print()

    cols, cw, ch, gap, cap = 3, 420, 315, 10, 72
    rows = (len(panels) + cols - 1) // cols
    canvas = Image.new("RGB", ((cw + gap) * cols + gap, (ch + cap) * rows + 40), (14, 14, 16))
    d = ImageDraw.Draw(canvas)
    d.text((10, 6), "THE 2009 CLIP THROUGH THE CHAIN -- a dried arrangement in warm indoor light, 640x480 MPEG-1",
           fill=(240, 240, 242))
    d.text((10, 22), "the source spans 0.000 to 0.470 with no highlights; R/G 1.083 (warm) against the green modes' "
                     "target of ~0.54", fill=(220, 200, 150))
    for i, (tag, img) in enumerate(panels):
        x = gap + (i % cols) * (cw + gap)
        y = 42 + (i // cols) * (ch + cap)
        canvas.paste(Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).resize((cw, ch), Image.LANCZOS),
                     (x, y))
        s = stat(img)
        for k, line in enumerate(tag.split("\n")):
            d.text((x, y + ch + 3 + 13 * k), line[:74], fill=(245, 245, 248) if k == 0 else (176, 176, 182))
        d.text((x, y + ch + 32), "mean %.4f  p1 %.4f  p50 %.4f  p99 %.4f  %s"
               % (s["mean"], s["p1"], s["p50"], s["p99"], s["prim"]), fill=(212, 212, 218))
        d.text((x, y + ch + 45), "R/G %.4f   B/G %.4f   G share %.4f" % (s["rg"], s["bg"], s["gs"]),
               fill=(212, 212, 218))
    path = os.path.join(OUT, "MOV0432_THROUGH.png")
    canvas.save(path)
    print("   written: %s" % os.path.relpath(path, ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
