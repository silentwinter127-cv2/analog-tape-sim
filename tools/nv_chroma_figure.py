"""
NIGHTSHOT 2 against the real frame: the colour content, and the display drive that was too low.

The user's report moved twice and both moves were right.  First: "still differs from the real NIGHTSHOT 2 picture" --
the hue was yellow-green against the frames' cyan-green.  Then: **"四张图根本没有除了绿色之外的彩色"** -- and this
round's frame settles what that means, because it is measurable:

    the real NIGHTSHOT 2 frame (c_0147)   chroma 0.10128   R/G 0.4739   B/G 0.8123
    the model at the shipped drive        chroma 0.07462   R/G 0.5374   B/G 0.8718

**The real frame carries MORE colour than a pure monochrome conversion produces.**  That is not inverted logic; it says
the display was under-driven, and the mechanism is the one the parameter is named for:

    nv_green_gain   chroma    R/G      B/G
    1.00            0.07462   0.5374   0.8718
    1.20            0.09281   0.5242   0.7807
    1.50            0.12743   0.4792   0.6847
    the frame       0.10128   0.4739   0.8123

so B/G crosses its target near 1.22 and the chroma near 1.3, and **1.25 is where both are closest at once**.  It is the
green primary's level -- above 1.0 the gun is driven harder -- so it moves chroma, R/G and B/G together rather than
being a grade applied to one channel.

**And the mix sweep settled the other half of the question, against the obvious reading.**  Keeping the scene's own
colour -- `green_mix` below 1 -- **lowers** the chroma rather than raising it:

    green_mix 1.00 -> 0.07462    0.55 -> 0.06537    0.00 -> 0.05469

because the mono conversion puts every pixel on the display's ratios, and those are themselves far from grey, so the
tint is *larger* than the scene's residual colour.  A green display is not a desaturation.
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


def load(p, frac=0.0, size=(W, H)):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0
    if frac:
        a = a[int(a.shape[0] * frac):]
    return np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize(size, Image.LANCZOS),
                      np.float32) / 255.0


def run(src, gain=None, mix=1.0):
    cam = {"nv_mode": 2.0, "nv_green_mix": mix}
    if gain is not None:
        cam["nv_green_gain"] = gain
    lay = {"camera": cam, "format": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0},
           "player": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0},
           "codec": {"dv_enabled": 1.0}}
    r = Renderer(Still(src), W, H, 25.0, merge_values(BASE, lay), seed=20240101)
    return np.clip(np.asarray(r.render_frame(0), np.float32), 0, 1)


def stat(x):
    a = np.asarray(x, np.float32)
    l = np.clip(a @ LUMA, 0, 1)
    r, g, b = (float(a[..., i].mean()) for i in range(3))
    return dict(ch=float(np.abs(a - (a @ LUMA)[..., None]).mean()), rg=r / max(1e-9, g),
                bg=b / max(1e-9, g), mean=float(l.mean()), blown=100 * float((l > 0.90).mean()))


def main() -> int:
    real = load(os.path.join(OUT, "ref_c0147.png"))
    dv = load(os.path.join(OUT, "sc1515", "d_nv3.png"), 0.20)

    panels = [
        ("THE REAL NIGHTSHOT 2 (c_0147)\nnative 1440x1080, chroma 0.1013",
         load(os.path.join(OUT, "ref_c0147.png"), 0.0, (W, int(W * 1080 / 1440)))),
        ("the model, the shipped drive 1.0\nchroma 0.0746  -- too little tint", run(dv, 1.0)),
        ("the model, drive 1.25\n** the value set now **", run(dv, None)),
        ("the model, drive 1.50\nchroma 0.1274  -- past it", run(dv, 1.5)),
        ("the model, drive 1.25, mix 0.55\n(keeping scene colour lowers chroma)", run(dv, None, 0.55)),
        ("the model on the user's own frame\ndrive 1.25", run(real, None)),
    ]
    print("=" * 122)
    print("  NIGHTSHOT 2 -- the colour content, and the display drive that was too low")
    print("=" * 122)
    print()
    print("   %-50s %-9s %-9s %-9s %-9s %-8s" % ("panel", "chroma", "R/G", "B/G", "mean", ">0.90"))
    print("   " + "-" * 116)
    for tag, img in panels:
        s = stat(img)
        print("   %-50s %-9.5f %-9.4f %-9.4f %-9.4f %-8.2f"
              % (tag.split(chr(10))[0][:49], s["ch"], s["rg"], s["bg"], s["mean"], s["blown"]))
    print()
    sr, sm = stat(real), stat(run(dv, None))
    print("   the real frame's chroma is %.5f and the model's is now %.5f -- %.0f%% of it."
          % (sr["ch"], sm["ch"], 100 * sm["ch"] / sr["ch"]))
    print("   its R/G is %.4f against %.4f, and its B/G %.4f against %.4f."
          % (sr["rg"], sm["rg"], sr["bg"], sm["bg"]))
    print("   **and its clipping is %.2f%% against the model's %.2f%% -- that is the remaining gap.**"
          % (sr["blown"], sm["blown"]))
    print()

    cols, cw, ch, gap, cap = 3, 430, 322, 10, 74
    rows = (len(panels) + cols - 1) // cols
    canvas = Image.new("RGB", ((cw + gap) * cols + gap, (ch + cap) * rows + 40), (14, 14, 16))
    d = ImageDraw.Draw(canvas)
    d.text((10, 6), "NIGHTSHOT 2 -- the display was under-driven: the real frame carries MORE colour than a pure "
                    "monochrome conversion makes", fill=(240, 240, 242))
    d.text((10, 22), "chroma 0.0746 -> 0.0984 against the frame's 0.1013; B/G 0.872 -> 0.762 against 0.812",
           fill=(220, 200, 150))
    for i, (tag, img) in enumerate(panels):
        x = gap + (i % cols) * (cw + gap)
        y = 42 + (i // cols) * (ch + cap)
        canvas.paste(Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).resize((cw, ch), Image.LANCZOS),
                     (x, y))
        s = stat(img)
        for k, line in enumerate(tag.split("\n")):
            d.text((x, y + ch + 3 + 13 * k), line[:76], fill=(245, 245, 248) if k == 0 else (176, 176, 182))
        d.text((x, y + ch + 32), "chroma %.5f   R/G %.4f   B/G %.4f" % (s["ch"], s["rg"], s["bg"]),
               fill=(212, 212, 218))
        d.text((x, y + ch + 45), "mean %.4f   >0.90 %.2f%%" % (s["mean"], s["blown"]), fill=(212, 212, 218))
    path = os.path.join(OUT, "NS2_CHROMA.png")
    canvas.save(path)
    print("   written: %s" % os.path.relpath(path, ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
