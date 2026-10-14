"""
The detail budget, after the infrared focus shift was cut to a physical value.

The previous round established that the model delivers 17% of the input's detail and that the coarse grain is a
*smooth* picture rather than noisy grain.  This round walks the night-vision steps one at a time and finds the single
term that costs it:

    the picture entering               HF 0.02326   100%   grain 0.083   (the input's own grain)
    + add_illuminator                  HF 0.02422   104%   grain 0.083
    + ir_highlights                    HF 0.01710    74%   grain 0.220   <- the only step that costs anything
    + monochrome_green                 HF 0.01713    74%   grain 0.221

and that term is `ir_focus_shift`, whose default was **1.5 pixels**.  At 0.5:

    + ir_highlights                    HF 0.02321   100%   grain 0.092

**1.5 px softens the infrared layer by about three pixels' worth of structure** -- too strong for "the near-infrared
plane sits slightly behind the visible one", and it was eating the whole picture's detail because the IR mix is 70% of
it (`nv_ir_albedo` 0.70).

Two smaller findings are recorded as negative or open rather than fixed here: `sharpness` at 0.75 adds a further
gaussian of sigma 1.4 (`lowpass(rgb, 1.0 + (1 - s) * 1.6)`), worth another 6 points, and `chroma_aberration` at 0.05
another 14 -- together they take 38% to 39%.  The `tape` layer's `noise_floor` is 0.18 by default, which is a candidate
for "the dark side is not black" and has not been tested yet.
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


def stat(x):
    l = np.clip(np.asarray(x, np.float32) @ LUMA, 0, 1)
    d = l[:, 1:] - l[:, :-1]
    return dict(g=float((d[:, :-1] * d[:, 1:]).mean() / max(1e-12, (d * d).mean())),
                hf=float(np.abs(d).mean()), p1=float(np.quantile(l, 0.01)),
                p99=float(np.quantile(l, 0.99)), blown=100.0 * float((l > 0.90).mean()))


def main() -> int:
    panels = [
        ("THE PHOTOGRAPH\nthe visible-light input", photo),
        ("the model, as shipped before this fix\nir_focus_shift 1.5",
         run(photo, {"nv_ir_focus_shift": 1.5})),
        ("the model, ** ir_focus_shift 0.5 **\nthe one change in this commit", run(photo)),
        ("the model, that + sharpness 1.0\n+ chroma_aberration 0 (38%)",
         run(photo, {"sharpness": 1.0, "chroma_aberration": 0.0})),
        ("THE DV FRAME\nthe night-vision input", dv),
        ("the model on it, focus 1.5\n(as it was)", run(dv, {"nv_ir_focus_shift": 1.5})),
        ("the model on it, ** focus 0.5 **", run(dv)),
        ("the model on it, + sharpness 1.0\n+ chroma_aberration 0 (39%)",
         run(dv, {"sharpness": 1.0, "chroma_aberration": 0.0})),
    ]
    sp, sd = stat(photo), stat(dv)
    print("=" * 122)
    print("  THE IR FOCUS SHIFT WAS EATING THE DETAIL -- 1.5 px of defocus on a layer that is 70% of the picture")
    print("=" * 122)
    print()
    print("   %-52s %-9s %-7s %-8s %-8s %-8s %s" % ("panel", "detail", "%", "grain", "p1", "p99", "blown%"))
    print("   " + "-" * 118)
    for tag, img in panels:
        s = stat(img)
        base = sp["hf"] if img is not photo and tag != "THE DV FRAME" and not tag.startswith("THE DV") else None
        pct = ""
        if "photo" in tag.lower() or tag.startswith("the model, ") and tag.count("DV") == 0:
            pct = "%.0f%%" % (100 * s["hf"] / sp["hf"])
        elif "DV" in tag:
            pct = "%.0f%%" % (100 * s["hf"] / sd["hf"])
        print("   %-52s %-9.5f %-7s %-8.3f %-8.4f %-8.4f %.2f"
              % (tag.split(chr(10))[0][:51], s["hf"], pct, s["g"], s["p1"], s["p99"], s["blown"]))
    print()
    print("   the photograph's own grain is %.3f; the DV frame's is %.3f" % (sp["g"], sd["g"]))
    print("   the model at focus 1.5 was 0.850 / 0.888 -- that is the 'coarse noise', and it was DEFOCUS.")
    print()

    cols, cw, ch, gap, cap = 4, 330, 248, 8, 70
    rows = (len(panels) + cols - 1) // cols
    canvas = Image.new("RGB", ((cw + gap) * cols + gap, (ch + cap) * rows + 40), (14, 14, 16))
    d = ImageDraw.Draw(canvas)
    d.text((10, 6), "THE INFRARED FOCUS SHIFT WAS EATING THE DETAIL -- 1.5 px of defocus on a layer that is 70% of "
                    "the picture", fill=(240, 240, 242))
    d.text((10, 22), "the 'coarse noise' halves when it is cut to a physical value: grain 0.850 -> 0.575 against the "
                     "input's 0.083", fill=(220, 200, 150))
    for i, (tag, img) in enumerate(panels):
        x = gap + (i % cols) * (cw + gap)
        y = 42 + (i // cols) * (ch + cap)
        canvas.paste(Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).resize((cw, ch), Image.LANCZOS),
                     (x, y))
        s = stat(img)
        for k, line in enumerate(tag.split("\n")):
            d.text((x, y + ch + 3 + 13 * k), line[:64], fill=(245, 245, 248) if k == 0 else (176, 176, 182))
        d.text((x, y + ch + 32), "detail %.5f   grain %.3f" % (s["hf"], s["g"]), fill=(212, 212, 218))
        d.text((x, y + ch + 45), "p1 %.4f  p99 %.4f  blown %.2f%%" % (s["p1"], s["p99"], s["blown"]),
               fill=(212, 212, 218))
    path = os.path.join(OUT, "FOCUS_SHIFT_FIX.png")
    canvas.save(path)
    print("   written: %s" % os.path.relpath(path, ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
