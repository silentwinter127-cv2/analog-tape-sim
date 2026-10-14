"""
The grain and the highlights, on both the new photograph and the DV frame.

The user's report: the wall has grubby blotches that the real footage does not have, the fan LEDs should be brighter
with a halo, and **the noise texture is completely different -- it looks coarse, and it may not be a night-vision
problem.**  The last clause turned out to be the important one.

**Two measurements decided it.**

    grain (lag-1 autocorrelation of the row difference) and detail (mean |dx|)

    the visible-light photograph   in  0.083 / 0.02326
    the model on it                    0.850 / 0.00396      <- 17% of the detail, grain 10x
    the noisiest setting available     0.845 / 0.00396      <- raising the noise does not touch it

**A high lag-1 with almost no detail is not coarse noise; it is a smoothed picture.**  When the high frequencies are
gone, what is left is low-frequency variation, and the eye reads that as blotchy grain.  So the coarse look comes from
the picture being *too smooth*, and it is present with **night vision off and every noise term at zero** -- which is
exactly what the user suspected.

Nothing is retuned here.  The panels are the picture, an aggressive sharpening, and the darkest and brightest ends,
so that the two ends can be judged separately from the grain.
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
from nv_targets import crop_osd                                              # noqa: E402

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


def load(p, frac=0.20):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0
    a = crop_osd(a, frac) if frac else a
    return np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS),
                      np.float32) / 255.0


photo = load(os.path.join(OUT, "new_photo.jpg"), 0.0)
dv = load(os.path.join(OUT, "sc1515", "d_nv3.png"))


def run(src, mode=1.0, cam=None, fmt=None):
    lay = {"camera": dict(nv_mode=mode, **(cam or {})),
           "format": dict({"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0}, **(fmt or {})),
           "player": dict({"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0}, **(fmt or {})),
           "codec": {"dv_enabled": 1.0}}
    r = Renderer(Still(src), W, H, 25.0, merge_values(BASE, lay), seed=20240101)
    return np.clip(np.asarray(r.render_frame(0), np.float32), 0, 1)


def grain(x):
    l = np.clip(np.asarray(x, np.float32) @ LUMA, 0, 1)
    d = l[:, 1:] - l[:, :-1]
    return (float((d[:, :-1] * d[:, 1:]).mean() / max(1e-12, (d * d).mean())),
            float(np.abs(d).mean()), float(np.quantile(l, 0.99)), float(np.quantile(l, 0.01)))


def main() -> int:
    variants = [
        ("THE NEW PHOTOGRAPH\nvisible light, the input", photo, None),
        ("the model on it, mode OFF\nno night vision at all", run(photo, 0.0), "mode OFF"),
        ("the model on it, NIGHTSHOT 1\n(as shipped)", run(photo, 1.0), "as shipped"),
        ("the model, sharpness at max\n(the only sharpening knob there is)", run(photo, 1.0, {"sharpness": 4.0}),
         "sharp 4.0"),
        ("the model, sensor noise raised\n(the grain knob, for comparison)",
         run(photo, 1.0, {"sensor_noise": 0.40, "noise_grain": 3.0}), "noise 0.40"),
        ("THE DV FRAME\nthe input that has the highlights", dv, None),
        ("the model on the DV frame\nNIGHTSHOT 1", run(dv, 1.0), "as shipped"),
        ("the model on the DV frame\nsaturation_stops 1.5 (lets the top clip)",
         run(dv, 1.0, {"nv_saturation_stops": 1.5}), "sat 1.5"),
    ]
    print("=" * 122)
    print("  THE GRAIN AND THE TWO ENDS -- grain is lag-1 of the row difference, detail is mean |dx|")
    print("=" * 122)
    print()
    print("   %-52s %-9s %-9s %-8s %-8s %s" % ("panel", "grain", "detail", "p99", "p1", "what it is"))
    print("   " + "-" * 118)
    for tag, img, note in variants:
        g, d, p99, p1 = grain(img)
        print("   %-52s %-9.3f %-9.5f %-8.4f %-8.4f %s" % (tag.split(chr(10))[0][:51], g, d, p99, p1, note or ""))
    print()
    gp, dp, _, _ = grain(photo)
    gm, dm, _, _ = grain(run(photo, 1.0))
    print("   the photograph carries detail %.5f; the model delivers %.5f -- %.0f%% of it."
          % (dp, dm, 100 * dm / dp))
    print("   **and raising the noise does not change that** -- the grain knob moves %.5f to %.5f."
          % (dm, grain(run(photo, 1.0, {"sensor_noise": 0.40, "noise_grain": 3.0}))[1]))
    print("   so 'the noise looks coarse' is the picture being SMOOTH: with the top frequencies gone, what is left")
    print("   is low-frequency variation, and that is what the eye reads as blotchy grain.")
    print()

    cols, cw, ch, gap, cap = 4, 330, 248, 8, 70
    rows = (len(variants) + cols - 1) // cols
    canvas = Image.new("RGB", ((cw + gap) * cols + gap, (ch + cap) * rows + 40), (14, 14, 16))
    d = ImageDraw.Draw(canvas)
    d.text((10, 6), "THE GRAIN AND THE TWO ENDS -- the coarse look is a SMOOTH picture, and it is there with "
                    "night vision off", fill=(240, 240, 242))
    d.text((10, 22), "the model delivers 17% of the photograph's detail; the noise knob does not change that",
           fill=(220, 200, 150))
    for i, (tag, img, note) in enumerate(variants):
        x = gap + (i % cols) * (cw + gap)
        y = 42 + (i // cols) * (ch + cap)
        canvas.paste(Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).resize((cw, ch), Image.LANCZOS),
                     (x, y))
        g, dd, p99, p1 = grain(img)
        for k, line in enumerate(tag.split("\n")):
            d.text((x, y + ch + 3 + 13 * k), line[:64], fill=(245, 245, 248) if k == 0 else (176, 176, 182))
        d.text((x, y + ch + 32), "grain %.3f  detail %.5f" % (g, dd), fill=(212, 212, 218))
        d.text((x, y + ch + 45), "p1 %.4f   p99 %.4f" % (p1, p99), fill=(212, 212, 218))
    path = os.path.join(OUT, "GRAIN_AND_ENDS.png")
    canvas.save(path)
    print("   written: %s" % os.path.relpath(path, ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
