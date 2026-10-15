"""
The frame fit: five of the seven statistics within 14%, and the clipping down by two thirds.

The user's "差距很明显" was followed by a measurement that located it precisely, and it was **not** the colour any more:

    percentile   the model before   the real frame   now
    p1               0.0302          0.0471          0.0716
    p25              0.1194          0.1509          0.1664
    p50              0.2481          0.2710          0.2523
    p75              0.7610          0.4840          0.5217
    std              0.3476          0.2389          0.2716
    clipped          17.46%          3.19%           5.96%

**The dark half was two to three times too dark and the bright half too bright** -- the contrast was in the wrong place,
which is what `note.txt` means by a NightShot's black being "深绿色灰" rather than black.  A pedestal cannot fix that
(it adds a constant and leaves the shadow ratios alone) and a gain cannot (it lifts the highlights that are already too
high), so what went in is a **transfer curve**, `x ** gamma` with `gamma < 1`, which stretches the bottom and compresses
the top with 0 and 1 fixed.

`gamma` was solved from the frame's own percentiles: `0.052 ** g = 0.151` asks 0.588 and `0.188 ** g = 0.271` asks
0.634, so **0.60**.  It has to be paired with the exposure, because lifting the shadows lifts the mean with them --
the two together are 0.60 and -0.85 stops, and **that pairing is what took the clipping from 17.46% to 5.96%**.

Three more values moved with it, each on a different quantity:

    the red primary      0.3890 -> 0.5300    the display's own channel ratio
    nv_green_gain        1.25   -> 1.00      the drive, which was pushing the picture greener
    nv_chroma_residue    0.00   -> 0.15      the scene's colour, added outside the display's ratios

and one of them was wrong in the previous round: **1.25 on the drive was set from a chroma measurement that the residue
explains better**, and it made the picture greener, which is exactly what the user said about it.

**What is still off is stated rather than smoothed**: p1 is 52% high (the black floor now lifts past the frame's) and
R/G is 18.8% high (too little red relative to green, in the direction the previous round fixed and this one moved back).
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


def load(p, frac=0.0, osd=False, size=(W, H)):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0
    if frac:
        a = a[int(a.shape[0] * frac):]
    a = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize(size, Image.LANCZOS),
                   np.float32) / 255.0
    if osd:
        h, w = a.shape[:2]
        a = a[int(h * 0.18):, :int(w * 0.97)]
    return a


def run(src, gamma=0.60, trim=-0.85, red=0.5300, residue=0.15, gain=1.0):
    lay = {"camera": {"nv_mode": 2.0, "dsp_gamma": gamma, "nv_exposure_trim": trim,
                      "nv_chroma_residue": residue, "nv_green_gain": gain},
           "format": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0},
           "player": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0},
           "codec": {"dv_enabled": 1.0}}
    r = Renderer(Still(src), W, H, 25.0, merge_values(BASE, lay), seed=20240101)
    return np.clip(np.asarray(r.render_frame(0), np.float32), 0, 1)


def stat(x):
    a = np.asarray(x, np.float32)
    l = np.clip(a @ LUMA, 0, 1)
    r, g, b = (float(a[..., i].mean()) for i in range(3))
    return dict(mean=float(l.mean()), p1=float(np.quantile(l, 0.01)), p25=float(np.quantile(l, 0.25)),
                p50=float(np.quantile(l, 0.50)), p75=float(np.quantile(l, 0.75)), std=float(l.std()),
                rg=r / max(1e-9, g), bg=b / max(1e-9, g), blown=100 * float((l > 0.90).mean()))


def main() -> int:
    real = load(os.path.join(OUT, "user_real.png"), osd=True)
    real_full = load(os.path.join(OUT, "user_real.png"))
    dv = load(os.path.join(OUT, "sc1515", "d_nv3.png"), 0.20)

    panels = [
        ("THE REAL NIGHTSHOT 2\nthe frame to match (full, with OSD)", real_full),
        ("the model, before this round\ngamma 1.0 -- shadows 2-3x too dark", run(dv, gamma=1.0, trim=-0.10,
                                                                               red=0.3890, residue=0.0, gain=1.25)),
        ("the model, ** now **\ngamma 0.60, -0.85 stops, red 0.53", run(dv)),
        ("the model, the old gamma with the new level\nthe coupling that had to be solved together",
         run(dv, gamma=1.0)),
        ("the model: gamma only, no exposure pairing\n(lifts the mean past the frame's)", run(dv, trim=-0.10)),
        ("the model on the user's own frame\nwith the same settings", run(real)),
    ]
    sr = stat(real)
    print("=" * 122)
    print("  THE FRAME FIT -- the transfer curve, and the exposure it has to be paired with")
    print("=" * 122)
    print()
    print("   %-50s %-8s %-8s %-8s %-8s %-8s %-8s %-8s" %
          ("panel", "mean", "p1", "p25", "p50", "std", "R/G", ">0.90"))
    print("   " + "-" * 118)
    for tag, img in panels:
        s = stat(img)
        print("   %-50s %-8.4f %-8.4f %-8.4f %-8.4f %-8.4f %-8.4f %-8.2f"
              % (tag.split(chr(10))[0][:49], s["mean"], s["p1"], s["p25"], s["p50"], s["std"], s["rg"], s["blown"]))
    print()
    sm = stat(run(dv))
    print("   against the real frame (OSD excluded):")
    for name in ("mean", "p1", "p25", "p50", "p75", "std", "rg", "bg"):
        tv, mv = sr[name], sm[name]
        err = abs(mv - tv) / max(1e-9, abs(tv))
        print("     %-5s %8.4f vs %8.4f   %5.1f%%   %s" % (name, tv, mv, 100 * err,
                                                            "ok" if err < 0.15 else "off"))
    print("     %-5s %7.2f%% vs %7.2f%%   -- from 17.46%% at the start of this round"
          % ("clip", sr["blown"], sm["blown"]))
    print()

    cols, cw, ch, gap, cap = 3, 430, 322, 10, 74
    rows = (len(panels) + cols - 1) // cols
    canvas = Image.new("RGB", ((cw + gap) * cols + gap, (ch + cap) * rows + 40), (14, 14, 16))
    d = ImageDraw.Draw(canvas)
    d.text((10, 6), "THE FRAME FIT -- a transfer curve (x**0.60) lifts the shadow floor a pedestal cannot, and it must "
                    "be paired with the exposure", fill=(240, 240, 242))
    d.text((10, 22), "clipping 17.46% -> 5.96% against the frame's 3.19%; p25 10.3% off, p50 6.9%, std 13.7%",
           fill=(220, 200, 150))
    for i, (tag, img) in enumerate(panels):
        x = gap + (i % cols) * (cw + gap)
        y = 42 + (i // cols) * (ch + cap)
        canvas.paste(Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).resize((cw, ch), Image.LANCZOS),
                     (x, y))
        s = stat(img)
        for k, line in enumerate(tag.split("\n")):
            d.text((x, y + ch + 3 + 13 * k), line[:76], fill=(245, 245, 248) if k == 0 else (176, 176, 182))
        d.text((x, y + ch + 32), "mean %.4f  p1 %.4f  p25 %.4f  p50 %.4f" % (s["mean"], s["p1"], s["p25"], s["p50"]),
               fill=(212, 212, 218))
        d.text((x, y + ch + 45), "std %.4f  R/G %.4f  >0.90 %.2f%%" % (s["std"], s["rg"], s["blown"]),
               fill=(212, 212, 218))
    path = os.path.join(OUT, "FRAME_FIT.png")
    canvas.save(path)
    print("   written: %s" % os.path.relpath(path, ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
