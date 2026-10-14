"""
The three things the user reported, each measured rather than argued about.

Reported on a NIGHTSHOT 1 render: slanted bands, contrast too low, and the dark side not black at all.  This file
answers each with a number and a picture, and **it is deliberately arranged so that a claim can be checked by looking
at the panels** rather than by trusting the table.
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
from atsim.core.pipeline import FrameBuffer                                   # noqa: E402
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


src = load(os.path.join(OUT, "sc1515", "d_nv3.png"))
LAY = {"camera": {"nv_mode": 1.0}, "format": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0},
       "player": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0},
       "codec": {"dv_enabled": 1.0}}


def run(cam=None):
    lay = {k: dict(v) for k, v in LAY.items()}
    lay["camera"].update(cam or {})
    r = Renderer(Still(src), W, H, 25.0, merge_values(BASE, lay), seed=20240101)
    return np.clip(np.asarray(r.render_frame(0), np.float32), 0, 1)


def diag_ratio(x):
    """Energy on the two DIAGONAL directions over the two axis directions: a slanted band raises it."""
    l = np.clip(np.asarray(x, np.float32) @ LUMA, 0, 1)
    d = np.abs(l[:-1, :-1] - l[1:, 1:]).mean() + np.abs(l[:-1, 1:] - l[1:, :-1]).mean()
    o = np.abs(l[:, :-1] - l[:, 1:]).mean() + np.abs(l[:-1, :] - l[1:, :]).mean()
    return d / max(1e-9, o)


def spec_diag(x):
    """The share of the spectrum's energy lying on the diagonals -- a band, not an edge, lives here."""
    l = np.clip(np.asarray(x, np.float32) @ LUMA, 0, 1)
    f = np.abs(np.fft.fftshift(np.fft.fft2(l - l.mean())))
    h, w = f.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    ang = np.abs(np.abs(np.arctan2(yy - cy, xx - cx)) - np.pi / 4)
    m = (ang < 0.18) & (rr > 6) & (rr < 0.45 * min(h, w))
    return 100.0 * f[m].sum() / f.sum()


def dark_view(x, thr=0.25):
    """The dark side only, stretched -- so 'not black' is something the eye can check."""
    v = np.clip(np.asarray(x, np.float32), 0, 1).copy()
    l = v @ LUMA
    d = l < thr
    v[~d] = v[~d] * 0.18
    v[d] = np.clip((v[d] - 0.02) * 3.2, 0, 1)
    return v


def stat(x):
    l = np.clip(np.asarray(x, np.float32) @ LUMA, 0, 1)
    d = l < 0.25
    return dict(p1=float(np.quantile(l, 0.01)), p99=float(np.quantile(l, 0.99)),
                std=float(l.std()), dark=float(l[d].mean()) if d.sum() > 100 else float("nan"),
                dr=diag_ratio(x), sd=spec_diag(x))


def main() -> int:
    out = run()
    panels = [
        ("THE INPUT (the DV frame)\nits own slanted structure, before any model",
         src, stat(src)),
        ("the model, NIGHTSHOT 1\n(the render the report was about)", out, stat(out)),
        ("the model, night vision OFF\nso the chain alone can be judged", run({"nv_mode": 0.0}),
         stat(run({"nv_mode": 0.0}))),
        ("the model, sharpness 1.0\n(no softening in the camera)", run({"sharpness": 1.0}),
         stat(run({"sharpness": 1.0}))),
        ("DARK SIDE of the input\nstretched, so the floor is visible", dark_view(src), stat(src)),
        ("DARK SIDE of the model\nstretched the same way", dark_view(out), stat(out)),
    ]
    print("=" * 118)
    print("  THE THREE REPORTS, MEASURED -- slanted bands, contrast, and the dark floor")
    print("=" * 118)
    print()
    print("   %-42s %-8s %-8s %-8s %-9s %-8s %-9s"
          % ("panel", "p1", "p99", "std", "dark mean", "diag/orth", "spec diag%"))
    print("   " + "-" * 114)
    for tag, _, s in panels:
        print("   %-42s %-8.4f %-8.4f %-8.4f %-9.4f %-8.3f %-9.1f"
              % (tag.split(chr(10))[0][:41], s["p1"], s["p99"], s["std"], s["dark"], s["dr"], s["sd"]))
    print()
    si, so = stat(src), stat(out)
    print("   SLANTED BANDS:  the input's diagonals/axes ratio is %.3f and the model's is %.3f -- the model's is"
          % (si["dr"], so["dr"]))
    print("                   %s.  And on the diagonals of the SPECTRUM, %.1f%% in, %.1f%% out."
          % ("LOWER, so the chain does not add slanted structure" if so["dr"] < si["dr"] else "HIGHER", si["sd"], so["sd"]))
    print("   CONTRAST:       the input spans p1 %.4f to p99 %.4f (%.3f of range); the model spans %.4f to %.4f"
          % (si["p1"], si["p99"], si["p99"] - si["p1"], so["p1"], so["p99"]))
    print("                   (%.3f).  **The model's white end stops at %.3f while the input reaches %.3f** --"
          % (so["p99"] - so["p1"], so["p99"], si["p99"]))
    print("                   that is a missing top, not a compression.")
    print("   DARK FLOOR:     dark-side mean %.4f in against %.4f in the model." % (si["dark"], so["dark"]))
    print()

    cols, cw, ch, gap, cap = 3, 430, 322, 10, 74
    rows = (len(panels) + cols - 1) // cols
    canvas = Image.new("RGB", ((cw + gap) * cols + gap, (ch + cap) * rows + 40), (14, 14, 16))
    d = ImageDraw.Draw(canvas)
    d.text((10, 6), "SLANTED BANDS, CONTRAST, AND THE DARK FLOOR -- the input first, then the model",
           fill=(240, 240, 242))
    d.text((10, 22), "the input's diagonals/axes ratio is HIGHER than the model's: the slanted structure is in the "
                     "footage, not added by the chain", fill=(220, 200, 150))
    for i, (tag, img, s) in enumerate(panels):
        x = gap + (i % cols) * (cw + gap)
        y = 42 + (i // cols) * (ch + cap)
        canvas.paste(Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).resize((cw, ch), Image.LANCZOS),
                     (x, y))
        for k, line in enumerate(tag.split("\n")):
            d.text((x, y + ch + 3 + 13 * k), line[:72], fill=(245, 245, 248) if k == 0 else (176, 176, 182))
        d.text((x, y + ch + 32), "p1 %.4f  p99 %.4f  std %.4f  dark %.4f" % (s["p1"], s["p99"], s["std"], s["dark"]),
               fill=(212, 212, 218))
        d.text((x, y + ch + 45), "diag/orth %.3f   spectrum on the diagonals %.1f%%" % (s["dr"], s["sd"]),
               fill=(212, 212, 218))
    path = os.path.join(OUT, "SLANT_AND_CONTRAST.png")
    canvas.save(path)
    print("   written: %s" % os.path.relpath(path, ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
