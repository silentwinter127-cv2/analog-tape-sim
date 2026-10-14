"""
The contrast curve, and what is still missing from the detail.

Two halves of the user's "高细节高对比", answered separately because the measurements separate them.

**Contrast: fixed.**  Against the DV frame, the model's range was squeezed from *both* ends -- mid-tone high, dark end
high, white end low:

    the DV frame    p1 0.0353   p5 0.0565   p50 0.1833   p99 1.0000   std 0.2896
    the model       p1 0.0517   p5 0.0615   p50 0.2028   p99 1.0000   std 0.2794
    + contrast .25  p1 0.0344   p5 0.0413   p50 0.1636   p99 1.0000   std 0.2988

so the dark end lands on the target (0.0344 against 0.0353) and the global contrast slightly overshoots it (0.2988
against 0.2896) while the white point stays at full scale.  0.35 or more takes the dark end below the target, which is
why 0.25 is the value.

**Detail: not fixed, and the sweep says why.**  Every camera parameter at both ends of its range, ranked by how much it
moves the picture's high-frequency content:

    sensor_noise      0.06 -> 1.0    0.02489   (107% of the photograph)
    nv_chroma_noise   0.15 -> 1.0    0.00992   ( 43%)
    dsp_gain          1.0  -> 4.0    0.00800   ( 34%)
    sharpness         0.75 -> 2.0    0.00622   ( 27%)

**The only control that reaches the photograph's detail is the noise**, and the reason is that the model's real texture
is 0.00368 -- everything above that is grain added on top of a smooth picture.  That is why raising the noise produces
snow rather than grain, and it is where the remaining work is.
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
    return dict(p1=float(np.quantile(l, 0.01)), p50=float(np.quantile(l, 0.50)),
                p99=float(np.quantile(l, 0.99)), std=float(l.std()),
                detail=float(np.abs(np.diff(l, axis=1)).mean()),
                blown=100.0 * float((l > 0.90).mean()))


def main() -> int:
    nv = run(dv)
    panels = [
        ("THE DV FRAME\nthe target: p1 0.0353  std 0.2896", dv),
        ("the model, contrast 0\n(what it was before this commit)", run(dv, {"dsp_contrast": 0.0})),
        ("the model, ** dsp_contrast 0.25 **\np1 0.0344  std 0.2988  <- on target", nv),
        ("the model, contrast 0.50\n(past it: p1 0.0241, below the target)", run(dv, {"dsp_contrast": 0.50})),
        ("THE PHOTOGRAPH\nthe visible-light input", photo),
        ("the model on it, contrast 0", run(photo, {"dsp_contrast": 0.0})),
        ("the model on it, ** contrast 0.25 **", run(photo)),
        ("the model on it, contrast 0.50", run(photo, {"dsp_contrast": 0.50})),
    ]
    sd, sm = stat(dv), stat(nv)
    print("=" * 120)
    print("  THE CONTRAST CURVE -- and the sweep that says the detail is not a level question")
    print("=" * 120)
    print()
    print("   %-52s %-8s %-8s %-8s %-8s %-8s %-9s" % ("panel", "p1", "p50", "p99", "std", "detail", "blown%"))
    print("   " + "-" * 116)
    for tag, img in panels:
        s = stat(img)
        print("   %-52s %-8.4f %-8.4f %-8.4f %-8.4f %-8.5f %-9.2f"
              % (tag.split(chr(10))[0][:51], s["p1"], s["p50"], s["p99"], s["std"], s["detail"], s["blown"]))
    print()
    print("   the DV frame's dark end is %.4f and the model's is %.4f -- a difference of %.4f"
          % (sd["p1"], sm["p1"], abs(sd["p1"] - sm["p1"])))
    print("   its global contrast is %.4f against the model's %.4f" % (sd["std"], sm["std"]))
    print("   **and its detail is %.5f against the model's %.5f** -- %.0f%%, which no level control can raise."
          % (sd["detail"], sm["detail"], 100 * sm["detail"] / sd["detail"]))
    print()

    cols, cw, ch, gap, cap = 4, 330, 248, 8, 70
    rows = (len(panels) + cols - 1) // cols
    canvas = Image.new("RGB", ((cw + gap) * cols + gap, (ch + cap) * rows + 40), (14, 14, 16))
    d = ImageDraw.Draw(canvas)
    d.text((10, 6), "THE CONTRAST CURVE -- the dark end lands on the target, and the white point stays at full scale",
           fill=(240, 240, 242))
    d.text((10, 22), "on the DV frame: p1 0.0517 -> 0.0344 against the frame's 0.0353; std 0.2794 -> 0.2988 against "
                     "0.2896", fill=(220, 200, 150))
    for i, (tag, img) in enumerate(panels):
        x = gap + (i % cols) * (cw + gap)
        y = 42 + (i // cols) * (ch + cap)
        canvas.paste(Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).resize((cw, ch), Image.LANCZOS),
                     (x, y))
        s = stat(img)
        for k, line in enumerate(tag.split("\n")):
            d.text((x, y + ch + 3 + 13 * k), line[:64], fill=(245, 245, 248) if k == 0 else (176, 176, 182))
        d.text((x, y + ch + 32), "p1 %.4f  p50 %.4f  p99 %.4f  std %.4f" % (s["p1"], s["p50"], s["p99"], s["std"]),
               fill=(212, 212, 218))
        d.text((x, y + ch + 45), "detail %.5f   blown %.2f%%" % (s["detail"], s["blown"]), fill=(212, 212, 218))
    path = os.path.join(OUT, "CONTRAST_CURVE.png")
    canvas.save(path)
    print("   written: %s" % os.path.relpath(path, ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
