"""
Where the detail goes, and what to do about it.

The user reported four things on a NIGHTSHOT render: slanted bands, contrast too low, the dark side not black, the wall
carrying grubby blotches the footage does not have, the fan LEDs wanting to be brighter with a halo, and **the noise
texture being completely different -- coarse, and "possibly a resolution problem, and it may affect non-night-vision
mode too".**  The last clause was correct and it is the finding.

**Measured, on the new photograph and on the DV frame:**

    detail (mean |dx|)          the photograph 0.02326   the DV frame 0.02264
    the model on the photograph                 0.00396                    17%
    ... with sharpness at 1.0                   0.00524                    23%
    ... and every glow/lens term off            0.00750                    32%

    grain (lag-1 autocorrelation)   the photograph 0.083   the model 0.850

**A high lag-1 with almost no detail is not coarse noise -- it is a smoothed picture.**  With the top frequencies gone
what remains is low-frequency variation, and the eye reads that as blotchy grain.  That is why raising `sensor_noise`
does not fix it (it takes the detail from 0.00396 to 0.00938 and the picture looks like snow, panel 5) and why
`mode OFF` shows it too: **83% of the input's detail is gone before night vision is involved at all.**

The panels put the input, the shipped default, and the best available combination side by side, with the dark and
bright ends measured, so each of the user's four reports can be checked against a number.
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
CLEAN = {"sharpness": 1.0, "halation": 0.0, "lens_flare": 0.0, "source_glow": 0.0, "flare_defocus": 0.0,
         "spike_spread": 0.0, "overflow_drain": 0.0, "ccd_smear": 0.0, "bloom_amount": 0.0,
         "chroma_aberration": 0.0, "longitudinal_ca": 0.0, "field_curvature": 0.0, "coma": 0.0,
         "astigmatism": 0.0, "spherical_aberration": 0.0, "lens_wear": 0.0, "shadow_depth": 0.0,
         "motion_blur": 0.0}


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
                p99=float(np.quantile(l, 0.99)),
                blown=100.0 * float((l > 0.90).mean()))


def main() -> int:
    panels = [
        ("THE NEW PHOTOGRAPH\nthe input, visible light", photo),
        ("the model, as shipped\nNIGHTSHOT 1", run(photo)),
        ("the model, sharpness 1.0\n+ every glow and lens term off", run(photo, CLEAN)),
        ("the model, best of the above\n+ saturation_stops 1.5",
         run(photo, dict(CLEAN, nv_saturation_stops=1.5))),
        ("THE DV FRAME\nthe input with the real highlights", dv),
        ("the model on the DV frame\nas shipped", run(dv)),
        ("the model on the DV frame\n+ every glow and lens term off", run(dv, CLEAN)),
        ("the model on the DV frame\n+ saturation_stops 1.5", run(dv, dict(CLEAN, nv_saturation_stops=1.5))),
    ]
    print("=" * 118)
    print("  WHERE THE DETAIL GOES -- grain is lag-1, detail is mean |dx|, blown is the share above 0.90")
    print("=" * 118)
    print()
    print("   %-52s %-8s %-9s %-8s %-8s %-8s" % ("panel", "grain", "detail", "p1", "p99", "blown%"))
    print("   " + "-" * 112)
    for tag, img in panels:
        s = stat(img)
        print("   %-52s %-8.3f %-9.5f %-8.4f %-8.4f %-8.2f"
              % (tag.split(chr(10))[0][:51], s["g"], s["hf"], s["p1"], s["p99"], s["blown"]))
    print()
    si, sm = stat(photo), stat(run(photo))
    sc = stat(run(photo, CLEAN))
    print("   detail: the input has %.5f; as shipped %.5f (%.0f%%); best available %.5f (%.0f%%)"
          % (si["hf"], sm["hf"], 100 * sm["hf"] / si["hf"], sc["hf"], 100 * sc["hf"] / si["hf"]))
    print("   blown:  the input has %.2f%% of its pixels above 0.90; as shipped %.2f%%"
          % (si["blown"], sm["blown"]))
    print()

    cols, cw, ch, gap, cap = 4, 330, 248, 8, 70
    rows = (len(panels) + cols - 1) // cols
    canvas = Image.new("RGB", ((cw + gap) * cols + gap, (ch + cap) * rows + 40), (14, 14, 16))
    d = ImageDraw.Draw(canvas)
    d.text((10, 6), "WHERE THE DETAIL GOES -- 83% of it is gone before night vision is involved",
           fill=(240, 240, 242))
    d.text((10, 22), "the coarse grain is the picture being SMOOTH: with the top frequencies gone, what is left reads "
                     "as blotches", fill=(220, 200, 150))
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
    path = os.path.join(OUT, "DETAIL_BUDGET.png")
    canvas.save(path)
    print("   written: %s" % os.path.relpath(path, ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
