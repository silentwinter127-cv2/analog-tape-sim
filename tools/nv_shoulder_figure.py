"""
The highlight shoulder: what it fixed, and the two faults found building it.

"光感太差了 不够通透" measured out as one specific thing, and it was not where the previous round was looking:

    statistic     the model   the real frame
    dark <0.15     20.6%       24.9%      ** already matched **
    p95            0.9387      0.8470     the bright end runs to the ceiling
    >0.95           4.76%       1.41%     several times too much of the frame is pinned there

**The dark end had arrived and the top had not**, so the black floor and the mid-tone calibration were left alone and the
missing operation was built: a **shoulder**, where the excess above a soft knee is compressed instead of added.  `_dsp_level`
drives a tenth of the frame's peaks above 0.95, and this is the step that takes it back.

**Two faults were found in it, and both are the kind this work keeps recording.**

1. **The first mapping did not send 1.0 to 1.0** -- a saturated source came out at **0.890**, so no picture could contain
   pure white.  That is the defect `_roll_off` was fixed for, reintroduced through a new mechanism, and the suite's
   `a saturated highlight reaches pure white` caught it.

2. **The replacement used `np.maximum` where a condition was needed**, so a pixel *below* the knee was raised onto it:
   measured on a grey ramp, inputs of 0.200 and of 0.500 both came out at exactly **0.700**, the dark share went to
   **0.0%** and eight checks failed downstream because every optical defect's amplitude had moved.

**And `nv_highlight_knee` was read from the stage while the schema did not declare it**, so three different values gave
bit-identical frames -- *a parameter that exists in one layer and not the other*, the same shape as the preset-override
fault found six times.

**One thing is still open and is stated rather than implied.** The shoulder moves `p95` by 0.003 on the real frame
(0.9387 -> 0.9359) while doing exactly what it should on a grey ramp, and the trace says why the two differ:

    after _expose      luma p95 0.5330   peak p95 0.5526
    after _roll_off    luma p95 0.5955   peak p95 0.6442
    after _dsp_level   luma p95 0.9423   peak p95 1.0000    10.18% of pixels have a peak above 0.95

**`_dsp_level` drives a tenth of the frame to the ceiling**, and the statistic being matched is the *luminance's* p95
while the compressor keys off the *peak channel* -- in a green-dominant picture those are far apart.  The exponent was
changed from `s**2` to `s**1.6` for this reason (the derivative of `s**2` is 0 at the top, the one place the compression
is needed) and it did not close the gap, which locates the remaining work in the **quantity being compressed**, not in the
curve: the shoulder has to act on the same measure that is being matched.
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


def run(src, **cam):
    lay = {"camera": dict(nv_mode=2.0, **cam),
           "format": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0},
           "player": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0},
           "codec": {"dv_enabled": 1.0}}
    r = Renderer(Still(src), W, H, 25.0, merge_values(BASE, lay), seed=20240101)
    return np.clip(np.asarray(r.render_frame(0), np.float32), 0, 1)


def stat(x):
    a = np.asarray(x, np.float32)
    l = np.clip(a @ LUMA, 0, 1)
    r, g, b = (float(a[..., i].mean()) for i in range(3))
    return dict(mean=float(l.mean()), p25=float(np.quantile(l, 0.25)), p50=float(np.quantile(l, 0.50)),
                p75=float(np.quantile(l, 0.75)), p95=float(np.quantile(l, 0.95)), std=float(l.std()),
                dark=100 * float((l < 0.15).mean()), blown=100 * float((l > 0.90).mean()),
                pin=100 * float((l > 0.95).mean()), rg=r / max(1e-9, g), bg=b / max(1e-9, g))


def main() -> int:
    real = load(os.path.join(OUT, "user_real.png"), osd=True)
    real_full = load(os.path.join(OUT, "user_real.png"))
    dv = load(os.path.join(OUT, "sc1515", "d_nv3.png"), 0.20)

    panels = [
        ("THE REAL NIGHTSHOT 2\nthe frame to match", real_full),
        ("the model, shoulder off\nthe top runs to the ceiling", run(dv, nv_highlight_compress=0.0)),
        ("the model, ** shoulder on **\ncompressed above the knee", run(dv)),
        ("the model, a lower knee\n0.55 -- the shoulder starts earlier", run(dv, nv_highlight_knee=0.55)),
        ("the model, the shape at 2.5\ncompression concentrated at the top", run(dv, nv_highlight_shape=2.5)),
        ("the model on the user's own frame\nsame settings", run(real)),
    ]
    sr = stat(real)
    print("=" * 122)
    print("  THE HIGHLIGHT SHOULDER -- what 'not transparent enough' measured out as")
    print("=" * 122)
    print()
    print("   %-48s %-8s %-8s %-8s %-8s %-8s %-8s" %
          ("panel", "mean", "p50", "p75", "p95", "dark%", "pin%"))
    print("   " + "-" * 118)
    for tag, img in panels:
        s = stat(img)
        print("   %-48s %-8.4f %-8.4f %-8.4f %-8.4f %-8.1f %-8.2f"
              % (tag.split(chr(10))[0][:47], s["mean"], s["p50"], s["p75"], s["p95"], s["dark"], s["pin"]))
    print()
    sm = stat(run(dv))
    print("   against the real frame, every statistic within 15%:")
    for name in ("mean", "p25", "p50", "p75", "p95", "std", "rg", "bg"):
        tv, mv = sr[name], sm[name]
        err = abs(mv - tv) / max(1e-9, abs(tv))
        print("     %-6s %8.4f vs %8.4f   %5.1f%%   %s" % (name, tv, mv, 100 * err, "ok" if err < 0.15 else "off"))
    print("     %-6s %7.2f%% vs %7.2f%%" % ("dark", sr["dark"], sm["dark"]))
    print("     %-6s %7.2f%% vs %7.2f%%   ** still open: the shoulder moves p95 by 0.003 on a frame, "
          "not by 0.09 **" % ("pinned", sr["pin"], sm["pin"]))
    print()

    cols, cw, ch, gap, cap = 3, 430, 322, 10, 74
    rows = (len(panels) + cols - 1) // cols
    canvas = Image.new("RGB", ((cw + gap) * cols + gap, (ch + cap) * rows + 40), (14, 14, 16))
    d = ImageDraw.Draw(canvas)
    d.text((10, 6), "THE HIGHLIGHT SHOULDER -- the dark end had arrived; the top was pinned, and this is the step that "
                    "takes it back", fill=(240, 240, 242))
    d.text((10, 22), "dark share 20.6% vs the frame's 24.9% (matched); p95 0.9387 vs 0.8470 (open)",
           fill=(220, 200, 150))
    for i, (tag, img) in enumerate(panels):
        x = gap + (i % cols) * (cw + gap)
        y = 42 + (i // cols) * (ch + cap)
        canvas.paste(Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).resize((cw, ch), Image.LANCZOS),
                     (x, y))
        s = stat(img)
        for k, line in enumerate(tag.split("\n")):
            d.text((x, y + ch + 3 + 13 * k), line[:76], fill=(245, 245, 248) if k == 0 else (176, 176, 182))
        d.text((x, y + ch + 32), "mean %.4f  p50 %.4f  p95 %.4f" % (s["mean"], s["p50"], s["p95"]),
               fill=(212, 212, 218))
        d.text((x, y + ch + 45), "dark %.1f%%   >0.95 %.2f%%" % (s["dark"], s["pin"]), fill=(212, 212, 218))
    path = os.path.join(OUT, "SHOULDER.png")
    canvas.save(path)
    print("   written: %s" % os.path.relpath(path, ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
