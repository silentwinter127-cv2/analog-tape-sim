"""
Locate the inversion stage by stage, correlating each stage's output with the picture it was given.

**What the polarity test just established, and it reframes the whole session.**  Applied to the source photograph
directly, every night-vision piece is correctly polarised -- `ir_highlights` +0.993, `monochrome_green` +1.000,
`add_illuminator` +1.000 -- while **the renderer's own output correlates -0.358 with the picture it was given**.  So the
chain inverts the image, and **no mean or standard deviation can see it**: a picture and its negative share both.

**That is why three sessions of scalar calibration could not fix it**, and why the user has said "inverted" while every
figure said "slightly flat".  **Both were true: the picture was flat *and* upside down.**

**So each stage's output is correlated against its own input**, which is the only measurement that can see a sign, and
the stage where the correlation goes negative is the one that does it.  `r = -1` is a perfect inversion, `r = +1` is
untouched, and a filter that merely blurs or bands a picture still correlates strongly positively with it.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from atsim.core.params import merge_values                                    # noqa: E402
from atsim.core.profiles import FACTORY_PRESETS, SCHEMA                       # noqa: E402
from atsim.render import renderer as R                                        # noqa: E402

COEF = np.array([0.299, 0.587, 0.114], np.float32)
OBJ = (r"C:\Users\Administrator\.dsh\attachments\v1\objects")
PHOTO = os.path.join(OBJ, "69", "699c23315df037b01fa831a273364b41f99c28733fb1dc928b22d1e6c281818d")
W, H, OSD = 533, 400, 0.22
sl = slice(int(H * OSD), H)

src = np.asarray(Image.open(PHOTO).convert("RGB"), np.float32) / 255.0
src = np.asarray(Image.fromarray((np.clip(src, 0, 1) * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS),
                 np.float32) / 255.0


def lum(a):
    return np.clip(np.asarray(a, np.float32)[sl] @ COEF, 0, 1)


def corr(a, b):
    x = (a - a.mean()).ravel()
    y = (b - b.mean()).ravel()
    d = float(np.sqrt((x * x).sum() * (y * y).sum()))
    return float((x * y).sum() / d) if d > 1e-12 else 0.0


class One:
    height, width = H, W
    fps = 25.0
    frame_count = 1

    def frame(self, i):
        return src


BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])
OFF = {"halation": 0.0, "ccd_smear": 0.0, "flare_defocus": 0.0, "lens_flare": 0.0,
       "overflow_drain": 0.0, "chroma_aberration": 0.0, "sensor_noise": 0.0, "noise_grain": 0.0,
       "fixed_pattern": 0.0, "spike_spread": 0.0, "motion_blur": 0.0, "stuck_fraction": 0.0,
       "bloom_threshold": 0.0, "overflow_width": 0.0, "sharpness": 1.0}

layers = {"camera": dict(OFF, nv_mode=0.0),
          "format": {"chroma_bandwidth": 3.0, "luma_bandwidth": 6.0},
          "player": {"chroma_bandwidth": 3.0, "luma_bandwidth": 6.0},
          "codec": {"dv_enabled": 0.0}, "display": {"enabled": 0.0}}

rr = R.Renderer(One(), W, H, 25.0, merge_values(BASE, layers), seed=1)
if rr.chain is None:
    rr._build_chain()

prev = lum(src)
print("   the source's own luma: mean %.4f  std %.4f" % (float(prev.mean()), float(prev.std())))
print()
print("   === each stage's output against its own input ===")
print("   %-12s %-10s %-11s %-12s %s" % ("stage", "mean", "std", "corr with input", "verdict"))

log = []


def measure(name, rgb):
    a = np.clip(np.asarray(rgb, np.float32), 0, 1)
    log.append((name, float(lum(a).mean()), float(lum(a).std()), lum(a)))


for st in rr.chain.stages:
    fn = st.apply

    def make(stage, f):
        def wrapped(buf):
            out = f(buf)
            measure(stage.name, getattr(out, "rgb", out))
            return out
        return wrapped
    st.apply = make(st, fn)

rr.render_frame(0)

for name, m, s, l in log:
    c = corr(l, prev)
    print("   %-12s %-10.4f %-11.4f %+-12.3f %s" % (name, m, s, c,
                                                    "<<<< INVERTED HERE" if c < 0 else ""))
    prev = l

print()
print("   a filter that blurs or bands a picture keeps a strong POSITIVE correlation with it, so a value near -1 means")
print("   the stage negates it and a value near 0 means it destroys the correspondence without inverting.")
print()
print("   and to say which is happening, the extremes are checked:")
for name, l in log:
    print("      %-12s luma min %.4f max %.4f" % (name, float(l.min()), float(l.max())))
