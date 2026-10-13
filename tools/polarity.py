"""
Measure the POLARITY, not the contrast: is the model's picture correlated with the real frame or inverted against it?

**Why the distinction matters and why it was missed.**  Every measurement this session compared *scalars* -- mean level,
standard deviation -- and those are blind to a sign.  **A picture and its negative have the same mean and the same
standard deviation**, so a compression and an inversion are indistinguishable to every statistic used so far.  The user
has said "inverted" three times while the measurements said "too flat", and both can be true at once: **the model can be
flat *and* the wrong way round.**

**So the test is a correlation in space.**  The real frame and the model cannot be aligned pixel-for-pixel -- different
cameras, different framing -- but **the subject is the same and dominant**, so a coarse comparison is enough: downsample
both to a small grid, and correlate the model's luminance against the real frame's.  **A correct render gives a positive
correlation; an inverted one gives a negative.**

**And to make the answer independent of framing**, the same test is run against the source photograph, which shares the
model's framing exactly -- so a negative correlation there is unambiguous, since the only difference between them is the
chain.

**Then, if it is inverted, the pieces are tested one at a time**: the night-vision scene stage, the green display, and
the IR term, each applied alone to the source, correlated against the source.
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
NS0 = os.path.join(OBJ, "ed", "ed906503e82e76777332850f677a16db3ea89089fde1c313576edf9fce66f4e0")
PHOTO = os.path.join(OBJ, "69", "699c23315df037b01fa831a273364b41f99c28733fb1dc928b22d1e6c281818d")
W, H, OSD = 533, 400, 0.22
sl = slice(int(H * OSD), H)


def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0
    return np.asarray(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS),
                      np.float32) / 255.0


def lum(a):
    return np.clip(a[sl] @ COEF, 0, 1)


def corr(a, b):
    x = (a - a.mean()).ravel()
    y = (b - b.mean()).ravel()
    d = float(np.sqrt((x * x).sum() * (y * y).sum()))
    return float((x * y).sum() / d) if d > 1e-12 else 0.0


src = load(PHOTO)
real = load(NS0)
ls, lr = lum(src), lum(real)
print("   === the references ===")
print("      source photograph   mean %.4f  contrast %.4f" % (float(ls.mean()), float(ls.std())))
print("      real NIGHTSHOT 0    mean %.4f  contrast %.4f" % (float(lr.mean()), float(lr.std())))
print("      their mutual correlation, at frame scale: %+.3f  (same subject, different framing)" % corr(ls, lr))


class One:
    height, width = H, W
    fps = 25.0
    frame_count = 1

    def __init__(self, a):
        self.a = a

    def frame(self, i):
        return self.a


BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])
OFF = {"halation": 0.0, "ccd_smear": 0.0, "flare_defocus": 0.0, "lens_flare": 0.0,
       "overflow_drain": 0.0, "chroma_aberration": 0.0, "sensor_noise": 0.0, "noise_grain": 0.0,
       "fixed_pattern": 0.0, "spike_spread": 0.0, "motion_blur": 0.0, "stuck_fraction": 0.0,
       "bloom_threshold": 0.0, "overflow_width": 0.0, "sharpness": 1.0}


def run(**cam):
    layers = {"camera": dict(OFF, **cam),
              "format": {"chroma_bandwidth": 3.0, "luma_bandwidth": 6.0},
              "player": {"chroma_bandwidth": 3.0, "luma_bandwidth": 6.0},
              "codec": {"dv_enabled": 0.0}, "display": {"enabled": 0.0}}
    rr = R.Renderer(One(src), W, H, 25.0, merge_values(BASE, layers), seed=1)
    return np.clip(np.asarray(rr.render_frame(0), np.float32), 0, 1)


print()
print("   === the polarity of the model, against the source it came from ===")
print("   %-46s %-10s %-11s %s" % ("setting", "luma", "contrast", "corr with the SOURCE"))
rows = [
    ("the mode off (the chain alone)", dict(nv_mode=0.0)),
    ("the mode on, everything", dict(nv_mode=1.0, nv_green_mix=1.0, nv_illuminator=0.15)),
    ("the mode on, lamp off", dict(nv_mode=1.0, nv_green_mix=1.0, nv_illuminator=0.0)),
    ("the mode on, IR term off", dict(nv_mode=1.0, nv_green_mix=1.0, nv_illuminator=0.0, nv_ir_albedo=0.0)),
    ("the mode on, green display only", dict(nv_mode=1.0, nv_green_mix=1.0, nv_illuminator=0.0,
                                             nv_ir_albedo=0.0, nv_ir_response=0.0)),
]
for tag, cam in rows:
    a = run(**cam)
    l = lum(a)
    c = corr(l, ls)
    print("   %-46s %-10.4f %-11.4f %+.3f%s"
          % (tag, float(l.mean()), float(l.std()), c, "   <- INVERTED" if c < 0 else ""))

print()
print("   === and the night-vision pieces applied alone to the source ===")
from atsim.codecs import nightvision as nv                                    # noqa: E402
spec = nv.NightVisionSpec(mode=1.0, illuminator=0.15, ir_response=1.0, ir_albedo_mix=0.7, green_mix=1.0)
pieces = [("ir_highlights", nv.ir_highlights(src, spec)),
          ("monochrome_green", nv.monochrome_green(src, spec)),
          ("add_illuminator", nv.add_illuminator(src, spec))]
for tag, a in pieces:
    a = np.clip(np.asarray(a, np.float32), 0, 1)
    l = lum(a)
    print("   %-46s %-10.4f %-11.4f %+.3f%s"
          % (tag, float(l.mean()), float(l.std()), corr(l, ls), "   <- INVERTED" if corr(l, ls) < 0 else ""))
mixed = np.clip(np.asarray(nv.ir_highlights(src, spec), np.float32), 0, 1)
mixed = np.clip(np.asarray(nv.monochrome_green(mixed, spec), np.float32), 0, 1)
print("   %-46s %-10.4f %-11.4f %+.3f%s"
      % ("ir_highlights then monochrome_green", float(lum(mixed).mean()), float(lum(mixed).std()),
         corr(lum(mixed), ls), "   <- INVERTED" if corr(lum(mixed), ls) < 0 else ""))
print()
print("   a NEGATIVE correlation means the picture is inverted relative to what it was made from -- which no mean or")
print("   standard deviation can show, and which is what the user has been reporting.")
