"""
Calibrate `chroma_residue` against the footage, by sweep, because it does not move the quantity monotonically.

**The two targets, both recovered from the footage itself:**

    hue spread      0.5519 / 0.1170 / 0.1129   (the three NightShot segments)
    saturation      0.1436 / 0.3095 / 0.3387

**and the model must land inside both ranges at once**, which is what makes this a calibration rather than a preference.
The hue spread was measured to be **non-monotonic** in the residue -- 0.50 gave 0.6966 where 0.70 gave 0.6019 -- so a
formula is not available and a sweep is the honest method.

**The saturation is the constraint that decides it.**  It was already below the footage's at 0.1950 against 0.310-0.339,
and widening the hue spread costs it a little, so the useful setting is the one that reaches the hue range **at the
smallest saturation cost** -- and if no setting reaches both, that is the finding and the mechanism is short a degree of
freedom.

**The luminance is checked too**, because `chroma_residue`'s whole design is that it carries the picture's chroma against
LUMA and so leaves the brightness exactly as it was -- a property an earlier round established, and one this calibration
must not disturb.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from atsim.core.params import merge_values                                    # noqa: E402
from atsim.core.profiles import FACTORY_PRESETS, SCHEMA                       # noqa: E402
from atsim.render.renderer import Renderer                                    # noqa: E402

COEF = np.array([0.299, 0.587, 0.114], np.float32)
OBJ = (r"C:\Users\Administrator\.dsh\attachments\v1\objects")
NS = [("NIGHTSHOT 0", os.path.join(OBJ, "ed",
                                   "ed906503e82e76777332850f677a16db3ea89089fde1c313576edf9fce66f4e0")),
      ("NIGHTSHOT 1", os.path.join(OBJ, "f2",
                                   "f2dd7803fd91d124a7c16be3c6f97df679c8a83d89631a462e9010f13a3e7c5e")),
      ("NIGHTSHOT 2", os.path.join(OBJ, "b9",
                                   "b94a0e8402a11788032bd0af9e81de7fc4713d3a648384f2c9ff57b21c21f4b6"))]
PHOTO = os.path.join(OBJ, "69", "699c23315df037b01fa831a273364b41f99c28733fb1dc928b22d1e6c281818d")
OUT = os.path.join(ROOT, "tools", "calib")
W, H, OSD = 533, 400, 0.22
sl = slice(int(H * OSD), H)


def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0
    return np.asarray(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS),
                      np.float32) / 255.0


def stats(a):
    sc = np.clip(np.asarray(a, np.float32), 0, 1)[sl]
    r, g, b = sc[..., 0], sc[..., 1], sc[..., 2]
    u, v = r - g, b - g
    m = np.sqrt(u * u + v * v)
    k = m > max(1e-4, float(np.quantile(m, 0.5)))
    ang = np.arctan2(v, u)
    sp = float(np.angle(np.exp(1j * (ang[k] - ang[k].mean()))).std()) if k.any() else 0.0
    sa = float(m[k].mean()) if k.any() else 0.0
    return sp, sa, float(np.clip(sc @ COEF, 0, 1).mean())


SRC = load(PHOTO)


class One:
    height, width = H, W
    fps = 25.0
    frame_count = 1

    def frame(self, i):
        return SRC


BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])
OFF = {"halation": 0.0, "ccd_smear": 0.0, "flare_defocus": 0.0, "lens_flare": 0.0,
       "overflow_drain": 0.0, "chroma_aberration": 0.0, "sensor_noise": 0.0, "noise_grain": 0.0,
       "fixed_pattern": 0.0, "spike_spread": 0.0, "motion_blur": 0.0, "stuck_fraction": 0.0,
       "bloom_threshold": 0.0, "overflow_width": 0.0, "sharpness": 1.0}


def run(res, luma_bw=3.0):
    layers = {"camera": dict(OFF, nv_mode=1.0, nv_chroma_residue=res),
              "format": {"chroma_bandwidth": 0.75, "luma_bandwidth": luma_bw},
              "player": {"chroma_bandwidth": 0.75, "luma_bandwidth": luma_bw},
              "codec": {"dv_enabled": 0.0}, "display": {"enabled": 0.0}}
    rr = Renderer(One(), W, H, 25.0, merge_values(BASE, layers), seed=20240101)
    return np.clip(np.asarray(rr.render_frame(0), np.float32), 0, 1)


print("   === the targets ===")
tg = []
for name, p in NS:
    sp, sa, lm = stats(load(p))
    tg.append((sp, sa))
    print("   %-14s hue spread %.4f   saturation %.4f   luma %.4f" % (name, sp, sa, lm))
SP_LO, SP_HI = min(t[0] for t in tg), max(t[0] for t in tg)
SA_LO, SA_HI = min(t[1] for t in tg), max(t[1] for t in tg)
print("   the ranges to land inside: hue %.4f-%.4f, saturation %.4f-%.4f" % (SP_LO, SP_HI, SA_LO, SA_HI))

print()
print("   === the sweep ===")
print("   %-10s %-12s %-12s %-10s %-10s %s" % ("residue", "hue spread", "saturation", "hue in?", "sat in?", "luma"))
best = None
for res in (0.0, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.0):
    a = run(res)
    sp, sa, lm = stats(a)
    hin = SP_LO <= sp <= SP_HI
    sin = SA_LO <= sa <= SA_HI
    if hin and (best is None or abs(sa - SA_LO) < abs(best[2] - SA_LO)):
        best = (res, sp, sa, lm)
    print("   %-10.2f %-12.4f %-12.4f %-10s %-10s %.4f" % (res, sp, sa, "yes" if hin else "-",
                                                          "yes" if sin else "-", lm))
print()
if best:
    print("   the best setting that lands the HUE inside the footage's range: residue %.2f" % best[0])
    print("      hue %.4f (the range is %.4f-%.4f), saturation %.4f (the range is %.4f-%.4f), luma %.4f"
          % (best[1], SP_LO, SP_HI, best[2], SA_LO, SA_HI, best[3]))
    print("      saturation is %+.4f from the bottom of the footage's range -- so the model is short of chroma by that"
          % (best[2] - SA_LO))
    print("      much, which is a SECOND defect rather than something this knob can reach.")
else:
    print("   no setting lands the hue inside the footage's range -- the mechanism cannot reach it at all.")
print()
print("   and the luminance, which the residue is designed to leave exactly alone:")
for res in (0.0, 0.5, 1.0):
    print("      residue %.2f -> luma %.4f" % (res, stats(run(res))[2]))
