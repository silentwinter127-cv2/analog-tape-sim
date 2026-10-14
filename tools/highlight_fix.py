"""
Find what can put the highlights where the footage has them, since nothing currently does.

**Measured, and it is a large departure rather than a detail:**

    the footage, NIGHTSHOT 0    p90 0.5870   p99 0.9511   max 1.0000   clipped 0.00657
    **the model, mode 1**       p90 0.5394   p99 **0.8011**  max **0.8662**  clipped **0.00000**
    departure                   -0.0477      **-0.1499**    **-0.1338**    **-0.00657**

**so the model's brightest pixels never reach the top of the range at all** -- its maximum is 0.866 and *nothing anywhere
in the frame is clipped* -- where the footage reaches 1.0 with 0.66% of the frame blown out.  **`note.txt` lists exactly
this as a mechanism** ("加入 … 以及高亮区域的饱和"), and it is the one of its six that was never implemented.

**And the mean is not the problem** (0.2298 against 0.2346, +2%), so the shortfall is in the top of the curve only --
which is why every whole-frame measurement taken so far agreed while this is badly wrong.  **The tenth statistic to be
blind to the thing being looked at.**

**So the available controls are swept against the footage's own top-of-range figures**, and the ones that could plausibly
carry a highlight shoulder are tested first:

  * **`illuminator`** -- the lamp is what blows out a near surface, and it is the dominant term in the mode;
  * **`dynamic_range`** -- the camera's own range, which `_roll_off` turns into a shoulder;
  * **`nv_exposure_steps`** -- the mode's exposure bias, which moves the whole curve up;
  * **`bloom_threshold` / `overflow_drain`** -- the terms that exist for bright regions.

**The prediction, stated so it can fail**: if any of them can reach `max ~= 1.0` with `clipped ~= 0.007` while leaving the
mean near 0.235, then the model has the control and it was simply never set.  **If none can, the mechanism is missing
rather than mistuned** -- which is a different and more serious finding, and it is what the six `note.txt` mechanisms
imply, since highlight saturation is one of them and no term implements it.
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
from atsim.render.renderer import Renderer                                    # noqa: E402

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


def hi(a):
    g = np.clip(np.asarray(a, np.float32)[sl] @ COEF, 0, 1)
    return (float(g.mean()), float(np.quantile(g, 0.90)), float(np.quantile(g, 0.99)),
            float(g.max()), float((g >= 0.98).mean()))


REAL = load(NS0)
rm, r90, r99, rmx, rcl = hi(REAL)
print("   the footage:  mean %.4f  p90 %.4f  p99 %.4f  max %.4f  clipped %.5f" % (rm, r90, r99, rmx, rcl))
print()
print("   %-40s %-9s %-9s %-9s %-9s %-9s %s" % ("", "mean", "p90", "p99", "max", "clipped", "vs the footage"))
print("   %-40s %-9s %-9s %-9s %-9s %-9s %s"
      % ("the target", "%.4f" % rm, "%.4f" % r90, "%.4f" % r99, "%.4f" % rmx, "%.5f" % rcl, "-"))


class One:
    height, width = H, W
    fps = 25.0
    frame_count = 1

    def frame(self, i):
        return load(PHOTO)


BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])
OFF = {"halation": 0.0, "ccd_smear": 0.0, "flare_defocus": 0.0, "lens_flare": 0.0,
       "overflow_drain": 0.0, "chroma_aberration": 0.0, "sensor_noise": 0.0, "noise_grain": 0.0,
       "fixed_pattern": 0.0, "spike_spread": 0.0, "motion_blur": 0.0, "stuck_fraction": 0.0,
       "bloom_threshold": 0.0, "overflow_width": 0.0, "sharpness": 1.0}


def run(**cam):
    cam = dict(cam)
    cam.setdefault("nv_mode", 1.0)
    layers = {"camera": dict(OFF, **cam),
              "format": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0},
              "player": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0},
              "codec": {"dv_enabled": 0.0}, "display": {"enabled": 0.0}}
    rr = Renderer(One(), W, H, 25.0, merge_values(BASE, layers), seed=20240101)
    return np.clip(np.asarray(rr.render_frame(0), np.float32), 0, 1)


def line(tag, **cam):
    m, p90, p99, mx, cl = hi(run(**cam))
    print("   %-40s %-9.4f %-9.4f %-9.4f %-9.4f %-9.5f dmax %+.3f" % (tag, m, p90, p99, mx, cl, mx - rmx))
    return mx, cl


print("   %-40s %-9s %-9s %-9s %-9s %-9s %s" % ("as it ships", "", "", "", "", "", ""))
line("as it ships (illuminator 0.15)")
print()
print("   the illuminator, which is what blows out a near surface:")
for v in (0.15, 0.3, 0.5, 0.8, 1.2, 2.0, 3.0):
    line("illuminator %.2f" % v, nv_illuminator=v)
print()
print("   the camera's dynamic range, which the shoulder is built from:")
for v in (8.0, 10.0, 12.0, 14.0, 16.0):
    line("dynamic_range %.1f" % v, dynamic_range=v)
print()
print("   the mode's exposure bias and the bright-region terms:")
line("exposure bias 1.0", nv_exposure_steps=1.0)
line("exposure bias 2.0", nv_exposure_steps=2.0)
for bt in (0.5, 0.7, 0.85):
    line("bloom_threshold %.2f, bloom 0.5" % bt, bloom_threshold=bt, bloom_amount=0.5)
line("halation 0.5", halation=0.5)
print()
print("   and the input itself, for scale:")
m, p90, p99, mx, cl = hi(load(PHOTO))
print("   %-40s %-9.4f %-9.4f %-9.4f %-9.4f %-9.5f" % ("the source photograph", m, p90, p99, mx, cl))
print()
print("   **the footage reaches max 1.000 with 0.66%% clipped.  A setting that only raises the mean without bringing the")
print("   maximum to the top of the range is not implementing saturation, it is just brighter.**")
