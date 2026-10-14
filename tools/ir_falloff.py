"""
Does the real footage show the illuminator's distance falloff, and what does the model produce in the same terms?

**`note.txt` names this as the first of six mechanisms and the one a filter cannot fake:**

    "模拟机身红外灯的照射范围、光束中心与边缘的亮度差异，以及物体距离增加后反射光减弱的现象。
     重点：近处物体可能很亮，远处背景却仍然很暗。"

**and the model has no distance term at all.**  `illuminator_field` builds a radial falloff from `beam_x`, `beam_y` and
`beam_angle`, normalises it **by its own mean**, and scales it by a constant -- so what it adds is a *shape*, never a
decline with distance.  **The note's point is that near and far differ**, which needs a depth term the model does not have.

**What can be measured instead of depth.**  The model has no scene geometry, so distance cannot be read; but **a picture
already encodes a consequence of it** -- the footage's brightness distribution across the frame.  If the illuminator
dominates, the frame should be **bright near the lamp and dark away from it**, and that is a radial profile, which is
measurable on both the footage and the model's render with no depth at all.

**So the profile is measured on both**, as brightness against distance from the brightest point, normalised so the shapes
can be compared.  **And the prediction is stated so it can fail**: if the footage's profile falls much faster than the
model's, the missing distance term is visible in the data; if the two match, the illuminator is not what is missing and
the note's other mechanisms are.

**The second thing measured is the noise's dependence on signal**, since the note is explicit that uniform grain is
wrong ("不要简单地对整张图叠加均匀颗粒；暗部和高光的表现应该不同"): the residual after removing a local mean is compared
between the frame's dark and bright regions on both.
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
from atsim.codecs.nightvision import illuminator_field, NightVisionSpec        # noqa: E402

COEF = np.array([0.299, 0.587, 0.114], np.float32)
OBJ = (r"C:\Users\Administrator\.dsh\attachments\v1\objects")
NS0 = os.path.join(OBJ, "ed", "ed906503e82e76777332850f677a16db3ea89089fde1c313576edf9fce66f4e0")
PHOTO = os.path.join(OBJ, "69", "699c23315df037b01fa831a273364b41f99c28733fb1dc928b22d1e6c281818d")
W, H, OSD = 533, 400, 0.22
sl = slice(int(H * OSD), H)


def load(p, size=(W, H)):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0
    if size is not None:
        a = np.asarray(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).resize(size, Image.LANCZOS),
                       np.float32) / 255.0
    return a


def radial_profile(a):
    """Brightness against distance from the brightest point, in bins of 1/10 of the frame's diagonal."""
    g = np.clip(np.asarray(a, np.float32)[sl] @ COEF, 0, 1)
    h, w = g.shape
    iy, ix = np.unravel_index(np.argmax(g), g.shape)
    y, x = np.mgrid[0:h, 0:w]
    d = np.sqrt((y - iy) ** 2 + (x - ix) ** 2) / np.sqrt(h * h + w * w)
    edges = np.linspace(0.0, 1.0, 11)
    out = []
    for k in range(10):
        m = (d >= edges[k]) & (d < edges[k + 1])
        out.append(float(g[m].mean()) if m.any() else float("nan"))
    return np.array(out, np.float32)


def noise_vs_signal(a):
    """The residual after removing a 3x3 mean, in the darkest and brightest thirds of the picture."""
    g = np.clip(np.asarray(a, np.float32) @ COEF, 0, 1)
    k = np.ones((3, 3), np.float32) / 9.0
    pad = np.pad(g, 1, mode="edge")
    loc = sum(pad[i:i + g.shape[0], j:j + g.shape[1]] * k[i, j] for i in range(3) for j in range(3))
    res = np.abs(g - loc)
    lo = np.quantile(g, 1 / 3)
    hi = np.quantile(g, 2 / 3)
    dark = float(res[g <= lo].mean())
    bright = float(res[g >= hi].mean())
    return dark, bright


real = load(NS0)
photo = load(PHOTO)
print("   === the illuminator's radial profile (brightness against distance from the brightest point) ===")
print("   %-26s %s" % ("", "  ".join("%4.1f" % (0.05 + 0.1 * k) for k in range(10))))
pr = radial_profile(real)
pp = radial_profile(photo)
print("   %-26s %s" % ("the real NIGHTSHOT 0", "  ".join("%4.2f" % v for v in pr)))
print("   %-26s %s" % ("the photograph (input)", "  ".join("%4.2f" % v for v in pp)))


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
layers = {"camera": dict(OFF, nv_mode=2.0),
          "format": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0},
          "player": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0},
          "codec": {"dv_enabled": 0.0}, "display": {"enabled": 0.0}}
rr = Renderer(One(photo), W, H, 25.0, merge_values(BASE, layers), seed=20240101)
model = np.clip(np.asarray(rr.render_frame(0), np.float32), 0, 1)
pm = radial_profile(model)
print("   %-26s %s" % ("the model (mode 2)", "  ".join("%4.2f" % v for v in pm)))
print()
print("   falloff, brightest bin to the outermost, normalised to the brightest:")
print("      the real frame   %.2f  (%.0f%% of its own peak)" % (pr[-1] / max(1e-9, pr[0]), 100 * pr[-1] / max(1e-9, pr[0])))
print("      the photograph   %.2f" % (pp[-1] / max(1e-9, pp[0])))
print("      the model        %.2f" % (pm[-1] / max(1e-9, pm[0])))
print()
print("   and the noise against signal the note asks for (dark third vs bright third):")
for tag, a in (("the real frame", real), ("the model", model)):
    dk, br = noise_vs_signal(a)
    print("      %-16s dark %.5f   bright %.5f   ratio %.2f" % (tag, dk, br, dk / max(1e-9, br)))
print()
print("   a real sensor's noise FALLS with signal (photon shot noise), so the dark:bright ratio should exceed 1.")
print("   THE NOTE'S POINT: uniform grain gives a ratio of about 1, which is what the model would show if the noise")
print("   does not depend on the signal.")

print()
print("   === and the illuminator field the model actually builds ===")
spec = NightVisionSpec(mode=1.0, illuminator=0.15, ir_response=1.0, ir_albedo_mix=0.7, green_mix=1.0,
                       beam_x=0.5, beam_y=0.5, beam_angle=0.55)
f = illuminator_field(H, W, spec)
print("      shape %s  min %.5f  max %.5f  mean %.5f" % (f.shape, f.min(), f.max(), f.mean()))
print("      **and it is normalised by its own mean**, so what it contributes is a SHAPE: there is no term in it that")
print("      depends on how far away a surface is, which is the note's first mechanism.")
