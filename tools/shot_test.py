"""
The shot noise term's own contribution, isolated: does it move the tonal distribution the way the physics says?

**`note.txt` asks for two noise terms with opposite behaviour:**

    "加入随曝光和信号强度变化的噪声、暗部读出噪声、固定图样噪声，以及高亮区域的饱和。
     不要简单地对整张图叠加均匀颗粒；暗部和高光的表现应该不同。"

**Read noise is a fixed absolute amount** -- relatively largest in the shadows, so raising it *raises* the dark-to-bright
ratio.  **Shot noise grows as the square root of the signal** -- so it is largest in absolute terms in the highlights, and
raising it should *lower* that ratio.  **That opposition is the whole point of having both**, and it is a prediction that
can fail: if the shot term moves the ratio the same way the read term does, it is not modelling what it claims.

**Measured with read noise held at zero**, so the only thing changing is the term under test, and with the noise isolated
temporally -- several seeds on a static subject, so anything that varies between renders is noise and nothing else.

**And the target is the footage's own statistic**, with its limitation stated: the 3x3 residual in its dark and bright
thirds reads **0.00546 against 0.01656, ratio 0.330**, but that residual includes the picture's real detail, which is
richer in bright regions -- so it is a **lower** bound on what structure alone contributes and cannot be matched
numerically by a noise model.  **The test is therefore the DIRECTION and the size of the effect, not equality.**
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
NS0 = os.path.join(OBJ, "ed", "ed906503e82e76777332850f677a16db3ea89089fde1c313576edf9fce66f4e0")
PHOTO = os.path.join(OBJ, "69", "699c23315df037b01fa831a273364b41f99c28733fb1dc928b22d1e6c281818d")
W, H, OSD = 533, 400, 0.22
sl = slice(int(H * OSD), H)


def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0
    return np.asarray(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS),
                      np.float32) / 255.0


real = load(NS0)
g = np.clip(real[sl] @ COEF, 0, 1)
pad = np.pad(g, 1, mode="edge")
loc = sum(pad[i:i + g.shape[0], j:j + g.shape[1]] / 9.0 for i in range(3) for j in range(3))
res = np.abs(g - loc)
lo, hi = np.quantile(g, 1 / 3), np.quantile(g, 2 / 3)
RD, RB = float(res[g <= lo].mean()), float(res[g >= hi].mean())
print("   the footage's own statistic (an upper bound, because bright areas carry more real detail):")
print("      dark %.5f  bright %.5f  ratio %.3f" % (RD, RB, RD / RB))

src = load(PHOTO)


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


def temporal(**cam):
    outs = []
    for s in (11, 22, 33, 44, 55):
        layers = {"camera": dict(OFF, nv_mode=1.0, **cam),
                  "format": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0},
                  "player": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0},
                  "codec": {"dv_enabled": 0.0}, "display": {"enabled": 0.0}}
        rr = Renderer(One(), W, H, 25.0, merge_values(BASE, layers), seed=s)
        outs.append(np.clip(np.asarray(rr.render_frame(0), np.float32), 0, 1))
    st = np.stack(outs)
    gg = np.clip(st @ COEF, 0, 1)
    noise = gg.std(axis=0)[sl]
    mg = gg.mean(axis=0)[sl]
    a, b = np.quantile(mg, 1 / 3), np.quantile(mg, 2 / 3)
    return float(noise[mg <= a].mean()), float(noise[mg >= b].mean())


print()
print("   === the two terms, isolated ===")
print("   %-38s %-12s %-12s %-10s %s" % ("", "dark third", "bright third", "ratio", "effect on the ratio"))
rows = [("read 0.00  shot 0.00 (neither)", dict(nv_read_noise=0.0, nv_shot_noise=0.0)),
        ("read 0.10  shot 0.00", dict(nv_read_noise=0.10, nv_shot_noise=0.0)),
        ("read 0.25  shot 0.00", dict(nv_read_noise=0.25, nv_shot_noise=0.0)),
        ("read 0.00  shot 0.10", dict(nv_read_noise=0.0, nv_shot_noise=0.10)),
        ("read 0.00  shot 0.25", dict(nv_read_noise=0.0, nv_shot_noise=0.25)),
        ("read 0.10  shot 0.10 (both)", dict(nv_read_noise=0.10, nv_shot_noise=0.10))]
base = None
for tag, cam in rows:
    d, b = temporal(**cam)
    r = d / max(1e-9, b)
    if base is None:
        base = r
    print("   %-38s %-12.5f %-12.5f %-10.3f %+.3f" % (tag, d, b, r, r - base))
print()
print("   **the prediction, stated so it can fail:** raising `read_noise` alone RAISES the ratio (a fixed amount is")
print("   relatively larger in the shadows) and raising `shot_noise` alone LOWERS it (it grows with the signal).")
print("   The two must move it in opposite directions; if they move it the same way, one of them is not modelling what")
print("   it claims.")
