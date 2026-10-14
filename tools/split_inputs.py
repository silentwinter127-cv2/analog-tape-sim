"""
Measure where the second split's 1.65x chroma comes from, since the per-row-shift mechanism was refuted by the code.

**What was refuted, and how.**  The previous round concluded that the per-row displacement (135 of 399 rows round to a
different shift) breaks the subcarrier's line-to-line inversion, so the comb cannot cancel and reads chroma at 1.65x the
composite.  **Reading the code shows that cannot be the mechanism**: the shift is applied to the **baseband** `i` and `q`,
and the carrier is generated *afterwards* by `remod_chroma(i, q, std)`, which takes its phase from the standard's own
per-line constant.  **Shifting baseband chroma and then modulating does not shift carrier phase** -- it moves the content
that the carrier describes, which is what a timebase error should do.

**So the 1.65x is still unexplained, and the honest position is that it was inferred from a number plus a reading rather
than measured.**  This measures it directly instead: the two splits are captured with **everything they are given**, so
the second split's own input is examined rather than assumed.

**The quantities that decide it:**

  * the **composite handed to the second split** -- if its chroma is genuinely large, the first split is at fault; if it
    is small and the split still returns 1.65x, the split itself is at fault;
  * the **`comb_depth` and `chroma_bandwidth_hz`** it is called with, since a narrow window at a high comb depth is where
    a band-pass can return more than it was given;
  * and the **band-pass's own response** to that input, which is measurable by applying it directly.
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
import atsim.core.signal as SG                                                # noqa: E402
from atsim.stages.player import PlayerStage                                   # noqa: E402

COEF = np.array([0.299, 0.587, 0.114], np.float32)
OBJ = (r"C:\Users\Administrator\.dsh\attachments\v1\objects")
PHOTO = os.path.join(OBJ, "69", "699c23315df037b01fa831a273364b41f99c28733fb1dc928b22d1e6c281818d")
W, H = 533, 400
src = np.asarray(Image.open(PHOTO).convert("RGB"), np.float32) / 255.0
src = np.asarray(Image.fromarray((np.clip(src, 0, 1) * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS),
                 np.float32) / 255.0

#: capture what the SECOND split receives, and what the FIRST one was told
seen = {}
o_sc = SG.split_composite
count = {"n": 0}


def split(comp, std, comb_depth=0.0, chroma_bandwidth_hz=0.8e6):
    count["n"] += 1
    n = count["n"]
    c = np.asarray(comp, np.float32)
    lum, ch = o_sc(comp, std, comb_depth=comb_depth, chroma_bandwidth_hz=chroma_bandwidth_hz)
    l = np.asarray(lum, np.float32)
    h = np.asarray(ch, np.float32)
    seen[n] = dict(comp=c, lum=l, chroma=h, cd=float(comb_depth), bw=float(chroma_bandwidth_hz))
    return lum, ch


SG.split_composite = split


class One:
    height, width = H, W
    fps = 25.0
    frame_count = 1

    def frame(self, i):
        return src


BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])
layers = {"camera": {"nv_mode": 0.0, "halation": 0.0, "ccd_smear": 0.0, "flare_defocus": 0.0,
                     "lens_flare": 0.0, "overflow_drain": 0.0, "chroma_aberration": 0.0,
                     "sensor_noise": 0.0, "noise_grain": 0.0, "fixed_pattern": 0.0,
                     "spike_spread": 0.0, "motion_blur": 0.0, "stuck_fraction": 0.0,
                     "bloom_threshold": 0.0, "overflow_width": 0.0, "sharpness": 1.0},
          "format": {"chroma_bandwidth": 3.0, "luma_bandwidth": 6.0},
          "player": {"chroma_bandwidth": 3.0, "luma_bandwidth": 6.0},
          "codec": {"dv_enabled": 0.0}, "display": {"enabled": 0.0}}
rr = R.Renderer(One(), W, H, 25.0, merge_values(BASE, layers), seed=1)
rr.render_frame(0)
SG.split_composite = o_sc

print("   === each split, with what it was given and what it returned ===")
for n in sorted(seen):
    d = seen[n]
    c, l, h = d["comp"], d["lum"], d["chroma"]
    cr = float(c.std())
    print()
    print("   split #%d   comb_depth %.2f   chroma_bandwidth %.3f MHz" % (n, d["cd"], d["bw"] / 1e6))
    print("      the composite it was given: rms %.5f  spread %.5f  mean %.5f  min %.4f  max %.4f"
          % (float(np.sqrt(np.mean(c ** 2))), cr, float(c.mean()), float(c.min()), float(c.max())))
    print("      returned luma:              rms %.5f  spread %.5f" % (float(np.sqrt(np.mean(l ** 2))), float(l.std())))
    print("      returned chroma:            rms %.5f  spread %.5f   chroma/comp rms %.3f"
          % (float(np.sqrt(np.mean(h ** 2))), float(h.std()),
             float(np.sqrt(np.mean(h ** 2))) / max(1e-9, float(np.sqrt(np.mean(c ** 2))))))
    # the band-pass on its own, with no comb and no division
    fsc = rr.std.subcarrier_hz
    half = max(d["bw"], 0.30e6)
    bp = SG.fft_bandpass(c, max(0.10e6, fsc - 1.6 * half), fsc + 1.6 * half, rr.std, unity_at_hz=fsc)
    bp = np.asarray(bp, np.float32)
    print("      the band-pass alone on that composite: rms %.5f  spread %.5f  (%.3f of the composite)"
          % (float(np.sqrt(np.mean(bp ** 2))), float(bp.std()), float(bp.std()) / max(1e-9, cr)))
    if d["cd"] > 0.001:
        print("      so the comb + division multiplies the band-pass by %.3f"
              % (float(np.sqrt(np.mean(h ** 2))) / max(1e-9, float(np.sqrt(np.mean(bp ** 2))))))
