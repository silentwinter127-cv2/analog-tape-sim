"""
Measure the chroma band against the composite it was split from, on the signal the real stage receives.

**The polarity chain has narrowed to one line.**  Measured with each stage against its own input:

    the composite entering `player`   correlates +0.923 with the source
    `_chroma_phase_error`'s chroma    correlates +1.000 with what it was given
    the timebase displacement         is at most 2.4 px on a 533-wide frame
    the reconstruction leaving it     correlates **-0.358** with the source

**so the inversion happens at `y_band = comp - chroma` inside `_separate`** -- and the only way that line produces a
negative correlation is if `chroma` carries **comparable or greater energy than `comp`**, so that subtracting it
dominates the result.

**And the two measurements that disagree say where to look.**  On a *clean* composite -- a bare luma with no modulation --
`split_composite` is correct: a column-constant picture comes back with a chroma spread of **0.00009** against its
input's **0.28958**, and a luma spread of **1.000**.  **But the signal the stage actually receives is a *modulated*
composite**, and a band-pass's response to a modulated signal is not the same as its response to a bare one.  So the
quantity to measure is the ratio `rms(chroma) / rms(comp)` **on the real signal**.

**The expectation is stated in advance so it can be falsified**: for a picture whose chroma is a fraction of its luma that
ratio should be well below 1.  **If it comes out near or above 1, the band-pass is taking more than the chroma band
holds, and the fix is to bound the chroma by the composite rather than to keep editing the comb's division** -- which
four corrections have already failed to fix.
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

#: capture every split the real call makes, with the luma it returned alongside
splits = []
o_sc = SG.split_composite


def split(comp, std, comb_depth=0.0, chroma_bandwidth_hz=0.8e6):
    lum, ch = o_sc(comp, std, comb_depth=comb_depth, chroma_bandwidth_hz=chroma_bandwidth_hz)
    c = np.asarray(comp, np.float32)
    l = np.asarray(lum, np.float32)
    h = np.asarray(ch, np.float32)
    splits.append((float(c.std()), float(l.std()), float(h.std()), float(comb_depth),
                   float(chroma_bandwidth_hz)))
    return lum, ch


SG.split_composite = split
import atsim.core.signal as SC2                                                # noqa: E402
SC2.split_composite = split
# `yc_separate` calls `split_composite` from its own module namespace, so patch there too
import atsim.core.signal as _s                                                 # noqa: E402
_s.split_composite = split


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
SC2.split_composite = o_sc

print("   === every band split the player made, on the real signal ===")
print("   %-6s %-12s %-12s %-12s %-10s %s"
      % ("#", "comp std", "luma std", "chroma std", "comb", "chroma/comp"))
for i, (cs, ls, hs, cd, bw) in enumerate(splits, 1):
    ratio = hs / max(1e-9, cs)
    flag = ""
    if ratio > 0.9:
        flag = "   <-- the chroma carries as much as the composite: subtracting it must invert the luma"
    elif ratio > 0.5:
        flag = "   <-- the chroma is a large fraction of the composite"
    print("   %-6d %-12.5f %-12.5f %-12.5f %-10.2f %.3f%s" % (i, cs, ls, hs, cd, ratio, flag))

print()
if splits:
    worst = max(s[2] / max(1e-9, s[0]) for s in splits)
    print("   the largest chroma/composite ratio: %.3f" % worst)
    if worst > 0.9:
        print("   -> as suspected: the band-pass is returning more than the composite holds, so `luma = comp - chroma`")
        print("      subtracts the picture from itself.  The fix is to bound the chroma by the composite.")
    elif worst > 0.5:
        print("   -> the chroma is large but not dominant; the inversion may come from its being SUBTRACTED rather")
        print("      than from its magnitude, which means `comp - chroma` is the wrong combination for this signal.")
    else:
        print("   -> the chroma is a modest fraction of the composite, so subtracting it cannot invert the luma and")
        print("      the inversion is elsewhere in `_separate`.")
