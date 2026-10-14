"""
Where the calibrated weights stop holding: the module alone against the whole chain.

**The calibration was solved on `monochrome_green`'s own expression** and predicted G/B 2.667 at `green_gain` 1.15.
**End to end the same setting measures 0.480**, so something between that function and the output redistributes the
channels -- and quoting the module-level fit as the result would be exactly the fault this work keeps finding, a figure
taken at a point where the effect does not appear.

**So each stage's per-channel chroma is measured in turn**, which is the seam method that located the inversion after
four scalar measurements had failed.  **The candidates are few and all identifiable**: the camera's own sensor terms, the
format and player Y/C stages (which demodulate and re-modulate chroma, and whose `chroma_noise` carries an explicit
per-channel shape), and the display.
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
import atsim.codecs.nightvision as NV                                         # noqa: E402

COEF = np.array([0.299, 0.587, 0.114], np.float32)
OBJ = (r"C:\Users\Administrator\.dsh\attachments\v1\objects")
PHOTO = os.path.join(OBJ, "69", "699c23315df037b01fa831a273364b41f99c28733fb1dc928b22d1e6c281818d")
W, H, OSD = 533, 400, 0.22
sl = slice(int(H * OSD), H)

src = np.asarray(Image.open(PHOTO).convert("RGB"), np.float32) / 255.0
src = np.asarray(Image.fromarray((np.clip(src, 0, 1) * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS),
                 np.float32) / 255.0


def chan(a):
    sc = np.clip(np.asarray(a, np.float32), 0, 1)[sl]
    g = np.clip(sc @ COEF, 0, 1)
    c = np.abs(sc - g[..., None]).mean(axis=(0, 1))
    return c, float(c[1] / max(1e-9, c[2])), float(c[0] / max(1e-9, c[1]))


print("   === the module alone, at the calibrated gain ===")
for gain in (1.0, 1.15):
    spec = NV.NightVisionSpec(mode=1.0, green_mix=1.0, green_gain=gain, chroma_residue=0.35,
                              chroma_noise=0.0, read_noise=0.0)
    out = np.clip(np.asarray(NV.monochrome_green(src, spec), np.float32), 0, 1)
    c, gb, rg = chan(out)
    print("      green_gain %.2f -> %s   G/B %.3f   R/G %.3f" % (gain, np.round(c, 4), gb, rg))

print()
print("   === and each stage of the chain ===")


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
layers = {"camera": dict(OFF, nv_mode=1.0, nv_green_gain=1.15),
          "format": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0},
          "player": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0},
          "codec": {"dv_enabled": 0.0}, "display": {"enabled": 0.0}}
rr = R.Renderer(One(), W, H, 25.0, merge_values(BASE, layers), seed=20240101)
if rr.chain is None:
    rr._build_chain()

prev = chan(src)
print("   %-12s %-28s %-9s %-9s %s" % ("stage", "|chroma| per channel", "G/B", "R/G", "change in G/B"))
print("   %-12s %-28s %-9.3f %-9.3f" % ("the source", np.round(prev[0], 4), prev[1], prev[2]))
for st in rr.chain.stages:
    fn = st.apply

    def make(stage, f):
        def wrapped(buf):
            out = f(buf)
            rgb = np.clip(np.asarray(getattr(out, "rgb", out), np.float32), 0, 1)
            c, gb, rg = chan(rgb)
            print("   %-12s %-28s %-9.3f %-9.3f %+.3f" % (stage.name, np.round(c, 4), gb, rg, gb - prev[1]))
            globals()["prev"] = (c, gb, rg)
            return out
        return wrapped
    st.apply = make(st, fn)
rr.render_frame(0)
print()
print("   **the stage where G/B collapses is the one that has to be understood**, and the camera is where the green")
print("   display is applied, so a change AFTER it is a redistribution of chroma the display did not choose.")
