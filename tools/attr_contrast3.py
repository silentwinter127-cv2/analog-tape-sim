"""
Test the format and player stages, which are the only ones the attribution has not ruled out.

**What has been eliminated, each by measurement rather than by reasoning:**

    the display stage      disabling it, flattening its gamma, and taking its saturation to 1.0 all leave the contrast
                           at a bit-identical 0.490 -- so it contributes nothing
    every camera term      neutralising each in turn moves the contrast by at most 0.0005, and neutralising all
                           fourteen at once makes it *worse* (0.1347 -> 0.1120)

**and the source photograph's own contrast is 0.1894 against the chain's 0.1347** -- so a picture with 93% of the
footage's contrast enters the chain and 71% of it leaves, **before the night-vision mode does anything at all**.

**That leaves `format` and `player`**, and they are the right suspects on physical grounds: a **digital component**
recording subsamples its chroma and quantises it coarsely, and this chain has `codec.dv_enabled` on with a
`chroma_bandwidth` of 3.0 MHz on both stages.  **A 4:1:1 chroma and a narrow band would flatten a picture's colour
contrast while leaving its luminance shape alone** -- which is what the numbers show, since the *luma* standard deviation
falls while the level stays put.

**So each is made a pass-through in turn**, with the bandwidths opened and the codec disabled, and the contrast recorded
against the source's own 0.1894.
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


real = load(NS0)
R_STD = float(np.clip(real[sl] @ COEF, 0, 1).std())
src = load(PHOTO)
S_STD = float(np.clip(src[sl] @ COEF, 0, 1).std())
print("   the footage's contrast %.4f   the source's own %.4f   (the target: the chain must not lose the source's)"
      % (R_STD, S_STD))


class One:
    height, width = H, W
    fps = 25.0
    frame_count = 1

    def frame(self, i):
        return src


BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])


def run(tag, fmt=None, codec=None):
    layers = {"camera": {"nv_mode": 0.0},
              "format": dict(fmt or {"chroma_bandwidth": 3.0, "luma_bandwidth": 6.0}),
              "player": dict(fmt or {"chroma_bandwidth": 3.0, "luma_bandwidth": 6.0}),
              "codec": dict(codec or {})}
    rr = Renderer(One(), W, H, 25.0, merge_values(BASE, layers), seed=1)
    a = np.clip(np.asarray(rr.render_frame(0), np.float32), 0, 1)
    lum = np.clip(a[sl] @ COEF, 0, 1)
    print("   %-50s %-10.4f %-11.4f %.3f" % (tag, float(lum.mean()), float(lum.std()), float(lum.std()) / S_STD))
    return float(lum.std())


print()
print("   === the format/player/codec stages ===")
print("   %-50s %-10s %-11s %s" % ("", "luma", "contrast", "vs the SOURCE"))
run("as the chain runs them (3.0 / 6.0 MHz, DV on)")
run("...bandwidths wide open (8.0 / 12.0 MHz)", {"chroma_bandwidth": 8.0, "luma_bandwidth": 12.0})
run("...bandwidths very wide (16.0 / 24.0 MHz)", {"chroma_bandwidth": 16.0, "luma_bandwidth": 24.0})
run("...DV disabled", {"chroma_bandwidth": 3.0, "luma_bandwidth": 6.0}, {"dv_enabled": 0.0})
run("...DV disabled and bandwidths very wide", {"chroma_bandwidth": 16.0, "luma_bandwidth": 24.0},
    {"dv_enabled": 0.0})

print()
print("   === and the whole chain with every stage that can be neutralised made neutral ===")
run("every stage neutral", {"chroma_bandwidth": 16.0, "luma_bandwidth": 24.0}, {"dv_enabled": 0.0})
print()
print("   if the contrast recovers here, the loss is the band limiting or the DV codec; if it does not, then the")
print("   source's own 0.1894 was never reachable through this chain and the difference is in the camera stage's")
print("   response curve rather than in any single term.")
