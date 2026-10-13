"""
Measure the source at the renderer's own size, since the last comparison was between two different resolutions.

**The mistake, and it invalidated the contrast attribution.**  The "source photograph's own contrast 0.1894" was measured
on the **2364x1773** original, while the renderer's output is **533x400** -- and **downsampling lowers the standard
deviation on its own**.  So the chain was being blamed for a 0.932 -> 0.711 loss that is at least partly the resampling,
and the attribution's fourteen camera terms and its format/player/codec sweep were all compared against a figure that did
not describe the same thing.

**The check is the renderer itself as an identity.**  With every stage neutral and the mode off, whatever contrast comes
out is the *chain's* contribution at that size; and the source measured **after being resized to 533x400** is the input
at the same size.  The two are then comparable, and the difference between them is the chain's own.

**And the other reference is the footage at that size**, which is what the model is being matched to -- also resized the
same way, so all three are measured on equal terms.
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


def contrast(a, label, ref=None):
    lum = np.clip(a[sl] @ COEF, 0, 1)
    c = float(lum.std())
    print("   %-46s luma %-9.4f contrast %-9.4f %s"
          % (label, float(lum.mean()), c, "" if ref is None else "(%.3f of it)" % (c / ref)))
    return c


print("   === everything measured at the renderer's own size, %dx%d ===" % (W, H))
ns0 = load(NS0)
r = contrast(ns0, "the real NIGHTSHOT 0, resized here")
src = load(PHOTO)
s = contrast(src, "the source photograph, resized here", r)

# the original sizes, for the record
for tag, p in (("the real frame at its own size", NS0), ("the source at its own size", PHOTO)):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0
    print("   %-46s %dx%d  contrast %.4f" % (tag, a.shape[1], a.shape[0], float((a @ COEF).std())))


class One:
    height, width = H, W
    fps = 25.0
    frame_count = 1

    def frame(self, i):
        return src


BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])
print()
print("   === the renderer as an identity, so the chain's own contribution is visible ===")


def run(tag, **kw):
    layers = {"camera": kw.get("camera", {"nv_mode": 0.0}),
              "format": kw.get("fmt", {"chroma_bandwidth": 16.0, "luma_bandwidth": 24.0}),
              "player": kw.get("fmt", {"chroma_bandwidth": 16.0, "luma_bandwidth": 24.0}),
              "codec": kw.get("codec", {"dv_enabled": 0.0}),
              "display": kw.get("disp", {})}
    rr = Renderer(One(), W, H, 25.0, merge_values(BASE, layers), seed=1)
    a = np.clip(np.asarray(rr.render_frame(0), np.float32), 0, 1)
    return contrast(a, tag, s)


c_ident = run("the chain with every stage neutral, mode off")
print("      -> so the chain's own effect on the source's contrast is %.3f" % (c_ident / s))
c_full = run("the chain as it has been run (mode on)", camera={"nv_mode": 1.0, "nv_green_mix": 1.0},
             fmt={"chroma_bandwidth": 3.0, "luma_bandwidth": 6.0}, codec={"dv_enabled": 1.0})
print("      -> and with the mode on at the settings used before: %.3f of the source" % (c_full / s))
print()
print("   the footage is at %.3f of the source, so a chain that lands near that is right: %.3f against %.3f."
      % (r / s, c_full, r))
