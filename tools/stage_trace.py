"""
Wrap each stage's `apply` and measure the contrast after every one, so the compression is located as a step.

**Why this is the right instrument and the eliminations were not.**  Neutralising parameters showed that *no setting*
causes the loss, but that cannot say *where* it happens -- only that it is in the fixed path.  The renderer's own loop is

    for pos in range(start, len(STAGE_ORDER)):
        st = self.chain.stages[pos]
        buf = st.apply(buf)

**so wrapping `apply` records the picture after each stage in order**, and the step where the luminance's standard
deviation drops is the answer.  `FrameBuffer` carries its pixels in `.rgb`, which is what the loop hands along.

**And the reference is measured on the same frame and at the same size**: the source as the renderer receives it, before
any stage runs.  That is the input the chain is acting on, so every later figure is comparable with it -- which is the
mistake that invalidated the previous round's numbers (a 2364x1773 source against a 533x400 render).
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
PHOTO = os.path.join(OBJ, "69", "699c23315df037b01fa831a273364b41f99c28733fb1dc928b22d1e6c281818d")
W, H = 533, 400

src = np.asarray(Image.open(PHOTO).convert("RGB"), np.float32) / 255.0
src = np.asarray(Image.fromarray((np.clip(src, 0, 1) * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS),
                 np.float32) / 255.0


class One:
    height, width = H, W
    fps = 25.0
    frame_count = 1

    def frame(self, i):
        return src


BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])

#: the chain as it has been run for every measurement in this session
LAYERS = {"camera": {"nv_mode": 1.0, "nv_green_mix": 1.0},
          "format": {"chroma_bandwidth": 3.0, "luma_bandwidth": 6.0},
          "player": {"chroma_bandwidth": 3.0, "luma_bandwidth": 6.0},
          "codec": {"dv_enabled": 1.0}}

#: and the same chain with everything neutral, so the fixed path is isolated
NEUTRAL = {"camera": {"nv_mode": 0.0, "halation": 0.0, "ccd_smear": 0.0, "flare_defocus": 0.0,
                      "lens_flare": 0.0, "overflow_drain": 0.0, "chroma_aberration": 0.0,
                      "sensor_noise": 0.0, "noise_grain": 0.0, "fixed_pattern": 0.0,
                      "spike_spread": 0.0, "motion_blur": 0.0, "stuck_fraction": 0.0,
                      "dynamic_range": 14.0, "sharpness": 1.0,
                      "bloom_threshold": 0.0, "overflow_width": 0.0},
           "format": {"chroma_bandwidth": 16.0, "luma_bandwidth": 24.0},
           "player": {"chroma_bandwidth": 16.0, "luma_bandwidth": 24.0},
           "codec": {"dv_enabled": 0.0}, "display": {"enabled": 0.0}}


def trace(tag, layers):
    print()
    print("   === %s ===" % tag)
    rr = R.Renderer(One(), W, H, 25.0, merge_values(BASE, layers), seed=1)
    log = []

    def measure(name, rgb):
        a = np.clip(np.asarray(rgb, np.float32), 0, 1)
        log.append((name, float(a.mean()), float((a @ COEF).std())))

    measure("(the source as received)", src)
    # wrap every stage's apply
    rr._build_chain() if rr.chain is None else None
    orig = []
    for st in rr.chain.stages:
        orig.append((st, st.apply))

        def make(stage, fn):
            def wrapped(buf):
                out = fn(buf)
                measure(stage.name, getattr(out, "rgb", out))
                return out
            return wrapped
        st.apply = make(st, st.apply)
    rr.render_frame(0)
    base = log[0][2]
    print("   %-16s %-10s %-11s %-11s %s" % ("stage", "luma", "contrast", "change", "vs the source"))
    prev = None
    for name, lm, c in log:
        print("   %-16s %-10.4f %-11.4f %-11s %.3f"
              % (name, lm, c, "" if prev is None else "%+.4f" % (c - prev), c / max(1e-9, base)))
        prev = c
    for st, fn in orig:
        st.apply = fn


trace("the chain as every measurement in this session has run it", LAYERS)
trace("the same chain with every neutralisable stage made neutral", NEUTRAL)
print()
print("   the step where the contrast falls is the one to fix; `connector` and `tape` are stages this session has")
print("   never touched, and they appear between `codec` and `player` in the order.")
