"""
Attribute the drift channel by channel: which one is being pulled, and by what.

The global number says the output loses saturation -- R-B falls from +0.734 to +0.65 -- and that the *most neutral*
settings lose more, not less, so the drift is in the signal path rather than in any of the mechanisms. What that
does not say is which channel moves and where in the chain it happens.

So the measurement is the three channel means at each stage boundary, on the same clean input, by reading the
pipeline's own frame at each point rather than the final render. The stages that can change colour are the camera's
white balance and colour science, the format's encoder, the codec, and the display -- and each is switched off in
turn while the channel means are printed, so the channel that moves is attributed to a stage.
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from PIL import Image                                                        # noqa: E402
from atsim.core.params import merge_values                                   # noqa: E402
from atsim.core.pipeline import FrameBuffer                                  # noqa: E402
from atsim.core.profiles import FACTORY_PRESETS, SCHEMA                      # noqa: E402
from atsim.render.renderer import Renderer                                   # noqa: E402

OUT = os.path.join(ROOT, "tools", "ntscrs_ref")
full = np.asarray(Image.open(os.path.join(OUT, "clean_source.jpg")).convert("RGB"),
                  np.float32) / 255.0
# a modest working size, and the same crop for every comparison
H, W = 360, 480
im = Image.fromarray((full * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
SRC = np.asarray(im, np.float32) / 255.0


class Still:
    def __init__(self, img):
        self.width, self.height, self.fps, self.frame_count = W, H, 25.0, 3
        self._img = img

    def frame(self, index):
        return self._img


def means(a):
    return a.reshape(-1, 3).mean(axis=0)


print("   the source at %dx%d: means %s, R-B %+.4f"
      % (W, H, np.round(means(SRC), 4), float(means(SRC)[0] - means(SRC)[2])))

BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])


def render(**layers):
    p = merge_values(BASE, layers)
    r = Renderer(Still(SRC), W, H, 25.0, p, seed=5)
    return r


def at_stage(stage_name, **layers):
    """The picture as it leaves one named stage."""
    r = render(**layers)
    r._build_chain()
    src = np.asarray(SRC, np.float32)
    buf = FrameBuffer(src, None, 1, 0.04, r.std, {"field": 0})
    order = r.chain.stages if hasattr(r.chain, "stages") else None
    for st in (order or []):
        buf = st.apply(buf)
        if st.name == stage_name or type(st).__name__.lower().startswith(stage_name):
            out = buf.rgb if buf.rgb is not None else buf.signal.to_rgb()
            return np.clip(np.asarray(out, np.float32), 0, 1)
    return None


print()
print("   stage by stage, with the display off so the tube is not in the path")
show = {"display": {"enabled": 0.0}, "camera": {"exposure": 0.0}}
r = render(**show)
r._build_chain()

CANDIDATES = ["white_balance", "roll_off", "color_science", "format", "codec", "tape", "player"]
src = np.asarray(SRC, np.float32)
buf = FrameBuffer(src, None, 1, 0.04, r.std, {"field": 0})
print("   %-18s %-26s %s" % ("after stage", "channel means", "R-B"))
print("   %-18s %-26s %+.4f" % ("(the source)", np.round(means(src), 4), float(means(src)[0] - means(src)[2])))
for st in (r.chain.stages if hasattr(r.chain, "stages") else []):
    buf = st.apply(buf)
    out = buf.rgb if buf.rgb is not None else buf.signal.to_rgb()
    out = np.clip(np.asarray(out, np.float32), 0, 1)
    m = means(out)
    print("   %-18s %-26s %+.4f" % (st.name, np.round(m, 4), float(m[0] - m[2])))
