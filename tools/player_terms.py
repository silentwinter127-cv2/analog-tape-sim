"""
List the player stage's factory defaults, then neutralise them one at a time to find the 30%.

**Where the last round left it.**  The stage trace showed the contrast falling in two places and nowhere else:

    the chain as run         source 0.2111 -> camera 0.1510 (-0.0601) -> everything else 0.0000
    everything neutralised   source 0.2111 -> camera 0.1903 (-0.0208) -> player 0.1337 (-0.0566)

**and `format`, `codec`, `connector` and `tape` contribute exactly 0.0000 in both**, so the loss is the camera's
factory defaults plus one in `player`.  **`player`'s bandwidth terms were swept repeatedly in this session and are
provably not it** -- nine combinations differing by 0.0004 -- so the term responsible is one of the others, which this
session has never looked at.

**So the defaults are listed first**, since the term has to be among them, and then each is neutralised in turn with the
stage trace measuring the contrast after `player`.  **The trace is the instrument that works here**: neutralising a
parameter only says it does nothing, while the trace says which stage the step happens in, and the two together are what
identified the last two findings.
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

print("   === the player stage's factory defaults ===")
player = BASE.get("player", {})
for k in sorted(player):
    print("      %-28s %s" % (k, player[k]))
print()
print("   === and the other stages' defaults, for completeness ===")
for stage in ("format", "codec", "connector", "tape"):
    d = BASE.get(stage, {})
    if d:
        print("      %s: %s" % (stage, ", ".join("%s=%s" % (k, d[k]) for k in sorted(d))))

#: the base chain: camera terms neutral so the player's own contribution is not masked by the camera's
CAM_NEUTRAL = {"nv_mode": 0.0, "halation": 0.0, "ccd_smear": 0.0, "flare_defocus": 0.0,
               "lens_flare": 0.0, "overflow_drain": 0.0, "chroma_aberration": 0.0,
               "sensor_noise": 0.0, "noise_grain": 0.0, "fixed_pattern": 0.0,
               "spike_spread": 0.0, "motion_blur": 0.0, "stuck_fraction": 0.0,
               "dynamic_range": 14.0, "sharpness": 1.0, "bloom_threshold": 0.0, "overflow_width": 0.0}


def contrast_after_player(**player_over):
    layers = {"camera": CAM_NEUTRAL,
              "format": {"chroma_bandwidth": 16.0, "luma_bandwidth": 24.0},
              "player": dict({"chroma_bandwidth": 16.0, "luma_bandwidth": 24.0}, **player_over),
              "codec": {"dv_enabled": 0.0}}
    rr = R.Renderer(One(), W, H, 25.0, merge_values(BASE, layers), seed=1)
    got = {}

    def measure(name, rgb):
        a = np.clip(np.asarray(rgb, np.float32), 0, 1)
        got[name] = float((a @ COEF).std())

    if rr.chain is None:
        rr._build_chain()
    for st in rr.chain.stages:
        fn = st.apply

        def make(stage, f):
            def wrapped(buf):
                out = f(buf)
                measure(stage.name, getattr(out, "rgb", out))
                return out
            return wrapped
        st.apply = make(st, fn)
    rr.render_frame(0)
    return got.get("camera", float("nan")), got.get("player", float("nan"))


print()
print("   === neutralising each player term in turn (the camera is already neutral) ===")
c0, p0 = contrast_after_player()
print("   %-30s %-14s %-14s %s" % ("setting", "after camera", "after player", "the player's own change"))
print("   %-30s %-14.4f %-14.4f %.4f" % ("as it stands", c0, p0, p0 - c0))
best = (abs(p0 - c0) * 0, "", p0)
for k in sorted(player):
    if k in ("chroma_bandwidth", "luma_bandwidth"):
        continue
    # every one of these is a defect term, so its neutral value is zero
    c1, p1 = contrast_after_player(**{k: 0.0})
    chg = p1 - c0
    print("   %-30s %-14.4f %-14.4f %+.4f%s"
          % ("%s = 0" % k, c1, p1, chg, "   <- recovers the most" if abs(chg) > abs(best[0]) else ""))
    if abs(chg) > abs(best[0]):
        best = (chg, k, p1)
print()
print("   the term whose neutralisation recovers the most: %s (%+.4f)" % (best[1] or "none", best[0]))
print("   the source at this size is 0.2111, and the camera-neutral chain reads %.4f after the camera." % c0)
