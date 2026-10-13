"""
Measure the contrast stage by stage, so the compression is located rather than attributed by elimination.

**What the eliminations established, each by measurement:**

    display    disabling it, flattening its gamma and taking its saturation to 1.0 leave the contrast bit-identical
    camera     14 terms neutralised individually move it by at most 0.0005, and all at once makes it worse
    format     bandwidths at 3.0, 8.0 and 16.0 MHz are identical
    player     the same
    codec      `dv_enabled` on or off is identical

**and with every one of those neutral, the chain still takes the source's contrast from 0.1894 to 0.1347** -- 71% --
while the footage sits at 1.073 of the source, so the chain should be preserving it rather than losing it.

**So the compression is in the fixed path**, and the way to find it is to read the contrast after each stage in order
rather than to keep neutralising parameters.  `STAGE_ORDER` is
`camera, format, codec, connector, tape, player, display, overlay`, and **`connector` is the one this session has never
touched at all.**

**The probe drives the renderer stage by stage** by asking it to render with the stage list truncated, if it supports
that, and otherwise by calling each stage directly on the previous one's output.  Either way the quantity is the same:
the luminance's standard deviation after each stage, so the step where it drops is visible as a step.
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
S0 = float((src @ COEF).std())
print("   the source at this size: contrast %.4f" % S0)

print()
print("   === what the renderer exposes ===")
print("      STAGE_ORDER = %s" % (getattr(R, "STAGE_ORDER", "not found"),))
for name in ("Renderer", "render_frame", "render_frames"):
    print("      %-14s %s" % (name, "present" if hasattr(R, name) else "absent"))

import inspect                                                                # noqa: E402
sig = inspect.signature(R.Renderer.__init__)
print("      Renderer(%s)" % ", ".join(sig.parameters))
print()
print("   the render_frame signature and its own text, to see whether a stage list can be given:")
try:
    print("      %s" % inspect.signature(R.Renderer.render_frame))
except Exception as exc:                                                       # noqa: BLE001
    print("      %s" % exc)
src_txt = inspect.getsource(R.Renderer.render_frame)
print()
print("   the first 40 lines of render_frame, which is where the stage order is walked:")
for line in src_txt.split("\n")[:40]:
    if line.strip():
        print("      %s" % line[:112])
