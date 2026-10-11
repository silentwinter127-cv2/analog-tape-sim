"""
Find which loop costs what, and why the selftest is twice as slow as a direct measurement.

Direct measurement at idle: loops off 12.67 fps, all three on 6.37 fps. The selftest measured 6.6 fps on
what should be the default preset -- the same figure as all three loops on. So either a loop is enabled
in the preset the selftest uses, or one loop alone costs as much as all three.

Measured per loop rather than reasoned about, because the two candidate explanations need different
fixes: an enabled-by-default parameter is a preset bug, a single expensive loop is a performance bug.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atsim.core.params import merge_values  # noqa: E402
from atsim.core.profiles import FACTORY_PRESETS, SCHEMA  # noqa: E402
from atsim.io.source import SyntheticSource  # noqa: E402
from atsim.render.renderer import Renderer  # noqa: E402

base = FACTORY_PRESETS["family_archive_1992"]["layers"]
p = SCHEMA.resolve_all(base)
print("what the preset actually sets:")
for k in ("ae_enabled", "awb_enabled", "af_enabled", "autopresettle_s"):
    print("   camera.%-18s = %s" % (k, p["camera"].get(k)))
print()

CASES = [
    ("nothing on", {}),
    ("AE only", {"ae_enabled": 1.0}),
    ("AWB only", {"awb_enabled": 1.0}),
    ("AF only", {"af_enabled": 1.0}),
    ("AE + AWB", {"ae_enabled": 1.0, "awb_enabled": 1.0}),
    ("all three", {"ae_enabled": 1.0, "awb_enabled": 1.0, "af_enabled": 1.0}),
]
print("%-14s %8s %12s" % ("case", "fps", "ms/frame"))
print("-" * 36)
for label, over in CASES:
    src = SyntheticSource(width=360, height=240, frames=40)
    params = merge_values(base, {"camera": over})
    r = Renderer(src, 360, 240, src.fps, params, seed=3)
    for i in range(4):
        r.render_frame(i)
    n, t0 = 12, time.perf_counter()
    for i in range(4, 4 + n):
        r.render_frame(i)
    dt = time.perf_counter() - t0
    print("%-14s %8.2f %12.0f" % (label, n / dt, 1000 * dt / n))

print()
print("and the cost inside each loop, on one frame:")
from atsim.codecs.autocontrol import (ExposureLoop, FocusLoop, WhiteBalanceLoop,  # noqa: E402
                                      centre_weights)
frame = np.random.default_rng(1).random((240, 360, 3)).astype(np.float32)
w = centre_weights(240, 360)
loops = [("exposure", ExposureLoop(target=0.46, tau_s=1.0)),
         ("white balance", WhiteBalanceLoop(tau_s=1.6)),
         ("focus", FocusLoop(tau_s=0.9))]
for name, lp in loops:
    t0 = time.perf_counter()
    for _ in range(50):
        if name == "exposure":
            lp.update(frame, 0.04, w)
        else:
            lp.update(frame, 0.04)
    dt = (time.perf_counter() - t0) / 50
    print("   %-14s %8.3f ms/call" % (name, dt * 1000))
