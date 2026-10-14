"""
Write every stage's output to disk as an IMAGE, and stop measuring.

**Why this and not another statistic.**  Six statistics have now been built to find this banding and every one disagreed
with the eye:

    the row-mean spread        said the banded render had LESS structure than the footage (0.240 against 0.431)
    the one-pixel high-pass    said the render had MORE detail (0.085 against 0.079)
    the peak-to-floor ratio    peaked in bin 1 for both, so it measured the frame's vertical gradient
    two shot/read comparisons  bit-identical rows, from configuration keys that never reached the call
    the row-difference         moves by half the excess for `shot_noise` while the picture does not visibly change

**And the picture has been right every time.**  So the banding is located by *looking at the chain's own output*, which
needs no statistic at all: each stage writes a PNG, and the stage whose output first shows the bands is the one that
produces them.

**The stage trace has to go through `meta["signal"]`, not `.rgb`** -- that is the trap recorded at the start of this work,
where a trace measured `.rgb` on a chain that passes `SignalFrame` through `meta` and so saw `None` throughout.  **Both
are written here**, so a stage that transforms only one of them cannot hide.

**And the modes are traced side by side** as well, because mode 0 is clean and mode 1 is not: **the pair of traces
isolates the term** -- whatever differs between them at a given stage is a candidate, with no guessing.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from atsim.core.params import merge_values                                    # noqa: E402
from atsim.core.profiles import FACTORY_PRESETS, SCHEMA                       # noqa: E402
from atsim.render import renderer as R                                        # noqa: E402

COEF = np.array([0.299, 0.587, 0.114], np.float32)
OBJ = (r"C:\Users\Administrator\.dsh\attachments\v1\objects")
PHOTO = os.path.join(OBJ, "69", "699c23315df037b01fa831a273364b41f99c28733fb1dc928b22d1e6c281818d")
OUT = os.path.join(ROOT, "tools", "calib")
W, H, OSD = 533, 400, 0.22

_a = np.asarray(Image.open(PHOTO).convert("RGB"), np.float32) / 255.0
SRC = np.asarray(Image.fromarray((np.clip(_a, 0, 1) * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS),
                 np.float32) / 255.0


class One:
    height, width = H, W
    fps = 25.0
    frame_count = 1

    def frame(self, i):
        return SRC


BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])
OFF = {"halation": 0.0, "ccd_smear": 0.0, "flare_defocus": 0.0, "lens_flare": 0.0,
       "overflow_drain": 0.0, "chroma_aberration": 0.0, "sensor_noise": 0.0, "noise_grain": 0.0,
       "fixed_pattern": 0.0, "spike_spread": 0.0, "motion_blur": 0.0, "stuck_fraction": 0.0,
       "bloom_threshold": 0.0, "overflow_width": 0.0, "sharpness": 1.0}


def as_rgb(x):
    """Whatever a stage returns, as a displayable array: `.rgb`, a bare array, or the `signal` in its meta."""
    if x is None:
        return None
    arr = getattr(x, "rgb", None)
    if arr is None and isinstance(x, np.ndarray):
        arr = x
    if arr is None:
        sig = getattr(x, "meta", {}).get("signal") if hasattr(x, "meta") else None
        if sig is not None:
            try:
                arr = sig.to_rgb()
            except Exception:                                                   # noqa: BLE001
                arr = None
    if arr is None:
        return None
    a = np.asarray(arr, np.float32)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    if a.shape[2] > 3:
        a = a[..., :3]
    return np.clip(a, 0, 1)


traces = {}
for mode, name in ((0.0, "mode0"), (1.0, "mode1")):
    layers = {"camera": dict(OFF, nv_mode=mode),
              "format": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0},
              "player": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0},
              "codec": {"dv_enabled": 0.0}, "display": {"enabled": 0.0}}
    rr = R.Renderer(One(), W, H, 25.0, merge_values(BASE, layers), seed=20240101)
    if rr.chain is None:
        rr._build_chain()
    shots = []
    for st in rr.chain.stages:
        fn = st.apply

        def make(stage, f, _shots=shots):
            def wrapped(buf):
                out = f(buf)
                _shots.append((stage.name, as_rgb(buf), as_rgb(out)))
                return out
            return wrapped
        st.apply = make(st, fn)
    rr.render_frame(0)
    traces[name] = shots
    print("   %s: traced %d stages" % (name, len(shots)))

names = [n for n, _, _ in traces["mode1"]]
print("   the stages: %s" % ", ".join(names))


def bandiness(a):
    """Written beside every panel as a guide only -- the pictures are the evidence, not this."""
    if a is None:
        return float("nan")
    g = np.clip(np.asarray(a, np.float32) @ COEF, 0, 1)
    return float(np.abs(np.diff(g, axis=0)).mean()) / max(1e-9, float(g.std()))


#: two rows: mode 0's output at each stage, then mode 1's, so the pair that differs is visible
cw, ch = W // 3, (H - int(H * 0.22)) // 3
gap = 5
n = len(names)
cols = n + 1
canvas = Image.new("RGB", ((cw + gap) * cols + gap, (ch + 34) * 2 + 50), (16, 16, 18))
d = ImageDraw.Draw(canvas)
d.text((6, 5), "EVERY STAGE'S OUTPUT AS AN IMAGE -- mode 0 (top) and mode 1 (bottom), with the source at the left",
       fill=(242, 242, 242))


def crop(a):
    if a is None:
        return np.zeros((1, 1, 3), np.float32)
    return a[int(H * 0.22):, :, :]


src_crop = crop(SRC)
canvas.paste(Image.fromarray((src_crop * 255).astype(np.uint8)).resize((cw, ch), Image.LANCZOS), (gap, 24))
d.text((gap, 24 + ch + 3), "the SOURCE", fill=(235, 235, 235))
d.text((gap, 24 + ch + 16), "row-diff %.4f" % bandiness(SRC), fill=(215, 215, 215))
for k, name in enumerate(names):
    x = gap + (k + 1) * (cw + gap)
    for row, tag in ((0, "mode0"), (1, "mode1")):
        y = 24 + row * (ch + 34)
        before, after = traces[tag][k][1], traces[tag][k][2]
        a = crop(after if after is not None else before)
        canvas.paste(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).resize((cw, ch), Image.LANCZOS),
                     (x, y))
        if row == 1:
            d.text((x, y + ch + 3), name, fill=(235, 235, 235))
            d.text((x, y + ch + 16), "%.4f" % bandiness(after if after is not None else before),
                   fill=(215, 215, 215))
out = os.path.join(OUT, "STAGE_IMAGES.png")
canvas.save(out)
print("   written: %s" % os.path.relpath(out, ROOT))

print()
print("   === and the same trace as numbers, for reference only ===")
print("   %-12s %-12s %-12s %s" % ("stage", "mode0", "mode1", "difference"))
for k, name in enumerate(names):
    b0 = bandiness(traces["mode0"][k][2] if traces["mode0"][k][2] is not None else traces["mode0"][k][1])
    b1 = bandiness(traces["mode1"][k][2] if traces["mode1"][k][2] is not None else traces["mode1"][k][1])
    print("   %-12s %-12.5f %-12.5f %+.5f" % (name, b0, b1, b1 - b0))
