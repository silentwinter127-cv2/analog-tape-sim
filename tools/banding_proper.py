"""
A metric that measures ROW STRUCTURE rather than pixel noise, and the final test figure.

**Why the previous metric was wrong, measured.**  `rowsig` is the mean absolute adjacent-row difference over the
picture's spread.  Tested directly on the source:

    a pure level change (gain 1.8)        rowsig 0.12737 -> 0.12291    -0.00446
    **shot_noise 0.10 alone**             rowsig 0.12737 -> **0.23482**  **+0.10745**

**so it responds almost entirely to per-pixel noise and hardly at all to level** -- which means the camera's "+0.038" and
the render's "0.10953 against the footage's 0.07562" were never measurements of banding.  **They were measurements of
noise, and they are the tenth statistic in this work to have disagreed with the eye.**

**The quantity the user actually reports is PERIODIC structure**: bands that repeat every N rows.  That is a
property of the row-mean profile's *autocorrelation* -- a periodic band pattern correlates with itself at its own period,
while both noise and ordinary content decorrelate within a few rows.  So the metric is:

    the strongest autocorrelation peak of the row-mean profile at a lag of 3 rows or more

**which is near zero for a photograph, large for a comb or a line-rate pattern, and unaffected by how noisy the picture
is** -- because the row means average the noise away.

**This is measured on the footage, the render, and at each stage**, and the figure is written for the user.
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
REF = [("NIGHTSHOT 0", os.path.join(OBJ, "ed",
                                    "ed906503e82e76777332850f677a16db3ea89089fde1c313576edf9fce66f4e0")),
       ("NIGHTSHOT 1", os.path.join(OBJ, "f2",
                                    "f2dd7803fd91d124a7c16be3c6f97df679c8a83d89631a462e9010f13a3e7c5e")),
       ("NIGHTSHOT 2", os.path.join(OBJ, "b9",
                                    "b94a0e8402a11788032bd0af9e81de7fc4713d3a648384f2c9ff57b21c21f4b6"))]
PHOTO = os.path.join(OBJ, "69", "699c23315df037b01fa831a273364b41f99c28733fb1dc928b22d1e6c281818d")
OUT = os.path.join(ROOT, "tools", "calib")
W, H, OSD = 533, 400, 0.22
sl = slice(int(H * OSD), H)


def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0
    return np.asarray(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS),
                      np.float32) / 255.0


def banding(a, min_lag=3, max_lag=60):
    """
    Periodic row structure: the strongest autocorrelation of the row-mean profile beyond a few rows.

    **The row means are what makes this a banding measure and not a noise measure**: averaging each row removes the
    per-pixel noise, so what is left is the structure that repeats down the frame.  A lag below `min_lag` is excluded
    because that is where ordinary vertical content and residual noise live.
    """
    g = np.clip(np.asarray(a, np.float32)[sl] @ COEF, 0, 1)
    r = g.mean(axis=1).astype(np.float64)
    r = r - r.mean()
    if r.std() < 1e-9:
        return 0.0, 0
    n = len(r)
    ac = np.correlate(r, r, mode="full")[n - 1:] / (r @ r)
    hi = min(max_lag, n - 1)
    if hi <= min_lag:
        return 0.0, 0
    k = int(np.argmax(ac[min_lag:hi])) + min_lag
    return float(ac[k]), k


def rowsig(a):
    """The old measure, kept beside the new one so the difference is visible rather than asserted."""
    g = np.clip(np.asarray(a, np.float32)[sl] @ COEF, 0, 1)
    return float(np.abs(np.diff(g, axis=0)).mean()) / max(1e-9, float(g.std()))


SRC = load(PHOTO)
print("   === the footage and the input, under BOTH measures ===")
print("   %-24s %-14s %-14s %s" % ("", "banding (new)", "rowsig (old)", "the lag"))
_b, _k = banding(SRC)
print("   %-24s %-14.4f %-14.5f %d" % ("the source photograph", _b, rowsig(SRC), _k))
for name, p in REF:
    b, k = banding(load(p))
    print("   %-24s %-14.4f %-14.5f %d" % (name, b, rowsig(load(p)), k))


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


def build(mode=1.0, **cam):
    layers = {"camera": dict(OFF, nv_mode=mode, **cam),
              "format": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0},
              "player": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0},
              "codec": {"dv_enabled": 0.0}, "display": {"enabled": 0.0}}
    rr = R.Renderer(One(), W, H, 25.0, merge_values(BASE, layers), seed=20240101)
    if rr.chain is None:
        rr._build_chain()
    return rr


print()
print("   === where the BANDING appears, stage by stage ===")
rr = build()
log = []
for st in rr.chain.stages:
    fn = st.apply

    def make(stage, f):
        def wrapped(buf):
            out = f(buf)
            arr = getattr(out, "rgb", None)
            if arr is None and hasattr(out, "meta"):
                sig = out.meta.get("signal")
                arr = sig.to_rgb() if sig is not None else None
            if arr is not None:
                a = np.clip(np.asarray(arr, np.float32), 0, 1)
                bb, kk = banding(a)
                log.append((stage.name, bb, kk, rowsig(a)))
            return out
        return wrapped
    st.apply = make(st, fn)
rr.render_frame(0)
print("   %-12s %-14s %-8s %-14s %s" % ("stage", "banding", "lag", "rowsig", "banding vs the input"))
b_src = banding(SRC)[0]
print("   %-12s %-14.4f %-8s %-14.5f" % ("(the input)", b_src, "-", rowsig(SRC)))
for name, b, k, rs in log:
    print("   %-12s %-14.4f %-8d %-14.5f %+.4f" % (name, b, k, rs, b - b_src))

print()
print("   === and the tests worth keeping, under the new measure ===")
print("   %-34s %-14s %-12s %s" % ("", "banding", "lag", "rowsig"))
CASES = [("mode 0 (the chain, no night vision)", dict(mode=0.0)),
         ("mode 1 as it ships", dict(mode=1.0)),
         ("mode 1, shot_noise = 0", dict(mode=1.0, nv_shot_noise=0.0)),
         ("mode 1, labelling off", dict(mode=1.0, nv_nir_material=0.0)),
         ("mode 2 (Super)", dict(mode=2.0))]
final = {}
for tag, kw in CASES:
    mode = kw.pop("mode")
    a = np.clip(np.asarray(build(mode, **kw).render_frame(0), np.float32), 0, 1)
    b, k = banding(a)
    final[tag] = a
    print("   %-34s %-14.4f %-12d %-14.5f" % (tag, b, k, rowsig(a)))

gap = 7
cw, ch = W // 2, (H - int(H * OSD)) // 2
panels = [("the real NIGHTSHOT 0", load(REF[0][1])),
          ("the source photograph", SRC),
          ("the render, mode 1 (as it ships)", final["mode 1 as it ships"]),
          ("the render, mode 0", final["mode 0 (the chain, no night vision)"])]
canvas = Image.new("RGB", ((cw + gap) * 2 + gap, (ch + 34) * 2 + 30), (16, 16, 18))
d = ImageDraw.Draw(canvas)
d.text((6, 5), "TEST FIGURE -- the footage, the input, and the render, at 533x400 with no crop",
       fill=(242, 242, 242))
for i, (tag, a) in enumerate(panels):
    x = gap + (i % 2) * (cw + gap)
    y = 26 + (i // 2) * (ch + 34)
    canvas.paste(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).resize((cw, ch), Image.LANCZOS), (x, y))
    bb, kk = banding(a)
    d.text((x, y + ch + 4), tag, fill=(238, 238, 238))
    d.text((x, y + ch + 17), "banding %.4f at lag %d   |   rowsig %.5f" % (bb, kk, rowsig(a)), fill=(214, 214, 214))
out = os.path.join(OUT, "TEST_FIGURE.png")
canvas.save(out)
print()
print("   written: %s" % os.path.relpath(out, ROOT))
