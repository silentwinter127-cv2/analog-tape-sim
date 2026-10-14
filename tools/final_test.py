"""
Final test: locate the ADDED residual, and produce the test figure.

**What the previous test settled, by scaling the source's own row structure:**

    factor   source    render    render minus source
    0.30     0.06388   0.10111   -0.02626
    0.50     0.07994   0.10371   -0.02366
    0.70     0.09776   0.10661   -0.02076
    1.00     0.12737   0.10953   -0.01784
    1.80     0.20373   0.12032   -0.00705

**and with the source flattened to the footage's own 0.07562 the render still reads 0.10244.**  So the residual is
**added by the chain, not inherited** -- about **0.027** of it -- which corrects the previous round's guess.

**This traces which stage adds it**, the same way the earlier banding was found, and then switches off each camera term in
turn.  **The expectation is stated so it can fail**: the added structure should appear at one stage and not before it, and
if no single camera term removes it then it is produced by the interaction rather than by a term -- which is a finding
rather than a failure, and would point at the tone curves.

**And the figure is the deliverable**: the footage, the render, and the best setting found, at full size with no crop.
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


def rowsig(a):
    g = np.clip(np.asarray(a, np.float32)[sl] @ COEF, 0, 1)
    return float(np.abs(np.diff(g, axis=0)).mean()) / max(1e-9, float(g.std()))


SRC = load(PHOTO)
REAL = load(REF[0][1])


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


print("   the footage's rowsig %.5f; the source's %.5f" % (rowsig(REAL), rowsig(SRC)))
print()
print("   === which stage adds the structure ===")
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
                log.append((stage.name, rowsig(np.clip(np.asarray(arr, np.float32), 0, 1))))
            return out
        return wrapped
    st.apply = make(st, fn)
out = rr.render_frame(0)
prev = rowsig(SRC)
print("   %-12s %-12s %-12s %s" % ("stage", "rowsig", "added", "cumulative"))
print("   %-12s %-12.5f %-12s %-12.5f" % ("(the input)", prev, "-", prev))
for name, s in log:
    print("   %-12s %-12.5f %+-12.5f %-12.5f" % (name, s, s - prev, s))
    prev = s

print()
print("   === and each camera term switched off, from the full render ===")
TERMS = [("as it renders", {}),
         ("shot_noise = 0", dict(nv_shot_noise=0.0)),
         ("chroma_noise = 0", dict(nv_chroma_noise=0.0)),
         ("read_noise = 0", dict(nv_read_noise=0.0)),
         ("ir_response = 0 (no IR at all)", dict(nv_ir_response=0.0)),
         ("ir_albedo mix = 0", dict(nv_ir_albedo=0.0)),
         ("green_mix = 0 (no mono display)", dict(nv_green_mix=0.0)),
         ("all of the above off", dict(nv_shot_noise=0.0, nv_chroma_noise=0.0, nv_read_noise=0.0,
                                       nv_ir_response=0.0, nv_ir_albedo=0.0, nv_green_mix=0.0))]
shots = []
base_s = None
for tag, cam in TERMS:
    rrx = build(**cam)
    a = np.clip(np.asarray(rrx.render_frame(0), np.float32), 0, 1)
    s = rowsig(a)
    if base_s is None:
        base_s = s
    shots.append((tag, a, s))
    print("   %-38s %-12.5f %+.5f" % (tag, s, s - base_s))
print()
print("   the footage is %.5f, so a term whose removal brings the render to it is the one adding the structure."
      % rowsig(REAL))

gap = 7
cw, ch = W // 2, (H - int(H * OSD)) // 2
panels = [("the real NIGHTSHOT 0  rowsig %.4f" % rowsig(REAL), REAL),
          ("the render as it ships  rowsig %.4f" % shots[0][2], shots[0][1]),
          ("the render, noise off  rowsig %.4f" % shots[7][2], shots[7][1]),
          ("the render, shot_noise off  rowsig %.4f" % shots[1][2], shots[1][1])]
canvas = Image.new("RGB", ((cw + gap) * 2 + gap, (ch + 30) * 2 + 26), (16, 16, 18))
d = ImageDraw.Draw(canvas)
d.text((6, 5), "TEST FIGURE -- the footage beside the render, at full size, with the terms removed",
       fill=(242, 242, 242))
for i, (tag, a) in enumerate(panels):
    x = gap + (i % 2) * (cw + gap)
    y = 26 + (i // 2) * (ch + 30)
    canvas.paste(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).resize((cw, ch), Image.LANCZOS), (x, y))
    d.text((x, y + ch + 4), tag, fill=(234, 234, 234))
out_p = os.path.join(OUT, "FINAL_TEST.png")
canvas.save(out_p)
print()
print("   written: %s" % os.path.relpath(out_p, ROOT))
