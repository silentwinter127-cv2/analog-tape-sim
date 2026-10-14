"""
The right comparison: drive the model with the camcorder's OWN frames, then compare the channel ratios.

**What went wrong in the previous round, and it is a measurement fault rather than a tuning one.**  The calibration was
fitted against a photograph of an RGB case with **red, magenta and pink LED emitters in it**, and those emitters are
*visible-light* sources: under this camcorder's NightShot -- IR-cut filter removed, near-infrared lamp on -- they contribute
to the picture in a way that a visible-light photograph does not record.  **So no amount of tuning on that input can reach
the footage, and the render's `R/G` of 1.34 against the footage's 0.37-0.85 is what that looks like.**

**And the footage's own spread is now measured**, one frame per second across the whole NightShot stretch:

    t = 52-53s    mean 0.11-0.18   R/G 0.85-0.81     (a transitional moment)
    t = 54-67s    mean 0.22-0.30   R/G 0.49-0.64     (NIGHTSHOT 1)
    t = 68-82s    mean 0.33-0.39   R/G 0.37-0.55     (NIGHTSHOT 2)

**so the target is not one frame but a range, and the green deepens as the clip goes on.**  **A single reference frame was
therefore the wrong target to fit**, which is the same fault as choosing the wrong configuration to sweep -- and it is the
reason "看上去不一样" kept coming back with different reference frames.

**The comparison that is actually fair**: the camera's NightShot mode is fed the *same scene* the camcorder saw.  The
camcorder's own frames are the only record of that scene under that light, so **the same frame is used twice** -- once as
the model's input (a plain RGB picture, which is what the chain takes) and once as the reference (the real NightShot
output).  **The model is then asked whether it reproduces the transformation, not whether it can invent near-infrared
content that the input never carried.**

**What this cannot test** is stated plainly: the input already IS a NightShot frame, so it carries the camcorder's own
green.  **The chain should therefore mostly preserve it rather than create it**, and a chain that turns it magenta would
fail loudly -- which is exactly the failure the previous round's input was hiding.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from atsim.core.params import merge_values                                    # noqa: E402
from atsim.core.profiles import FACTORY_PRESETS, SCHEMA                       # noqa: E402
from atsim.render.renderer import Renderer                                    # noqa: E402

COEF = np.array([0.299, 0.587, 0.114], np.float32)
FRAMES = sorted(glob.glob(os.path.join(ROOT, ".server", "nvframes", "*.png")))
OUT = os.path.join(ROOT, "tools", "calib")
W, H, OSD = 533, 400, 0.22
sl = slice(int(H * OSD), H)


def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0
    return np.asarray(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS),
                      np.float32) / 255.0


def fig(a):
    x = np.clip(np.asarray(a, np.float32), 0, 1)[sl]
    g = np.clip(x @ COEF, 0, 1)
    r, gg, b = (float(x[..., i].mean()) for i in range(3))
    u, v = x[..., 0] - x[..., 1], x[..., 2] - x[..., 1]
    m = np.sqrt(u * u + v * v)
    k = m > max(1e-4, float(np.quantile(m, 0.5)))
    ang = np.arctan2(v, u)[k]
    return dict(mean=float(g.mean()), std=float(g.std()), p95=float(np.quantile(g, 0.95)),
                RG=r / max(1e-6, gg), BG=b / max(1e-6, gg),
                hue=float(np.angle(np.exp(1j * ang).mean()) * 180.0 / np.pi) if k.any() else 0.0,
                lead=float((x[..., 0] > x[..., 1]).mean()))


BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])
OFF = {"halation": 0.0, "ccd_smear": 0.0, "flare_defocus": 0.0, "lens_flare": 0.0,
       "overflow_drain": 0.0, "chroma_aberration": 0.0, "sensor_noise": 0.0, "noise_grain": 0.0,
       "fixed_pattern": 0.0, "spike_spread": 0.0, "motion_blur": 0.0, "stuck_fraction": 0.0,
       "bloom_threshold": 0.0, "overflow_width": 0.0, "sharpness": 1.0}


class One:
    height, width = H, W
    fps = 25.0
    frame_count = 1

    def __init__(self, a):
        self.a = a

    def frame(self, i):
        return self.a


def run(src, **cam):
    layers = {"camera": dict(OFF, **cam),
              "format": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0},
              "player": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0},
              "codec": {"dv_enabled": 0.0}, "display": {"enabled": 0.0}}
    rr = Renderer(One(src), W, H, 25.0, merge_values(BASE, layers), seed=20240101)
    return np.clip(np.asarray(rr.render_frame(0), np.float32), 0, 1)


COLS = [("mean", "mean"), ("std", "std"), ("p95", "p95"), ("R/G", "RG"), ("B/G", "BG"),
        ("hue", "hue"), ("%R>G", "lead")]

print("   the model is FED each real NightShot frame, and the result compared with that same frame.")
print("   **the input already carries the camcorder's green, so the chain should preserve it, not create it.**")
print()
print("   %-6s %-38s %s" % ("t(s)", "the real frame (also the target)", "the model's output"))
print("   %-6s %s" % ("", "  ".join("%-8s" % c for c, _ in COLS)))
worst = None
for i, f in enumerate(FRAMES):
    src = load(f)
    out = run(src, nv_mode=2.0)
    fr, fo = fig(src), fig(out)
    t = 52 + i
    print("   %-6d %s" % (t, "  ".join("%-8.4f" % fr[k] for _, k in COLS)))
    print("   %-6s %s" % ("", "  ".join("%-8.4f" % fo[k] for _, k in COLS)))
    err = abs(fo["RG"] - fr["RG"]) / max(1e-6, fr["RG"]) + abs(fo["mean"] - fr["mean"]) / max(1e-6, fr["mean"])
    if worst is None or err > worst[0]:
        worst = (err, t, src, out)
print()
print("   **the largest disagreement: t = %ds, error %.3f**" % (worst[1], worst[0]))
gap = 7
cw, ch = W // 2, (H - int(H * OSD)) // 2
canvas = Image.new("RGB", ((cw + gap) * 2 + gap, (ch + 34) * 2 + 28), (16, 16, 18))
d = ImageDraw.Draw(canvas)
d.text((6, 5), "FED THE REAL NIGHTSHOT FRAMES -- the chain should preserve the green it was handed, not invent it",
       fill=(242, 242, 242))
for i, (tag, arr) in enumerate([("the real frame, t=%ds (target)" % worst[1], worst[2]),
                                ("the model fed that frame", worst[3]),
                                ("the real frame, t=60s", load(FRAMES[8])),
                                ("the model fed t=60s", run(load(FRAMES[8]), nv_mode=2.0))]):
    x = gap + (i % 2) * (cw + gap)
    y = 24 + (i // 2) * (ch + 34)
    canvas.paste(Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).resize((cw, ch), Image.LANCZOS), (x, y))
    s = fig(arr)
    d.text((x, y + ch + 3), tag, fill=(238, 238, 238))
    d.text((x, y + ch + 16), "mean %.3f  R/G %.3f  B/G %.3f  hue %+.1f" % (s["mean"], s["RG"], s["BG"], s["hue"]),
           fill=(214, 214, 214))
out = os.path.join(OUT, "REAL_INPUT.png")
canvas.save(out)
print("   written: %s" % os.path.relpath(out, ROOT))
