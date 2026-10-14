"""
The new DV test clip as the input and the target: a second real sample, on the same scene.

**Why this comparison is the one that can settle the green.**  `10月13日.mp4` gave the model a target of
`G share 0.4199 / R-G 0.5402 / B-G 0.8416`, and the model came out at 0.3905 / 0.6664 / 0.8946 -- read as "too red and
not green enough".  **But a single real frame is a sample, not a standard**, and this clip is a second sample of the
*same scene* from the same camera: its green segments read

    this clip, its greenest segment   G 0.3950   R/G 0.6285   B/G 0.9029      (mean 0.3276, blown 3.96%)
    the 10月13 clip, NIGHTSHOT 1      G 0.4199   R/G 0.5402   B/G 0.8416      (mean 0.2394, blown 0.24%)

so **two real recordings of one scene differ from each other by as much as the model differs from either.** That does
not excuse the model -- but it does mean a single-frame difference is not yet evidence of a mechanism fault, and the
honest move is to measure the model against both and report the spread rather than tune to one.

**And it separates two things this work has been conflating.**  `R/G` moves from 0.540 to 0.629 between the two real
clips -- a 16% swing on the same scene -- which is the same direction and roughly the same size as the model's
overshoot.  `B/G` moves only 0.842 to 0.903.  So whatever is varying between recordings is mostly the **red** channel,
which is exactly what `REVIEW.md` section 4.2 recorded as unsettled (the "real knob is `u_gain`" note, last measured at
533x400 and never re-measured at native resolution).

Nothing is retuned here.  Both real clips are measured at their own size, the model is run on the new clip's own
frames with the schema's defaults, and every panel carries its figures.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from atsim.core.params import merge_values                                    # noqa: E402
from atsim.core.profiles import FACTORY_PRESETS, SCHEMA                       # noqa: E402
from atsim.render.renderer import Renderer                                    # noqa: E402
from nv_targets import crop_osd, measure                                      # noqa: E402

OUT = os.path.join(ROOT, "tools", "nightshot")
NEW = os.path.join(OUT, "sc1515")
OLD = os.path.join(OUT, "frames")
W, H = 720, 540
BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])


class Still:
    fps, frame_count = 25.0, 1

    def __init__(self, a):
        self.a = a
        self.height, self.width = a.shape[:2]

    def frame(self, i):
        return self.a


def load(path: str, frac: float = 0.20) -> np.ndarray:
    a = np.asarray(Image.open(path).convert("RGB"), np.float32) / 255.0
    a = crop_osd(a, frac) if frac else a
    return np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS),
                      np.float32) / 255.0


def render(src: np.ndarray, mode: float) -> np.ndarray:
    layers = {"camera": {"nv_mode": float(mode)},
              "format": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0},
              "player": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0},
              "codec": {"dv_enabled": 1.0}}
    r = Renderer(Still(src), W, H, 25.0, merge_values(BASE, layers), seed=20240101)
    return np.clip(np.asarray(r.render_frame(0), np.float32), 0.0, 1.0)


def main() -> int:
    new = {n: load(os.path.join(NEW, n + ".png")) for n in ("a_dark", "b_nv1", "c_nv2", "d_nv3")}
    old = {n: load(os.path.join(OLD, n + ".png")) for n in ("nv1_1", "nv2_1")}

    panels = [
        ("THE NEW CLIP, mode OFF\n(1-15s: no night vision)", new["a_dark"]),
        ("THE NEW CLIP, night vision on\n(16-27s)", new["b_nv1"]),
        ("THE NEW CLIP, night vision\n(28-37s)", new["c_nv2"]),
        ("THE NEW CLIP, its greenest\n(38-48s)  <- the comparison", new["d_nv3"]),
        ("THE 10月13 CLIP, NIGHTSHOT 1\n(the target used until now)", old["nv1_1"]),
        ("THE 10月13 CLIP, NIGHTSHOT 2", old["nv2_1"]),
        ("the model on the NEW clip\nmode OFF", render(new["d_nv3"], 0.0)),
        ("the model on the NEW clip\n** NIGHTSHOT 1, defaults untouched **", render(new["d_nv3"], 1.0)),
        ("the model on the NEW clip\nSUPER NIGHTSHOT", render(new["d_nv3"], 2.0)),
        ("the model on the 10月13 frame\nNIGHTSHOT 1 (for the same comparison)",
         render(old["nv1_1"], 1.0)),
    ]

    print("=" * 128)
    print("  TWO REAL SAMPLES OF ONE SCENE, AND THE MODEL AGAINST BOTH")
    print("=" * 128)
    print()
    print("   %-52s %-8s %-8s %-8s %-7s %-7s %-8s %s"
          % ("panel", "mean", "primary", "G share", "R/G", "B/G", "blown", "hue: dark  mid   bright brightest"))
    print("   " + "-" * 124)
    for tag, a in panels:
        s = measure(a)
        print("   %-52s %-8.4f %-8s %-8.4f %-7.4f %-7.4f %-8.5f %s"
              % (tag.split("\n")[0][:51], s["mean"], s["primary"], s["g_share"], s["rg"], s["bg"],
                 s["blown"], " ".join("%.3f" % v for v in s["spreads"])))
    print()

    # --- the spread between the two REAL clips, which is the yardstick the model should be judged by ----------
    rn, ro = measure(new["d_nv3"]), measure(old["nv1_1"])
    mn = measure(render(new["d_nv3"], 1.0))
    print("   the two REAL recordings of this scene differ by:")
    print("      G share  %.4f vs %.4f   (%.1f%%)" % (rn["g_share"], ro["g_share"],
                                                     100 * abs(rn["g_share"] / ro["g_share"] - 1)))
    print("      R/G      %.4f vs %.4f   (%.1f%%)" % (rn["rg"], ro["rg"], 100 * abs(rn["rg"] / ro["rg"] - 1)))
    print("      B/G      %.4f vs %.4f   (%.1f%%)" % (rn["bg"], ro["bg"], 100 * abs(rn["bg"] / ro["bg"] - 1)))
    print()
    print("   the model on the NEW clip against the NEW clip's own greenest frame:")
    print("      G share  %.4f vs %.4f   (%.1f%%)" % (mn["g_share"], rn["g_share"],
                                                     100 * abs(mn["g_share"] / rn["g_share"] - 1)))
    print("      R/G      %.4f vs %.4f   (%.1f%%)" % (mn["rg"], rn["rg"], 100 * abs(mn["rg"] / rn["rg"] - 1)))
    print("      B/G      %.4f vs %.4f   (%.1f%%)" % (mn["bg"], rn["bg"], 100 * abs(mn["bg"] / rn["bg"] - 1)))
    print()

    cols, cw, ch, gap, cap = 5, 340, 255, 8, 58
    rows = (len(panels) + cols - 1) // cols
    canvas = Image.new("RGB", ((cw + gap) * cols + gap, (ch + cap) * rows + 40), (14, 14, 16))
    d = ImageDraw.Draw(canvas)
    d.text((8, 6), "TWO REAL SAMPLES OF ONE SCENE, AND THE MODEL AGAINST BOTH -- the real panels come first",
           fill=(240, 240, 242))
    d.text((8, 21), "a single real frame is a sample, not a standard: the two recordings differ from each other by "
                    "as much as the model differs from either", fill=(220, 200, 150))
    for i, (tag, a) in enumerate(panels):
        x = gap + (i % cols) * (cw + gap)
        y = 40 + (i // cols) * (ch + cap)
        canvas.paste(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).resize((cw, ch), Image.LANCZOS),
                     (x, y))
        s = measure(a)
        lines = tag.split("\n")
        d.text((x, y + ch + 2), lines[0][:56], fill=(245, 245, 248))
        if len(lines) > 1:
            d.text((x, y + ch + 15), lines[1][:56], fill=(178, 178, 184))
        d.text((x, y + ch + 30), "mean %.3f  %s  G %.3f  R/G %.3f  B/G %.3f"
               % (s["mean"], s["primary"], s["g_share"], s["rg"], s["bg"]), fill=(212, 212, 218))
    path = os.path.join(OUT, "TWO_SAMPLES.png")
    canvas.save(path)
    print("   written: %s" % os.path.relpath(path, ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
