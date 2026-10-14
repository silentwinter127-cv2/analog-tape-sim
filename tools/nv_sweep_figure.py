"""
One-dimensional sweeps on a REAL frame, output as a figure so the effect can be looked at.

**Why this replaces the previous round's table.**  The table said `chroma_aberration` was non-monotonic and left it
there; the user looked at the WebUI and said "this is completely different from the reference I gave you", which was
true and which no column of numbers in a terminal could have shown.  **A sweep that cannot be looked at cannot be
argued with**, so every row of this file is a panel.

**And the measurement that makes the difference a magnitude rather than a direction.**  The previous round reported
the *trend* (output minus input) and called the fringe insufficient.  The absolute figures say the opposite about the
same picture -- the real frames' own dark-band hue spread is **0.209-0.212** and the model's is **0.774 on the DV
branch**, a factor of 3.6.  Both are true, because they answer different questions:

  * "does the chain move the hue spread in the right direction?" -- the trend;
  * "is the result the right distance from a real camcorder's picture?" -- the absolute.

**The model was closer on the trend and three and a half times further out on the absolute**, and only the second can
be seen by a person looking at the two pictures side by side.  So both are printed, and the absolute leads.

**The subject is a real frame, not a synthetic scene**, and it is cropped to remove the camcorder's own overlay --
text is high-contrast white and would dominate every statistic it appears in.
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
FRAMES = os.path.join(OUT, "frames")
OBJ = r"C:\Users\Administrator\.dsh\attachments\v1\objects"
CASE = os.path.join(OBJ, "e0", "e0e36b261fbbad42777702d3b775fcf30d4f8505fdc104e08b24405e69eda14c")

W, H = 720, 540
REF = "nv1_1.png"

BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])
QUIET = dict(halation=0.0, ccd_smear=0.0, flare_defocus=0.0, lens_flare=0.0, spike_spread=0.0,
             overflow_drain=0.0, source_glow=0.0, shadow_depth=0.0, sensor_noise=0.0, noise_grain=0.0,
             fixed_pattern=0.0, motion_blur=0.0, sharpness=1.0)
RASTER = dict(enabled=1.0, scanlines=0.35, mask_depth=0.2, interlace_flicker=0.1, spot_sigma=0.25,
              defocus=0.08, vignette=0.15, bloom=0.15, gamma=1.0, saturation=1.0,
              contrast=0.55, brightness=0.5, crt_age=0.25)


class Still:
    fps, frame_count = 25.0, 1

    def __init__(self, a):
        self.a = a
        self.height, self.width = a.shape[:2]

    def frame(self, i):
        return self.a


def load(path: str, frac: float) -> np.ndarray:
    a = np.asarray(Image.open(path).convert("RGB"), np.float32) / 255.0
    a = crop_osd(a, frac) if frac else a
    return np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS),
                      np.float32) / 255.0


def render(src: np.ndarray, cam: dict, display: bool = True) -> np.ndarray:
    layers = {"camera": dict(QUIET, **cam),
              "format": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0},
              "player": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0},
              "codec": {"dv_enabled": 1.0},
              "display": dict(RASTER) if display else {"enabled": 0.0}}
    r = Renderer(Still(src), W, H, 25.0, merge_values(BASE, layers), seed=20240101)
    return np.clip(np.asarray(r.render_frame(0), np.float32), 0.0, 1.0)


def main() -> int:
    case = load(CASE, 0.0)
    refs = [load(os.path.join(FRAMES, n), 0.20) for n in ("nv1_0.png", "nv1_1.png", "nv1_2.png")]
    ref = refs[1]
    sm = measure(ref)
    print("=" * 128)
    print("  THE TARGET, measured on the camcorder's own frames at this size (720x540)")
    print("=" * 128)
    for n, r in zip(("nv1_0", "nv1_1", "nv1_2"), refs):
        s = measure(r)
        print("   %-8s mean %.4f  %s  G %.4f  R/G %.3f  B/G %.3f   hue %s"
              % (n, s["mean"], s["primary"], s["g_share"], s["rg"], s["bg"],
                 " ".join("%.3f" % v for v in s["spreads"])))
    print()

    #: The sweeps.  Each is a knob that could plausibly explain "the picture keeps the LEDs' colour", and the values
    #: step through the whole declared range rather than around the current default.
    SWEEPS = [
        ("nv_chroma_residue", [0.0, 0.15, 0.35, 0.65, 1.0]),
        ("nv_green_mix", [0.6, 0.8, 0.95, 1.0, 1.0]),
        ("nv_green_gain", [0.8, 1.0, 1.15, 1.3, 1.5]),
        ("nv_highlight_saturation", [0.0, 1.0, 2.0, 4.0, 8.0]),
        ("nv_exposure_trim", [-0.8, -0.45, -0.1, 0.2, 0.6]),
    ]

    rows = [("THE REAL FRAME\n(the target)", ref), ("the source photograph", case)]
    for key, values in SWEEPS:
        for v in values:
            tag = "%s = %.2f" % (key.replace("nv_", ""), v)
            rows.append((tag, render(case, {"nv_mode": 1.0, key: float(v)})))

    print("   %-28s %-8s %-8s %-8s %-7s %-7s %s"
          % ("panel", "mean", "primary", "G share", "R/G", "B/G", "hue: dark  mid   bright brightest"))
    print("   " + "-" * 124)
    for tag, a in rows:
        s = measure(a)
        print("   %-28s %-8.4f %-8s %-8.4f %-7.3f %-7.3f %s"
              % (tag.split("\n")[0][:27], s["mean"], s["primary"], s["g_share"], s["rg"], s["bg"],
                 " ".join("%.3f" % v for v in s["spreads"])))
    print()

    # --- the figure ---------------------------------------------------------------------------------------
    cols = 5
    cw, ch, gap, cap = 340, 255, 8, 54
    nrows = (len(rows) + cols - 1) // cols
    canvas = Image.new("RGB", ((cw + gap) * cols + gap, (ch + cap) * nrows + 36), (14, 14, 16))
    d = ImageDraw.Draw(canvas)
    d.text((8, 6), "ONE KNOB AT A TIME, on a real frame -- the first panel is the target, the second is the input",
           fill=(240, 240, 242))
    d.text((8, 20), "the target's own dark-band hue spread is 0.209-0.212; the model's default is 0.774 -- 3.6x",
           fill=(150, 190, 215))
    for i, (tag, a) in enumerate(rows):
        x = gap + (i % cols) * (cw + gap)
        y = 36 + (i // cols) * (ch + cap)
        canvas.paste(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).resize((cw, ch), Image.LANCZOS), (x, y))
        s = measure(a)
        lines = tag.split("\n")
        d.text((x, y + ch + 2), lines[0][:56], fill=(245, 245, 248))
        if len(lines) > 1:
            d.text((x, y + ch + 14), lines[1][:56], fill=(176, 176, 182))
        d.text((x, y + ch + 28), "mean %.3f %s G %.3f R/G %.3f  hue %.3f"
               % (s["mean"], s["primary"], s["g_share"], s["rg"], s["spreads"][0]), fill=(210, 210, 216))
    path = os.path.join(OUT, "SWEEP_FIGURE.png")
    canvas.save(path)
    print("   written: %s" % os.path.relpath(path, ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
