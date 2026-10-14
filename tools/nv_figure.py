"""
The DV NightShot figure, after the physical correction.

**What changed, in one paragraph.**  NightShot takes the IR-cut filter out and then routes the sensor's output as **one
luminance** onto a green display.  The model had that step followed by a term that added the *scene's own visible
chroma* back (`out = mono + (rgb - lum) * r`), which is a colour grade's operation and not a display's: measured on
eight flat fields, the output's R:G:B ratio moved by **21.7x** depending on what it was shown, where a display must
give one ratio.  That term is gone, the display's primary ratios are **solved** from the footage instead of fitted, and
the green primary's gain is 1.0 because that is the only value at which the solve is exact (two free ratios, two
targets -- a third ratio change has nowhere to go but onto the targets).

**So the panels below are the same chain as before with one physically-wrong term removed**, and the target is the
camcorder's own frames at the size they were shot.
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
BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])
QUIET = dict(halation=0.0, ccd_smear=0.0, flare_defocus=0.0, lens_flare=0.0, spike_spread=0.0,
             overflow_drain=0.0, source_glow=0.0, shadow_depth=0.0, sensor_noise=0.0, noise_grain=0.0,
             fixed_pattern=0.0, motion_blur=0.0, sharpness=1.0)
#: **The display stage's configuration, and why this one.**  `broadcast_1992_weak` -- the preset these figures
#: calibrate against -- ships with `display.enabled = 0`, so the branch is the camera, the recorder and the deck with
#: no second display on top.  That matters here because **the reference frames are already a camcorder's finished
#: output**: the LCD and the recording chain did their work before the file exists, so running a 1990s CRT over it
#: would be modelling a monitor that is not in the picture.
#:
#: Measured, the difference is not subtle: with the stage off the model's mean is **0.3324** and with
#: `gamma = 2.35` -- the schema's own default, and its minimum is 1.5 so it cannot be asked for less -- it is
#: **0.5179**, more than double the footage's 0.2394.  The `enabled = 1` row below is those defaults deliberately
#: applied, so the cost of stacking the two displays is visible rather than described.
DISPLAY_OFF = {"enabled": 0.0}
#: `enabled = 1` with the schema's other defaults left alone -- i.e. what a project that turns the CRT on gets.
DISPLAY_ON = {"enabled": 1.0, "gamma": 2.35, "saturation": 0.5, "contrast": 0.5, "brightness": 0.5,
              "color_temp": 0.5, "scanlines": 0.35, "mask_depth": 0.35, "crt_age": 0.30,
              "interlace_flicker": 0.0, "spot_sigma": 0.0, "defocus": 0.0, "vignette": 0.0, "bloom": 0.0,
              "geometry_warp": 0.0, "overscan": 0.0, "curvature": 0.0, "convergence": 0.0, "reflection": 0.0}


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


def render(src: np.ndarray, cam: dict, dv: bool = True, display=None) -> np.ndarray:
    layers = {"camera": dict(QUIET, **cam),
              "format": {"chroma_bandwidth": 2.0 if dv else 0.75, "luma_bandwidth": 5.0 if dv else 3.0},
              "player": {"chroma_bandwidth": 2.0 if dv else 0.75, "luma_bandwidth": 5.0 if dv else 3.0},
              "codec": {"dv_enabled": 1.0 if dv else 0.0},
              "display": dict(DISPLAY_OFF if display is None else display)}
    r = Renderer(Still(src), W, H, 25.0, merge_values(BASE, layers), seed=20240101)
    return np.clip(np.asarray(r.render_frame(0), np.float32), 0.0, 1.0)


def main() -> int:
    refs = [load(os.path.join(FRAMES, n), 0.20) for n in ("nv1_0.png", "nv1_1.png", "nv1_2.png")]
    ref2 = load(os.path.join(FRAMES, "nv2_1.png"), 0.20)
    case = load(CASE, 0.0)

    panels = [
        ("THE REFERENCE 1\n(the camcorder's own frame)", refs[0]),
        ("THE REFERENCE 2\n(the one the ratios came from)", refs[1]),
        ("THE REFERENCE 3\n(another frame, same segment)", refs[2]),
        ("THE REFERENCE 4\n(NIGHTSHOT 2: brighter end)", ref2),
        ("the source photograph\n(visible light, colour LEDs)", case),
        ("OLD behaviour\nmono + (rgb-lum)*0.15  <- the grade", None),
        ("the model, NIGHTSHOT 1\n** the display is a display **", render(case, {"nv_mode": 1.0})),
        ("the model, NIGHTSHOT 2\nDV + display", render(case, {"nv_mode": 2.0})),
        ("the model, NIGHTSHOT 1\n**with the CRT stage ON** (cost of two displays)", render(case, {"nv_mode": 1.0}, display=DISPLAY_ON)),
        ("the model on the footage\n(NIGHTSHOT 1 fed its own frame)", render(refs[1], {"nv_mode": 1.0})),
    ]

    print("=" * 124)
    print("  DV NIGHTSHOT -- after removing the scene's chroma from the mono conversion")
    print("=" * 124)
    print()
    print("   %-44s %-8s %-8s %-8s %-7s %-7s %s"
          % ("panel", "mean", "primary", "G share", "R/G", "B/G", "hue: dark  mid   bright brightest"))
    print("   " + "-" * 120)
    for tag, a in panels:
        if a is None:
            continue
        s = measure(a)
        print("   %-44s %-8.4f %-8s %-8.4f %-7.4f %-7.4f %s"
              % (tag.split("\n")[0][:43], s["mean"], s["primary"], s["g_share"], s["rg"], s["bg"],
                 " ".join("%.3f" % v for v in s["spreads"])))
    print()

    cols, cw, ch, gap, cap = 5, 340, 255, 8, 56
    rows = (len(panels) + cols - 1) // cols
    canvas = Image.new("RGB", ((cw + gap) * cols + gap, (ch + cap) * rows + 38), (14, 14, 16))
    d = ImageDraw.Draw(canvas)
    d.text((8, 6), "DV NIGHTSHOT -- the camcorder's own frames first, then the model after the physical correction",
           fill=(240, 240, 242))
    d.text((8, 21), "the display's R:G:B is now one ratio for EVERY input (measured on 8 flat fields: 1.000x spread, "
                    "was 21.7x)", fill=(150, 215, 170))
    for i, (tag, a) in enumerate(panels):
        x = gap + (i % cols) * (cw + gap)
        y = 38 + (i // cols) * (ch + cap)
        if a is not None:
            canvas.paste(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).resize((cw, ch), Image.LANCZOS),
                         (x, y))
        else:
            d.rectangle([x, y, x + cw - 1, y + ch - 1], fill=(30, 26, 26))
            d.text((x + 8, y + ch // 2 - 8), "removed", fill=(210, 130, 130))
        lines = tag.split("\n")
        d.text((x, y + ch + 2), lines[0][:56], fill=(245, 245, 248))
        if len(lines) > 1:
            d.text((x, y + ch + 15), lines[1][:56], fill=(178, 178, 184))
        if a is not None:
            s = measure(a)
            d.text((x, y + ch + 30), "mean %.3f %s  R/G %.3f  B/G %.3f"
                   % (s["mean"], s["primary"], s["rg"], s["bg"]), fill=(212, 212, 218))
    path = os.path.join(OUT, "DV_NIGHTSHOT_FIXED.png")
    canvas.save(path)
    print("   written: %s" % os.path.relpath(path, ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

