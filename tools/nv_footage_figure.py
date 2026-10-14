"""
The test figure on the clip's own frames: what the model makes of the footage it was calibrated against.

**Why this is a different measurement from the ones before it.**  Every bench in this work has switched OFF the terms
that would blur, tint or noise the picture -- `QUIET_OPTICS` in `nv_figure.py` sets `halation`, `lens_flare`,
`shadow_depth`, `sensor_noise` and a dozen others to zero -- and that is right for isolating a mechanism.  **It is
wrong for a figure meant to be looked at**, because a camcorder does not have those terms switched off: the model run
that way is not a model of the device, it is the device minus most of its optics.  The dark-band hue spread is the
term that shows the difference most plainly, and it was being read off the quiet rig.

**So here the schema's own defaults are used, and nothing is zeroed.**  The only choices made are the ones a user
makes: which mode, and whether the recorder is DV (a TRV310E records MiniDV, so it is).

**And the input is the camcorder's own frame**, which makes this the strictest form of the comparison available: the
model is being asked to reproduce a picture *through* the whole chain of a picture that already came out of one.  A
photograph of the scene would be the honest input in production; the footage is the honest input for asking whether
the chain lands where the device's own output does.
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


def render(src: np.ndarray, mode: float, dv: bool = True) -> np.ndarray:
    """
    The schema's own defaults, with only the mode and the recorder chosen.

    **Nothing is zeroed**, and that is the point of this file: a figure made on a quiet rig shows what the mechanism
    does, not what the device does.
    """
    layers = {"camera": {"nv_mode": float(mode)},
              "format": {"chroma_bandwidth": 2.0 if dv else 0.75, "luma_bandwidth": 5.0 if dv else 3.0},
              "player": {"chroma_bandwidth": 2.0 if dv else 0.75, "luma_bandwidth": 5.0 if dv else 3.0},
              "codec": {"dv_enabled": 1.0 if dv else 0.0}}
    r = Renderer(Still(src), W, H, 25.0, merge_values(BASE, layers), seed=20240101)
    return np.clip(np.asarray(r.render_frame(0), np.float32), 0.0, 1.0)


def main() -> int:
    #: **The footage as the input, and the reference as the target.**  Frame 1 is the one the display ratios were
    #: solved from; frames 0 and 2 are the same segment, so the spread between them is the shot's own variation.
    refs = [load(os.path.join(FRAMES, n), 0.20) for n in ("nv1_0.png", "nv1_1.png", "nv1_2.png")]
    ref2 = load(os.path.join(FRAMES, "nv2_1.png"), 0.20)
    case = load(CASE, 0.0)

    panels = [
        ("THE REFERENCE 1\n(the camcorder's own frame)", refs[0]),
        ("THE REFERENCE 2\n(the target ratios came from this)", refs[1]),
        ("THE REFERENCE 3\n(same segment, third frame)", refs[2]),
        ("THE REFERENCE 4\n(NIGHTSHOT 2: the brighter end)", ref2),
        ("the source photograph\n(a visible-light photo, for reference)", case),
        ("the model, mode OFF\nthe footage through the chain, no night vision",
         render(refs[1], 0.0)),
        ("the model, NIGHTSHOT 1\n** the footage through the chain, defaults untouched **",
         render(refs[1], 1.0)),
        ("the model, SUPER NIGHTSHOT\n(8 frames accumulated)", render(refs[1], 2.0)),
        ("the model, COLOUR SLOW SHUTTER\n(the mode that keeps colour)", render(refs[1], 3.0)),
        ("the model, NIGHTSHOT 1 on the PHOTOGRAPH\n(a real scene input, not the footage)",
         render(case, 1.0)),
    ]

    print("=" * 126)
    print("  THE FOOTAGE THROUGH THE CHAIN -- schema defaults, nothing zeroed")
    print("=" * 126)
    print()
    print("   %-52s %-8s %-8s %-8s %-7s %-7s %s"
          % ("panel", "mean", "primary", "G share", "R/G", "B/G", "hue: dark  mid   bright brightest"))
    print("   " + "-" * 122)
    for tag, a in panels:
        s = measure(a)
        print("   %-52s %-8.4f %-8s %-8.4f %-7.4f %-7.4f %s"
              % (tag.split("\n")[0][:51], s["mean"], s["primary"], s["g_share"], s["rg"], s["bg"],
                 " ".join("%.3f" % v for v in s["spreads"])))
    print()
    # the one comparison that matters most, stated in one place
    t = measure(refs[1])
    m = measure(render(refs[1], 1.0))
    print("   NIGHTSHOT 1 on the footage against the footage itself:")
    print("      mean      %.4f  vs  %.4f     (%+.1f%%)" % (m["mean"], t["mean"], 100.0 * (m["mean"] / t["mean"] - 1)))
    print("      R/G       %.4f  vs  %.4f" % (m["rg"], t["rg"]))
    print("      B/G       %.4f  vs  %.4f" % (m["bg"], t["bg"]))
    print("      dark hue  %.3f   vs  %.3f" % (m["spreads"][0], t["spreads"][0]))
    print()

    cols, cw, ch, gap, cap = 5, 340, 255, 8, 58
    rows = (len(panels) + cols - 1) // cols
    canvas = Image.new("RGB", ((cw + gap) * cols + gap, (ch + cap) * rows + 40), (14, 14, 16))
    d = ImageDraw.Draw(canvas)
    d.text((8, 6), "THE FOOTAGE THROUGH THE CHAIN -- the camcorder's own frames first, then the model on them",
           fill=(240, 240, 242))
    d.text((8, 21), "schema defaults, nothing zeroed: this is the device, not the device minus its optics",
           fill=(150, 215, 170))
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
        d.text((x, y + ch + 43), "hue dark %.3f  mid %.3f  bright %.3f"
               % (s["spreads"][0], s["spreads"][1], s["spreads"][2]), fill=(196, 196, 202))
    path = os.path.join(OUT, "FOOTAGE_THROUGH.png")
    canvas.save(path)
    print("   written: %s" % os.path.relpath(path, ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
