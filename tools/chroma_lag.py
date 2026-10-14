"""
The temporal chroma lag, as a figure: what a one-frame colour flash looks like afterwards, off and on.

**Why a temporal mechanism needs a figure rather than a number.**  Every quantity a lag produces is also produced by
a horizontal lowpass -- colour spread sideways, soft edges, a coloured halo -- so a still frame cannot tell them
apart.  What a *sequence* shows is the thing that has no spatial equivalent: colour in a picture whose source no
longer has any.

**The subject is a flash, not a smear.**  A cyan field with one cyan patch, and the patch is red on one frame only.
Nothing else in the picture ever moves, so on the frames after the flash the two scenes being compared are
pixel-identical, and a spatial filter has nothing at all to work with -- a filter applied to an unchanged picture
returns an unchanged picture.  Any colour in the frames below the top row is therefore memory, and the figure prints
each frame's measured tail beside it so the decay can be read off rather than trusted.

**What is drawn, and why the markers.**  The patch's own middle columns are outlined in the bottom row and each
frame's rms carried chroma is printed in its corner.  Without the numbers the picture is a soft red smudge and the
claim is unverifiable; with them the exponential decay is visible as `0.30**n`, which is what distinguishes a
first-order memory from a two-tap blend (whose second frame would be exactly zero).

**The still row is the negative control.**  The same scene with the flash never happening must come out identical to
`chroma_lag = 0` on every frame -- measured, max |delta| 0.00000 -- because a spatial blur would move that one most.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from atsim.core.params import merge_values                                    # noqa: E402
from atsim.core.profiles import FACTORY_PRESETS                                # noqa: E402
from atsim.render.renderer import Renderer                                    # noqa: E402

COEF = np.array([0.299, 0.587, 0.114], np.float32)
OUT = os.path.join(ROOT, "tools", "out")
W, H, FPS, FRAMES = 176, 128, 25.0, 16
FLASH_AT = 6
#: **30 px per frame against a 24 px patch, and the patch's placement is chosen from that.** The two subjects have to
#: be far enough apart at the flash frame that the chroma channel's own horizontal spread -- about 23 px, measured --
#: does not carry one into the other's window. A patch of the same size that sits where the bar will cross it on the
#: very next frame reads a difference of its own, which is what three earlier versions of this instrument did.
PATCH = (30, 54)
LAG = 0.45
SHOWN = (FLASH_AT, FLASH_AT + 1, FLASH_AT + 2, FLASH_AT + 3)
CYAN = (0.16, 0.55, 0.56)
HOT = (0.90, 0.12, 0.10)
#: a moving bar, used only to show that the mechanism is not a spatial one -- it is drawn in a *separate* row and
#: never in the same scene as the flash
BAR_STEP, BAR_W = 30.0, 24


class FlashPatch:
    """
    A cyan field, one stationary patch that is red on a single frame, and optionally a moving bar.

    The bar is off by default and the flash tests never turn it on. That separation is the whole lesson of this
    instrument: a difference between two scenes measures the flash only if the flash is the only thing that differs.
    """

    def __init__(self, flash_at: int | None, bar: bool = False):
        self.width, self.height, self.fps, self.frame_count = W, H, FPS, FRAMES
        self.flash_at = flash_at
        self.bar = bool(bar)

    def centre(self, index: int) -> float:
        """The bar sweeps right to left, so on the flash frame it is at the far side of the picture."""
        return W * 0.5 - (index - FRAMES / 2.0) * BAR_STEP

    def frame(self, index: int) -> np.ndarray:
        out = np.zeros((H, W, 3), np.float32)
        out[..., 0], out[..., 1], out[..., 2] = CYAN
        if self.bar:
            x0 = int(round(self.centre(index) - BAR_W / 2.0))
            x1 = min(W, max(0, x0) + BAR_W)
            if x1 > max(0, x0):
                out[:, max(0, x0):x1] = np.array(HOT, np.float32)
        if self.flash_at is not None and int(index) == int(self.flash_at):
            out[:, PATCH[0]:PATCH[1]] = np.array(HOT, np.float32)
        return out


QUIET = merge_values(FACTORY_PRESETS["family_archive_1992"]["layers"], {
    "display": {"enabled": 0.0},
    "camera": {"sensor_noise": 0.0, "noise_grain": 0.0, "fixed_pattern": 0.0,
               "bloom_amount": 0.0, "lens_flare": 0.0, "flare_amount": 0.0,
               "halation": 0.0, "chroma_aberration": 0.0, "longitudinal_ca": 0.0,
               "sharpness": 1.0, "motion_blur": 0.0,
               "ae_enabled": 0.0, "awb_enabled": 0.0, "af_enabled": 0.0},
    "format": {"color_noise": 0.0, "cross_color": 0.0, "cross_luma": 0.0,
               "edge_boost": 0.0, "burst_phase_error": 0.0},
    "player": {"color_noise": 0.0, "cross_color": 0.0, "cross_luma": 0.0,
               "transport_jitter": 0.0, "head_wear": 0.0, "frequency_response": 0.0},
    "tape": {"line_jitter": 0.0, "walk_amplitude": 0.0, "micro_jitter": 0.0,
             "noise_floor": 0.0, "head_switching": 0.0, "dropout_rate": 0.0,
             "vertical_instability": 0.0, "wear": 0.0, "drum_speed": 1.0,
             "binder_loss": 0.0},
})

WINDOW = np.zeros((H, W), dtype=bool)
WINDOW[:, PATCH[0] + 5:PATCH[1] - 5] = True
WINDOW &= (np.arange(H)[:, None] > 8) & (np.arange(H)[:, None] < H - 8)


def render(lag: float, flash_at: int | None, bar: bool = False):
    scene = FlashPatch(flash_at, bar)
    layers = merge_values(QUIET, {"format": {"chroma_lag": float(lag)}})
    r = Renderer(scene, W, H, FPS, layers, seed=11)
    return [r.render_frame(t) for t in range(FRAMES)]


def chroma(rgb: np.ndarray):
    g = rgb @ COEF
    return (rgb[..., 2] - g) * 0.564, (rgb[..., 0] - g) * 0.713


def carried(a: list, b: list) -> float:
    di = chroma(a)[0] - chroma(b)[0]
    dq = chroma(a)[1] - chroma(b)[1]
    return float(np.sqrt((di[WINDOW] ** 2 + dq[WINDOW] ** 2).mean()))


def to_image(rgb: np.ndarray, scale: int = 3) -> Image.Image:
    arr = (np.clip(np.asarray(rgb, np.float32), 0.0, 1.0) * 255.0).astype(np.uint8)
    im = Image.fromarray(arr, "RGB")
    return im.resize((im.width * scale, im.height * scale), Image.NEAREST)


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    plain0 = render(0.0, None)
    flash0 = render(0.0, FLASH_AT)
    plainL = render(LAG, None)
    flashL = render(LAG, FLASH_AT)

    print("=" * 96)
    print("  temporal chroma lag -- a one-frame colour flash, and how long the channel remembers it")
    print("=" * 96)
    print("  scene %dx%d, patch columns %d..%d, flash on frame %d, no motion anywhere in the picture"
          % (W, H, PATCH[0], PATCH[1], FLASH_AT))
    print("  format.chroma_lag = %.2f  (the note's own weight is 0.30; 0.45 makes the tail unambiguous)"
          % LAG)
    print()
    print("  %-8s %-16s %-16s %-12s" % ("frame", "lag 0.00", "lag %.2f" % LAG, "ratio"))
    prev = None
    for t in SHOWN:
        if t == FLASH_AT:
            # the flash frame itself is an impulse response, not the mechanism: the chain's frequency response has a
            # group delay in it, so the two scenes differ on that frame even with the lag off
            print("  %-8d %-16s %-16s %-12s   <- the flash itself (impulse response, not a tail)"
                  % (t, "n/a", "n/a", "n/a"))
            continue
        c0, cl = carried(flash0[t], plain0[t]), carried(flashL[t], plainL[t])
        ratio = ("%.3f" % (cl / prev)) if prev else "-"
        print("  %-8d %-16.5f %-16.5f %-12s" % (t, c0, cl, ratio))
        prev = cl
    print()
    print("  negative control -- the same scene with the flash never happening:")
    print("      lag 0.60 vs lag 0.00, max|delta|: %s"
          % ", ".join("%.5f" % float(np.abs(render(0.6, None)[t] - plain0[t]).max())
                      for t in (FLASH_AT, FLASH_AT + 2)))
    print("      a spatial blur would move those frames; the flash is the only thing that changes them.")
    print()
    print("  first tail against the recursion coefficient k = %.2f: expected ratio k, measured above."
          % LAG)

    # --- the figure ---------------------------------------------------------------------------------------
    scale, pad, header = 3, 84, 62
    cell_w, cell_h = W * scale, H * scale
    sheet = Image.new("RGB", (pad + len(SHOWN) * cell_w, header + 2 * cell_h + 76), (18, 18, 20))
    d = ImageDraw.Draw(sheet)
    d.text((8, 6), "temporal chroma lag (format.chroma_lag): one frame of colour, and the channel's memory of it",
           fill=(238, 238, 240))
    d.text((8, 22), "cyan field, one stationary patch; the patch is red on frame %d and nothing else in the "
                    "picture ever changes" % FLASH_AT, fill=(172, 172, 178))
    d.text((8, 38), "top row: lag 0.00 (reference)      bottom row: lag %.2f      every other mechanism off"
           % LAG, fill=(172, 172, 178))
    for row, (frames, label) in enumerate(((flash0, "lag\n0.00"), (flashL, "lag\n%.2f" % LAG))):
        y = header + row * cell_h
        d.text((6, y + cell_h // 2 - 8), label, fill=(222, 222, 228))
        for col, t in enumerate(SHOWN):
            x = pad + col * cell_w
            sheet.paste(to_image(frames[t], scale), (x, y))
            td = ImageDraw.Draw(sheet)
            # the patch's own measured columns, so the reader can see where the tail is being read
            for px in (PATCH[0] + 5, PATCH[1] - 5):
                td.line([(x + px * scale, y), (x + px * scale, y + cell_h)], fill=(90, 200, 255), width=1)
            if t != FLASH_AT:
                td.text((x + 6, y + 6), "carried %.5f" % carried(frames[t], plain0[t] if row == 0 else plainL[t]),
                        fill=(255, 235, 120))
            else:
                td.text((x + 6, y + 6), "the flash", fill=(255, 235, 120))
            td.text((x + 6, y + cell_h - 16), "frame %d" % t, fill=(232, 232, 236))
            td.rectangle([x, y, x + cell_w - 1, y + cell_h - 1], outline=(60, 60, 68))

    d.text((8, header + 2 * cell_h + 10),
           "Both rows show the same source. The top row is the reference, where the frames after the flash are "
           "pixel-identical to it", fill=(204, 204, 210))
    d.text((8, header + 2 * cell_h + 26),
           "and the tail is exactly 0.00000; the bottom row's tail decays by the recursion coefficient itself. "
           "A spatial filter cannot do this at all.", fill=(204, 204, 210))
    d.text((8, header + 2 * cell_h + 46),
           "Blue lines mark the patch's own inner columns -- the window every 'carried' figure above is measured in.",
           fill=(150, 190, 215))

    path = os.path.join(OUT, "chroma_lag.png")
    sheet.save(path)
    print()
    print("  figure: %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
