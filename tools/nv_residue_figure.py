"""
The chroma residue, which is the lever that adds colour without adding green.

The user's two reports -- "太绿了" and "还是没有彩色" -- look like opposites and are not.  They are two coordinates, and
the parameters move them differently:

    nv_green_gain up      chroma up, R/G up      -> MORE colour, but MORE green
    nv_chroma_residue up  chroma up, R/G DOWN    -> more colour, LESS green

so reaching for the display drive, as the previous round did, is the one that cannot satisfy either.  Against the real
NIGHTSHOT 2 frame:

    the real frame (c_0147)   chroma 0.10128   R/G 0.4739   B/G 0.8123
    residue 0.00              chroma 0.09839   R/G 0.5172   B/G 0.7622
    residue 0.15              chroma 0.10583   R/G 0.4850   B/G 0.7565   <- set
    residue 0.30              chroma 0.11306   R/G 0.4545   B/G 0.7511

**And the mechanism was recorded as inert**, with the reason in its own comment:
`out = mono + (rgb - lum) * r` carries the display's own weights, so the input's colour is pulled onto the very ratios
the monochrome path already produces and the two cannot be separated -- which is why `nv_chroma_residue` at 0.00, 0.35 and
0.95 all arrived as 0.350 and gave identical frames.  Added **outside** those ratios it does what it is named for, and
that is what `note.txt` asks for as its first proposal.
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

OUT = os.path.join(ROOT, "tools", "nightshot")
W, H = 720, 540
BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])
LUMA = np.array([0.299, 0.587, 0.114], np.float32)


class Still:
    fps, frame_count = 25.0, 1

    def __init__(self, a):
        self.a = a
        self.height, self.width = a.shape[:2]

    def frame(self, i):
        return self.a


def load(p, frac=0.0, size=(W, H)):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0
    if frac:
        a = a[int(a.shape[0] * frac):]
    return np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize(size, Image.LANCZOS),
                      np.float32) / 255.0


def run(src, residue=None, gain=None):
    cam = {"nv_mode": 2.0}
    if residue is not None:
        cam["nv_chroma_residue"] = residue
    if gain is not None:
        cam["nv_green_gain"] = gain
    lay = {"camera": cam, "format": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0},
           "player": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0},
           "codec": {"dv_enabled": 1.0}}
    r = Renderer(Still(src), W, H, 25.0, merge_values(BASE, lay), seed=20240101)
    return np.clip(np.asarray(r.render_frame(0), np.float32), 0, 1)


def stat(x):
    a = np.asarray(x, np.float32)
    l = np.clip(a @ LUMA, 0, 1)
    r, g, b = (float(a[..., i].mean()) for i in range(3))
    return dict(ch=float(np.abs(a - (a @ LUMA)[..., None]).mean()), rg=r / max(1e-9, g),
                bg=b / max(1e-9, g), mean=float(l.mean()), blown=100 * float((l > 0.90).mean()))


def main() -> int:
    real = load(os.path.join(OUT, "ref_c0147.png"))
    dv = load(os.path.join(OUT, "sc1515", "d_nv3.png"), 0.20)

    panels = [
        ("THE REAL NIGHTSHOT 2 (c_0147)\nnative 1440x1080", load(os.path.join(OUT, "ref_c0147.png"), 0.0,
                                                                (W, int(W * 1080 / 1440)))),
        ("the model, residue 0.00\n(what you saw as 太绿)", run(dv, 0.0)),
        ("the model, ** residue 0.15 **\nthe value now", run(dv, None)),
        ("the model, residue 0.30\n(past it: R/G below the frame's)", run(dv, 0.30)),
        ("the model, residue 0.15 + drive 1.5\n(the wrong lever: more colour AND more green)", run(dv, None, 1.5)),
        ("the model on the user's own frame\nresidue 0.15", run(real, None)),
    ]
    sr = stat(real)
    print("=" * 120)
    print("  COLOUR WITHOUT GREEN -- the chroma residue, added outside the display's ratios")
    print("=" * 120)
    print()
    print("   %-50s %-9s %-9s %-9s %-9s %-8s" % ("panel", "chroma", "R/G", "B/G", "mean", ">0.90"))
    print("   " + "-" * 116)
    for tag, img in panels:
        s = stat(img)
        print("   %-50s %-9.5f %-9.4f %-9.4f %-9.4f %-8.2f"
              % (tag.split(chr(10))[0][:49], s["ch"], s["rg"], s["bg"], s["mean"], s["blown"]))
    print()
    sm = stat(run(dv, None))
    print("   against the real frame, the three quantities:")
    print("     chroma  %.5f vs %.5f   -> %.0f%% of the frame's" % (sm["ch"], sr["ch"], 100 * sm["ch"] / sr["ch"]))
    print("     R/G     %.4f vs %.4f   -> off by %.4f" % (sm["rg"], sr["rg"], abs(sm["rg"] - sr["rg"])))
    print("     B/G     %.4f vs %.4f   -> off by %.4f" % (sm["bg"], sr["bg"], abs(sm["bg"] - sr["bg"])))
    print("     clipping %.2f%% vs %.2f%%  -> the remaining disagreement" % (sm["blown"], sr["blown"]))
    print()

    cols, cw, ch, gap, cap = 3, 430, 322, 10, 74
    rows = (len(panels) + cols - 1) // cols
    canvas = Image.new("RGB", ((cw + gap) * cols + gap, (ch + cap) * rows + 40), (14, 14, 16))
    d = ImageDraw.Draw(canvas)
    d.text((10, 6), "COLOUR WITHOUT GREEN -- the input's chroma residue added OUTSIDE the display's ratios: chroma up, "
                    "R/G down", fill=(240, 240, 242))
    d.text((10, 22), "chroma 0.0984 -> 0.1058 against the frame's 0.1013; R/G 0.517 -> 0.485 against 0.474",
           fill=(220, 200, 150))
    for i, (tag, img) in enumerate(panels):
        x = gap + (i % cols) * (cw + gap)
        y = 42 + (i // cols) * (ch + cap)
        canvas.paste(Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).resize((cw, ch), Image.LANCZOS),
                     (x, y))
        s = stat(img)
        for k, line in enumerate(tag.split("\n")):
            d.text((x, y + ch + 3 + 13 * k), line[:76], fill=(245, 245, 248) if k == 0 else (176, 176, 182))
        d.text((x, y + ch + 32), "chroma %.5f   R/G %.4f   B/G %.4f" % (s["ch"], s["rg"], s["bg"]),
               fill=(212, 212, 218))
        d.text((x, y + ch + 45), "mean %.4f   >0.90 %.2f%%" % (s["mean"], s["blown"]), fill=(212, 212, 218))
    path = os.path.join(OUT, "NS2_RESIDUE.png")
    canvas.save(path)
    print("   written: %s" % os.path.relpath(path, ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
