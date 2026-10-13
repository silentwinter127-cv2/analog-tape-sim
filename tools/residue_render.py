"""
Render the chroma residue's effect so the calibration can be judged by eye, beside the numbers it was fitted to.

**What is being shown, and why this comparison and not another.** The residue's purpose is that the mono display's
own R:G:B ratio should **follow the subject** rather than being fixed. So the render uses **the case photograph** --
the same machine the clip shows, and the same subject the three NightShot segments were shot of -- and the three
residue values are rendered through the whole chain.

**And the measured targets are printed beside the results**, because the question is not "does 0.35 look nice" but
"does the display move the way the three real segments did". The numbers to compare against:

    segment                        its red share, measured
    first mode / NightShot (23-45)          0.2892
    NIGHTSHOT 1 (54-68 s)                   0.2339
    NIGHTSHOT 2 (68-83 s)                   0.2011
    and their spread: a factor of 1.44

**The caveat is repeated here rather than buried**, because it decides how much the images can settle: the three
segments were all shot **in the mode**, so the colour source material they came from does not exist. **The residue is
therefore calibrated so the display follows the subject at all** -- it went from a fixed output ratio for every input to
a spread of 1.20 -- **and not to reproduce 1.44 exactly**, which would need the colour originals.

**Output**: a three-panel comparison, and each panel's own R:G:B ratio printed.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from atsim.core.params import merge_values                                    # noqa: E402
from atsim.core.profiles import FACTORY_PRESETS, SCHEMA                       # noqa: E402
from atsim.render.renderer import Renderer                                    # noqa: E402

OUT = os.path.join(ROOT, "tools", "calib")
os.makedirs(OUT, exist_ok=True)
COEF = np.array([0.299, 0.587, 0.114], np.float32)
PHOTO = (r"C:\Users\Administrator\.dsh\attachments\v1\objects\69"
         r"\699c23315df037b01fa831a273364b41f99c28733fb1dc928b22d1e6c281818d")

src = np.asarray(Image.open(PHOTO).convert("RGB"), np.float32) / 255.0
# a modest scale, because the point is the colour and not the resolution
H, W = 540, 720
img = Image.fromarray((np.clip(src, 0, 1) * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS)
src = np.asarray(img, np.float32) / 255.0


class One:
    height, width = H, W
    fps = 25.0
    frame_count = 1

    def frame(self, i):
        return src


BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])
#: the chain the earlier calibration used, so this is comparable with it: the defects off, tape off, plain display
CHAIN = {
    "camera": {"nv_mode": 1.0, "nv_illuminator": 0.15, "nv_green_mix": 1.0, "nv_accumulate": 1.0,
               "halation": 0.0, "ccd_smear": 0.0, "flare_defocus": 0.0, "lens_flare": 0.0,
               "overflow_drain": 0.0, "chroma_aberration": 0.0, "exposure": 0.0},
    "format": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0},
    "player": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0},
    "codec": {"dv_enabled": 1.0},
    "tape": {"head_switching": 0.0},
    "display": {"enabled": 1.0, "gamma": 1.0, "saturation": 0.62, "scanlines": 0.0, "mask_depth": 0.0},
}

print("   rendering at %dx%d, NightShot with the defects off so the colour is what is being compared" % (W, H))
print()
print("   %-10s %-10s %-24s %-12s %s" % ("residue", "mean", "R:G:B", "red share", "as a picture"))
panels = []
for r in (0.0, 0.35, 1.0):
    cam = dict(CHAIN["camera"], nv_chroma_residue=r)
    p = merge_values(BASE, dict(CHAIN, camera=cam))
    rr = Renderer(One(), W, H, 25.0, p, seed=20240101)
    a = np.clip(np.asarray(rr.render_frame(0), np.float32), 0, 1)
    m = a.reshape(-1, 3).mean(axis=0)
    share = m / max(1e-9, m.sum())
    panels.append((r, a, share))
    print("   %-10.2f %-10.4f %-24s %-12.4f %s"
          % (r, float((a @ COEF).mean()), np.round(share, 4), share[0],
             "the rigid projection" if r == 0 else ("the calibration" if r == 0.35 else "chroma untouched")))

print()
print("   === the measured targets this is judged against ===")
MEAS = [("first mode / NightShot (23-45 s)", 0.2892), ("NIGHTSHOT 1 (54-68 s)", 0.2339),
        ("NIGHTSHOT 2 (68-83 s)", 0.2011)]
for name, red in MEAS:
    print("      %-34s its red share was measured at %.4f" % (name, red))
print("      the three span a factor of %.3f" % (max(m[1] for m in MEAS) / min(m[1] for m in MEAS)))
print()
print("   the model's own spread across those three residues, on this one subject: %.4f to %.4f, a factor of %.3f"
      % (min(p[2][0] for p in panels), max(p[2][0] for p in panels),
         max(p[2][0] for p in panels) / max(1e-9, min(p[2][0] for p in panels))))
print("   (a single subject cannot show the spread the three segments show -- they are three different subjects,")
print("    so the comparable quantity is the sensitivity, which is what the fitted 0.35 sets.)")

# --- the comparison image
gap = 12
canvas = Image.new("RGB", (W * 3 + gap * 2, H + 34), (18, 18, 20))
for i, (r, a, share) in enumerate(panels):
    im = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
    canvas.paste(im, (i * (W + gap), 34))
    from PIL import ImageDraw
    d = ImageDraw.Draw(canvas)
    label = "residue %.2f   R:G:B %.4f:%.4f:%.4f" % (r, share[0], share[1], share[2])
    d.text((i * (W + gap) + 6, 10), label, fill=(230, 230, 230))
p = os.path.join(OUT, "residue_compare.png")
canvas.save(p)
print()
print("   written: %s" % os.path.relpath(p, ROOT))

# and the same three as separate files, for a closer look
for r, a, share in panels:
    q = os.path.join(OUT, "residue_%.2f.png" % r)
    Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).save(q)
print("   and: %s" % ", ".join(os.path.relpath(os.path.join(OUT, "residue_%.2f.png" % r), ROOT)
                                for r, _, _ in panels))
