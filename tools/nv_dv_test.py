"""
The DV NightShot test bench: this round's inputs through the chain, in both codec branches, against the native targets.

**What is under test.**  REVIEW.md section 4.4 records that "the pipeline order the note draws" is only PARTIAL
because `Chroma degradation` sits in the *decoder* (`player`) rather than before the codec -- and the mechanism
implemented last round (`format.chroma_lag`) sits in the *encoder*.  So the model now has chroma damage on both sides
of the recorder and no measurement of whether that matters.  This bench renders the same input with DV **on** and
**off**, which is the cheapest way to see whether the recorder's position in the chain changes what the picture
looks like at all.

**Why both codec branches and not just DV.**  A TRV310E puts this picture on MiniDV, so DV is the honest branch --
but the model's green matrix is applied in the *camera*, and a DV recorder is 4:1:1 with a fixed bit budget.  If the
green output survives 4:1:1 essentially intact, then the "chroma damage before compression" question is about the
*analog* branch's colour-under loss and not about DV's subsampling.  That distinction is measurable and nobody has
measured it.

**Native resolution, and the resolution printed beside every number.**  The real references kept in the attachment
store are 691x520, which is why every earlier figure was taken at 533x400 -- REVIEW.md section 4.1.  The clip itself
is 1440x1080, so this bench takes its targets at that size and renders at the input's own resolution, and it prints
the size on every row.  A hue-spread figure without its scale is not a figure.

**Nothing is retuned here.**  This is a bench: it runs the schema's own defaults and reports what came out, so that
whatever is changed next is aimed at what the measurement shows rather than at what seems plausible.
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
OBJ = r"C:\Users\Administrator\.dsh\attachments\v1\objects"

#: The inputs, each labelled with what it is, because a photograph and a camcorder's own output are not the same kind
#: of thing and REVIEW.md section 6 forbids mixing them.
INPUTS = [
    (os.path.join(OUT, "ref_osd_1440.png"), "the camcorder's own 1440x1080 frame", 0.20),
    (os.path.join(OUT, "case_new_2048.png"), "the new case photograph (visible light)", 0.0),
    (os.path.join(OBJ, "e0", "e0e36b261fbbad42777702d3b775fcf30d4f8505fdc104e08b24405e69eda14c"),
     "the earlier case photograph, kept as a control", 0.0),
]

BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])

#: The optics that would blur or tint the picture and hide the effect under test, switched off so the bench measures
#: the night-vision and codec mechanisms rather than the lens.  This is the same list `tools/nv_deliver.py` uses.
QUIET_OPTICS = dict(halation=0.0, ccd_smear=0.0, flare_defocus=0.0, lens_flare=0.0, spike_spread=0.0,
                    overflow_drain=0.0, chroma_aberration=0.0, longitudinal_ca=0.0, axis_offset=0.0,
                    source_glow=0.0, shadow_depth=0.0, exposure=0.0, sensor_noise=0.0, noise_grain=0.0,
                    fixed_pattern=0.0, motion_blur=0.0)

#: What the analog branch's own bandwidth is, so the composite and the component branch differ in the way they
#: actually differ rather than in one extra switch.
FORMAT_ANALOG = {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0}
FORMAT_DV = {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0}


class Still:
    fps = 25.0

    def __init__(self, a: np.ndarray):
        self.a = a
        self.height, self.width = a.shape[:2]
        self.frame_count = 1

    def frame(self, i: int) -> np.ndarray:
        return self.a


class Sequence:
    """
    A moving version of one input, so the accumulating modes have something to accumulate.

    **This exists because the first run of this bench reported NIGHTSHOT 1 and NIGHTSHOT 2 as bit-identical** --
    same primary, same mean, same standard deviation, same hue in all four bands.  The mechanism was not missing:
    `nv_accumulate` sums *separate exposures* and Super NightShot's whole purpose is that their uncorrelated noise
    falls as `1/sqrt(N)`, so **with one frame as the input there is nothing to average and the two modes must come
    out the same.**  The bench was measuring its own input, which is the failure this repository keeps recording: a
    mechanism that looks inert because the rig gave it nothing to act on.

    The frames are small rigid displacements of the input, which is the cheapest way to make consecutive exposures
    differ.  The renderer is pumped to the last frame, as `tools/nv_deliver.py` does, so the accumulation window is
    full by the time the picture is read.
    """

    fps = 25.0

    def __init__(self, a: np.ndarray, n: int = 8):
        self.a = a
        self.height, self.width = a.shape[:2]
        self.frame_count = int(n)
        self._pics = [np.roll(np.roll(a, 2 * k, axis=1), k, axis=0) for k in range(int(n))]

    def frame(self, i: int) -> np.ndarray:
        return self._pics[int(i) % len(self._pics)]


def run(src: np.ndarray, cam: dict, dv: bool, frames: int = 1) -> np.ndarray:
    layers = {"camera": dict(QUIET_OPTICS, **cam),
              "format": dict(FORMAT_DV if dv else FORMAT_ANALOG),
              "player": dict(FORMAT_DV if dv else FORMAT_ANALOG),
              "codec": {"dv_enabled": 1.0 if dv else 0.0},
              "display": {"enabled": 0.0}}
    h, w = src.shape[:2]
    source = Still(src) if frames <= 1 else Sequence(src, frames)
    r = Renderer(source, w, h, 25.0, merge_values(BASE, layers), seed=20240101)
    return np.clip(np.asarray(r.render_frame(max(0, frames - 1)), np.float32), 0.0, 1.0)


#: The configurations.  `nv_mode` selects the function inside the camera; the rest of each mode's own values come
#: from the schema, because a test that supplies its own value cannot report on a default (the fault
#: `tools/nv_deliver.py` was written to fix).  `frames` is the length of the sequence handed to the renderer: the
#: accumulating modes need more than one, or their whole mechanism has nothing to sum.
CONFIGS = [
    ("no night vision", {}, 1, True),
    ("NIGHTSHOT 1", {"nv_mode": 1.0}, 1, True),
    ("NIGHTSHOT 2 (8 frames)", {"nv_mode": 2.0}, 8, True),
    ("NIGHTSHOT 1, DV OFF", {"nv_mode": 1.0}, 1, False),
    ("NIGHTSHOT 2, DV OFF (8f)", {"nv_mode": 2.0}, 8, False),
]

#: The targets, measured from the clip itself at native resolution by `tools/nv_targets.py`.
TARGETS = {
    "nv1": dict(g=0.4175, rg=0.5434, bg=0.8525, mean=0.2516, p1=0.0383, blown=0.00238,
                hue=(0.193, 0.147, 0.216, 0.302)),
    "nv2": dict(g=0.4356, rg=0.4766, bg=0.8192, mean=0.3332, p1=0.0406, blown=0.00593,
                hue=(0.322, 0.107, 0.158, 0.276)),
}

HEAD = ("   %-26s %-10s %-8s %-8s %-8s %-8s %-8s %-8s %-7s %-7s %s"
        % ("", "resolution", "primary", "mean", "std", "p1", "blown", "G share", "R/G", "B/G",
           "hue: dark  mid  bright  brightest"))
print("=" * 148)
print("  DV NIGHTSHOT -- this round's inputs through the chain, both codec branches")
print("=" * 148)
print(HEAD)


def row(tag: str, s: dict, note: str = "") -> None:
    print("   %-26s %-10s %-8s %-8.4f %-8.4f %-8.4f %-8.5f %-8.4f %-7.4f %-7.4f %s  %s"
          % (tag, s["res"], s["primary"], s["mean"], s["std"], s["p1"], s["blown"], s["g_share"],
             s["rg"], s["bg"], " ".join("%.3f" % v for v in s["spreads"]), note))


print("   --- the targets: what the camcorder actually produced, at 1440x1080 ---")
TGT = {}
for seg, t in TARGETS.items():
    TGT[seg] = t
    print("   %-26s %-10s %-8s %-8.4f %-8.4f %-8.4f %-8.5f %-8.4f %-7.4f %-7.4f %s"
          % ("the real " + seg.upper(), "1440x864", "-", t["mean"], float("nan"), t["p1"], t["blown"],
             t["g"], t["rg"], t["bg"], " ".join("%.3f" % v for v in t["hue"])))
print()

panels: list[tuple[str, np.ndarray]] = []
for path, tag, frac in INPUTS:
    if not os.path.exists(path):
        print("   MISSING, skipped: %s" % path)
        continue
    src = np.asarray(Image.open(path).convert("RGB"), np.float32) / 255.0
    src = crop_osd(src, frac) if frac else src
    print("   --- %s ---" % tag)
    row("  the input", measure(src))
    panels.append((tag + " (input)", src))
    for label, cam, frames, dv in CONFIGS:
        out = run(src, cam, dv, frames)
        s = measure(out)
        note = ""
        if cam.get("nv_mode") == 1.0:
            note = "G %.3f (target %.3f)   R/G %.3f (target %.3f)" % (
                s["g_share"], TARGETS["nv1"]["g"], s["rg"], TARGETS["nv1"]["rg"])
        elif cam.get("nv_mode") == 2.0:
            note = "G %.3f (target %.3f)   R/G %.3f (target %.3f)" % (
                s["g_share"], TARGETS["nv2"]["g"], s["rg"], TARGETS["nv2"]["rg"])
        row("  " + label, s, note)
        panels.append(("%s\n%s" % (tag.split(" (")[0][:22], label), out))
    print()

# --- the contact sheet, with each panel's own figures beside it -----------------------------------------------
if panels:
    cols = len(CONFIGS) + 1
    cw, ch, gap, cap = 300, 225, 6, 40
    rows = (len(panels) + cols - 1) // cols
    canvas = Image.new("RGB", ((cw + gap) * cols + gap, (ch + cap) * rows + 30), (16, 16, 18))
    d = ImageDraw.Draw(canvas)
    d.text((6, 6), "DV NIGHTSHOT -- this round's inputs through the chain.  Each panel carries its own figures.",
           fill=(242, 242, 242))
    for i, (tag, a) in enumerate(panels):
        x = gap + (i % cols) * (cw + gap)
        y = 30 + (i // cols) * (ch + cap)
        canvas.paste(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).resize((cw, ch), Image.LANCZOS), (x, y))
        s = measure(a)
        d.text((x, y + ch + 2), tag, fill=(238, 238, 238))
        d.text((x, y + ch + 15), "G %.3f  R/G %.3f  B/G %.3f  mean %.3f"
               % (s["g_share"], s["rg"], s["bg"], s["mean"]), fill=(206, 206, 212))
    out = os.path.join(OUT, "DV_NIGHTSHOT.png")
    canvas.save(out)
    print("   written: %s" % os.path.relpath(out, ROOT))
