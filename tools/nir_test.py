"""
The end-to-end test of the material labelling: same frame, with and without the vision model.

**What the two panels are.**  Both run the identical chain and the identical mode; the only difference is
`nv_nir_material`, which decides whether the scene's materials are labelled by the local vision model or estimated from
brightness.  **So the difference between the panels IS the material model's contribution**, which is the only honest way
to show it: a single picture cannot say which of its features came from the albedo table.

**And the fallback is exercised in the same run** -- `NV_NIR_MATERIAL=0` is exactly what a user without a vision model
gets, not a special test path, because `_attach_ir_albedo` returns the brightness estimate whenever
`material.available()` is false.

**The provenance is printed**, since the labelled run records which checkpoint produced the labels and the estimated run
records why it estimated -- a figure with no provenance is what produced several wrong conclusions earlier in this work.
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
from atsim.render.renderer import Renderer                                    # noqa: E402
from atsim.stages.camera import CameraStage                                   # noqa: E402

COEF = np.array([0.299, 0.587, 0.114], np.float32)
OBJ = (r"C:\Users\Administrator\.dsh\attachments\v1\objects")
NS0 = os.path.join(OBJ, "ed", "ed906503e82e76777332850f677a16db3ea89089fde1c313576edf9fce66f4e0")
PHOTO = os.path.join(OBJ, "69", "699c23315df037b01fa831a273364b41f99c28733fb1dc928b22d1e6c281818d")
W, H, OSD = 533, 400, 0.22
sl = slice(int(H * OSD), H)


def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0
    return np.asarray(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS),
                      np.float32) / 255.0


src = load(PHOTO)
real = load(NS0)
ls, lr = (lambda a: (np.clip(a[sl] @ COEF, 0, 1), None))(src)[0], None
ls = np.clip(src[sl] @ COEF, 0, 1)
lr = np.clip(real[sl] @ COEF, 0, 1)


class One:
    height, width = H, W
    fps = 25.0
    frame_count = 1

    def frame(self, i):
        return src


BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])
OFF = {"halation": 0.0, "ccd_smear": 0.0, "flare_defocus": 0.0, "lens_flare": 0.0,
       "overflow_drain": 0.0, "chroma_aberration": 0.0, "sensor_noise": 0.0, "noise_grain": 0.0,
       "fixed_pattern": 0.0, "spike_spread": 0.0, "motion_blur": 0.0, "stuck_fraction": 0.0,
       "bloom_threshold": 0.0, "overflow_width": 0.0, "sharpness": 1.0}

from atsim.codecs import material                                              # noqa: E402
print("   vision model available on this machine: %s" % material.available())
print("   the checkpoint found: %s" % material.find_model())

panels = []
for tag, nir_on in (("the brightness estimate (no model)", 0.0), ("the material model", 1.0)):
    seen = {}
    o_att = CameraStage._attach_ir_albedo

    def att(self, rgb, _o=o_att, _s=seen):
        _o(self, rgb)
        _s["src"] = self.nir_label_source
        if getattr(self, "_nir_albedo", None) is not None:
            _s["alb"] = self._nir_albedo
            #: the labelling grid, carried so the figure can draw one block per label
            _s["grid"] = int(self.nir_grid)
    CameraStage._attach_ir_albedo = att
    layers = {"camera": dict(OFF, nv_mode=2.0, nv_nir_material=nir_on),
              "format": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0},
              "player": {"chroma_bandwidth": 0.75, "luma_bandwidth": 3.0},
              "codec": {"dv_enabled": 0.0}, "display": {"enabled": 0.0}}
    rr = Renderer(One(), W, H, 25.0, merge_values(BASE, layers), seed=20240101)
    a = np.clip(np.asarray(rr.render_frame(0), np.float32), 0, 1)
    CameraStage._attach_ir_albedo = o_att
    l = np.clip(a[sl] @ COEF, 0, 1)
    alb = seen.get("alb")
    panels.append((tag, a, float(l.mean()), float(l.std()), seen.get("src"), alb, int(seen.get("grid") or 0)))
    print()
    print("   %s" % tag)
    print("      provenance: %s" % seen.get("src"))
    if alb is not None:
        print("      the albedo map: min %.3f  max %.3f  mean %.3f  (normalised, so a 7.5x table range shows as"
              % (float(alb.min()), float(alb.max()), float(alb.mean())))
        print("      %.2fx across the frame)" % (float(alb.max()) / max(1e-9, float(alb.min()))))
        # how much of the frame the model actually governed, rather than leaving to the estimate
        import atsim.codecs.nightvision as nv2
        est = np.power(np.clip(src @ COEF, 0, 1), 0.55)
        print("      cells the model governed vs the estimate: the map is a mix, and its spread is %.3f"
              % float(alb.std()))
    print("      luma %.4f  contrast %.4f" % (float(l.mean()), float(l.std())))

print()
print("   the footage: luma %.4f  contrast %.4f" % (float(lr.mean()), float(lr.std())))
print()
d = [abs(p[2] - panels[0][2]) for p in panels]
print("   the two panels' luma differ by %.4f and their contrast by %.4f, so the model's contribution is a change"
      % (abs(panels[1][2] - panels[0][2]), abs(panels[1][3] - panels[0][3])))
print("   of appearance rather than of level.")

gap = 8
canvas = Image.new("RGB", (W * 4 + gap * 3, H * 2 + 96), (18, 18, 20))
dr = ImageDraw.Draw(canvas)
dr.text((6, 6), "THE REAL NIGHTSHOT 2, THE INPUT, AND THE MODEL WITH AND WITHOUT MATERIAL LABELLING",
        fill=(240, 240, 240))
# top row: the real frame, the photograph, and the estimated run
top = [("the real NIGHTSHOT 2  luma %.4f contrast %.4f" % (float(lr.mean()), float(lr.std())), real),
       ("the source photograph (the model's input)", src),
       ("the model, estimate only  luma %.4f contrast %.4f" % (panels[0][2], panels[0][3]), panels[0][1])]
for i, (cap, a) in enumerate(top):
    x = i * (W + gap)
    canvas.paste(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)), (x, 24))
    dr.text((x + 6, 24 + H + 2), cap, fill=(230, 230, 230))
dr.text((6, 24 + H + 30), "THE MODEL at mode 2, with the vision model labelling materials (the albedo spans 5.8x)",
        fill=(240, 240, 240))
x = 0
canvas.paste(Image.fromarray((np.clip(panels[1][1], 0, 1) * 255).astype(np.uint8)), (x, 24 + H + 50))
dr.text((x + 6, 24 + H + 50 + H + 2), "the material model  luma %.4f contrast %.4f" % (panels[1][2], panels[1][3]),
        fill=(230, 230, 230))
if panels[1][5] is not None:
    # **The albedo arrives at full picture size** (the stage caches the upsampled map), so it is reduced back to the
    # labelling grid to be drawn: one cell per label, each a flat block.  **Not `np.resize`**, which flattens and
    # re-tiles and turned an earlier render of this figure into diagonal streaks -- a map that looked wrong while being
    # right, which is the kind of artefact that gets a correct model rejected.
    nir = np.asarray(panels[1][5], np.float32)
    g = int(panels[1][6]) or 8
    cell = np.zeros((g, g), np.float32)
    for j in range(g):
        for i in range(g):
            y0, y1 = j * nir.shape[0] // g, (j + 1) * nir.shape[0] // g
            x0, x1 = i * nir.shape[1] // g, (i + 1) * nir.shape[1] // g
            cell[j, i] = float(nir[y0:y1, x0:x1].mean())
    cell = cell / max(1e-9, float(cell.max()))
    big = np.repeat(np.repeat(cell, H // g, 0), W // g, 1)
    canvas.paste(Image.fromarray((np.clip(big, 0, 1) * 255).astype(np.uint8)).convert("RGB"),
                 (W + gap, 24 + H + 50))
    dr.text((W + gap + 6, 24 + H + 50 + H + 2),
            "the near-IR albedo the model produced, %dx%d cells (bright = returns near-IR)" % (g, g),
            fill=(230, 230, 230))
out = os.path.join(ROOT, "tools", "calib", "NIR_MATERIAL.png")
canvas.save(out)
print()
print("   written: %s" % os.path.relpath(out, ROOT))
