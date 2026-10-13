"""
The locked-geometry sweep, run cleanly -- the one measurement the control says is valid.

**The design, fixed by the control.** The detector must be **given** the geometry, not asked to find it: asking it to
locate the pole on a frame the mechanisms have altered makes the measurement follow the thing being measured, which is
the fault behind both failed instruments. So the columns and the surround offset come from the source **once** -- where
the pole is unambiguous at **columns 270..275** -- and the detector's only remaining role is a **visibility check**:
does the band still clear its surround on this frame.

**And the control proved the pieces this depends on**: the unconstrained detector is fooled by a bright edge (it picked
330..335 instead of 200..205 on a synthetic frame), the constrained one is not, and neither invents a pole on a bare
wall. That is why the geometry is supplied rather than searched for.

**What the sweep answers**, with the source's own figure as the target:

    the source, at columns 270..275      excess +0.0363   (pole +0.0062, surround -0.0388)

Each mechanism is varied with the others at the reproduction's values, and two things are read: the **excess** -- the
quantity the reference defines -- and the **visible row count**, because a mechanism that buries the pole shows up as
that count collapsing rather than as a number going the wrong way.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from atsim.core.params import merge_values                                   # noqa: E402
from atsim.core.profiles import FACTORY_PRESETS, SCHEMA                      # noqa: E402
from atsim.render.renderer import Renderer                                   # noqa: E402

OUT = os.path.join(ROOT, "tools", "ntscrs_ref")
COEF = np.array([0.299, 0.587, 0.114], np.float32)
W, H = 452, 382
LEFT, RIGHT, OFF = 270, 275, 24

src = np.asarray(Image.open(os.path.join(OUT, "refscene.png")).convert("RGB"),
                 np.float32) / 255.0
PIC = np.asarray(Image.fromarray((src * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR),
                 np.float32) / 255.0


class Still:
    def __init__(self):
        self.width, self.height, self.fps, self.frame_count = W, H, 25.0, 3

    def frame(self, index):
        return PIC


BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])
RASTER_OFF = {"scanlines": 0.0, "mask_depth": 0.0, "interlace_flicker": 0.0, "scan_pitch": 0.0,
              "spot_sigma": 0.0, "defocus": 0.0, "vignette": 0.0, "reflection": 0.0,
              "bloom": 0.0, "geometry_warp": 0.0, "overscan": 0.0, "curvature": 0.0,
              "breathing": 0.0, "crt_age": 0.0, "convergence": 0.0, "crt_moisture": 0.0,
              "crt_corrosion": 0.0}
CHOSEN = dict(halation=0.25, ccd_smear=0.20, flare_defocus=0.35, lens_flare=0.30,
              spike_spread=0.40, overflow_drain=0.60, chroma_aberration=0.6,
              longitudinal_ca=0.4, axis_offset=0.5, halation_source_colour=0.0,
              shadow_depth=0.0, exposure=0.0, color_cast=0.0, lens_fog=0.0, lens_wear=0.0,
              bloom_amount=0.0, dead_sites=0.0)


def run(camera, sat=0.62):
    layers = {"camera": dict(camera), "format": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0},
              "player": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0},
              "codec": {"dv_enabled": 1.0}, "tape": {"head_switching": 0.0},
              "display": dict(RASTER_OFF, enabled=1.0, gamma=1.0, saturation=sat)}
    p = merge_values(BASE, layers)
    r = Renderer(Still(), W, H, 25.0, p, seed=20240101)
    return np.clip(np.asarray(r.render_frame(0), np.float32), 0, 1)


def fixed_measure(img, left=LEFT, right=RIGHT, off=OFF, min_contrast=0.02):
    """
    Measure at **fixed columns given from outside**, and report whether the band still clears its surround.

    A frame where it does not is reported as having too few visible rows rather than being measured elsewhere -- which
    is the property both earlier instruments lacked.
    """
    lum = img @ COEF
    Hh, Ww = img.shape[:2]
    k = max(5, Hh // 40)
    box = np.lib.stride_tricks.sliding_window_view(lum, (k, k)).mean(axis=(2, 3))
    cy, _ = np.unravel_index(int(np.argmax(box)), box.shape)
    cy += k // 2
    rows = np.arange(min(Hh - 2, cy + 8), Hh)
    c0, c1 = max(0, left - off), max(1, left - 3)
    d0, d1 = min(Ww - 1, right + 3), min(Ww, right + off)
    pole_lum = lum[rows][:, left:right + 1].mean(axis=1)
    sur_lum = np.concatenate([lum[rows][:, c0:c1], lum[rows][:, d0:d1]], axis=1).mean(axis=1)
    keep = pole_lum > sur_lum + min_contrast
    n_ok, n_tot = int(keep.sum()), int(len(rows))
    if n_ok < 10:
        return None, n_ok, n_tot
    sel = rows[keep]
    pole = float(np.median([float((img[q, left:right + 1, 2] - img[q, left:right + 1, 0]).mean())
                            for q in sel]))
    sur = float(np.median([float(np.concatenate([
        img[q, c0:c1, 2] - img[q, c0:c1, 0],
        img[q, d0:d1, 2] - img[q, d0:d1, 0]]).mean()) for q in sel]))
    con = float(np.median(pole_lum[keep] - sur_lum[keep]))
    return dict(pole=pole, surround=sur, excess=pole - sur, contrast=con), n_ok, n_tot


REF, ns, ntot = fixed_measure(PIC)
print("   THE SOURCE, measured at the columns the control validated (270..275, surround +/-24):")
print("      pole %+.4f, surround %+.4f, EXCESS %+.4f, contrast %.4f, visible %d of %d rows"
      % (REF["pole"], REF["surround"], REF["excess"], REF["contrast"], ns, ntot))
print()
print("   %-40s %-11s %-11s %-11s %-11s %s"
      % ("setting", "pole", "surround", "EXCESS", "contrast", "visible rows"))


def line(label, cam):
    q, n_ok, n_tot = fixed_measure(run(cam))
    if q is None:
        print("   %-40s %-45s %d of %d  <- THE POLE IS BURIED HERE"
              % (label, "", n_ok, n_tot))
        return None
    flag = ""
    if q["excess"] < 0:
        flag = "  (the pole is DARKER than its surround)"
    print("   %-40s %-+11.4f %-+11.4f %-+11.4f %-11.4f %d of %d%s"
          % (label, q["pole"], q["surround"], q["excess"], q["contrast"], n_ok, n_tot, flag))
    return q


line("the reproduction (source_glow 0.04)", dict(CHOSEN, source_glow=0.04))
print()
for label, key, values in (
    ("source_glow", "source_glow", (0.0, 0.01, 0.02, 0.04, 0.08, 0.15)),
    ("halation", "halation", (0.0, 0.6, 1.0)),
    ("halation_source_colour", "halation_source_colour", (0.5, 1.0)),
    ("shadow_depth", "shadow_depth", (0.25, 0.5)),
    ("chroma_aberration", "chroma_aberration", (0.0, 1.0)),
):
    for v in values:
        line("%s = %.2f" % (label, v), dict(dict(CHOSEN, source_glow=0.04), **{key: v}))
    print()

print("   the source's excess is %+.4f.  A setting matches it when its excess is near that AND the row count")
print("   stays high -- a buried pole shows up as the row count collapsing, not as a number moving." % REF["excess"])
