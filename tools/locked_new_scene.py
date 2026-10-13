"""
Lock the geometry to the source's own pole, then dial the light group against a measurement that does not move.

**Why the previous sweep was unusable.** The detector jumped between structures as the render's brightness profile
changed -- the same failure the positive control identified (a bright vertical edge defeating the rule) -- so its pole
excess read +0.2176, +0.0297, +0.2729, -0.0387 across neighbouring settings. **A measurement that moves with the thing
being measured cannot tune anything.**

**The fix is the one that worked before**: the columns and the surround offset come from the **source** once, where the
pole is unambiguous, and the detector's only job is the **visibility check** -- does the band still clear its surround
on this frame. The location never moves, so what changes between settings is the colour and nothing else.

**And this subject needs a lower setting than the reproduction's**, since at those values the blown fraction goes from
0.79% to 3.07% and the fully-white fraction from 0.096% to 2.32%, which swallows the composition. The sweep is run with
the geometry locked and two guards: the pole's excess must stay comfortably positive, and the blown and white fractions
must stay near the control's -- because a picture driven to white has no colour left to show, which is the lesson from
several turns of this work.
"""
from __future__ import annotations

import json
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
BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])
pic = np.asarray(Image.open(os.path.join(OUT, "newscene.png")).convert("RGB"),
                 np.float32) / 255.0
# the pole's own geometry, located on the source and then FIXED
POLE_L, POLE_R, OFF = 222, 231, 40


class Still:
    def __init__(self, p):
        self.height, self.width = p.shape[:2]
        self.fps, self.frame_count = 25.0, 3
        self._pic = p

    def frame(self, index):
        return self._pic


RASTER_OFF = {"scanlines": 0.0, "mask_depth": 0.0, "interlace_flicker": 0.0, "scan_pitch": 0.0,
              "spot_sigma": 0.0, "defocus": 0.0, "vignette": 0.0, "reflection": 0.0,
              "bloom": 0.0, "geometry_warp": 0.0, "overscan": 0.0, "curvature": 0.0,
              "breathing": 0.0, "crt_age": 0.0, "convergence": 0.0, "crt_moisture": 0.0,
              "crt_corrosion": 0.0}
ZERO = dict(halation=0.0, ccd_smear=0.0, flare_defocus=0.0, lens_flare=0.0, spike_spread=0.0,
            overflow_drain=0.0, chroma_aberration=0.0, longitudinal_ca=0.0, axis_offset=0.0,
            source_glow=0.0, halation_source_colour=0.0, shadow_depth=0.0, exposure=0.0,
            color_cast=0.0, lens_fog=0.0, lens_wear=0.0, bloom_amount=0.0, dead_sites=0.0)
ABERR = dict(chroma_aberration=0.6, longitudinal_ca=0.4, axis_offset=0.5)


def run(cam=None, name=None):
    layers = {"camera": dict(ZERO, **(cam or {})),
              "format": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0, "chroma_gain": 1.0,
                         "color_noise": 0.0, "cross_color": 0.0, "cross_luma": 0.0,
                         "edge_boost": 0.0},
              "player": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0, "color_noise": 0.0,
                         "cross_color": 0.0, "cross_luma": 0.0},
              "codec": {"dv_enabled": 1.0}, "tape": {"head_switching": 0.0},
              "display": dict(RASTER_OFF, enabled=1.0, gamma=1.0, saturation=0.62)}
    p = merge_values(BASE, layers)
    r = Renderer(Still(pic), pic.shape[1], pic.shape[0], 25.0, p, seed=20240101)
    out = np.clip(np.asarray(r.render_frame(0), np.float32), 0, 1)
    if name:
        Image.fromarray((out * 255).astype(np.uint8)).save(os.path.join(OUT, name))
    return out, p


def locked(img, left=POLE_L, right=POLE_R, off=OFF, min_contrast=0.02):
    """
    Measure at **fixed columns taken from the source**, reporting whether the band still clears its surround.

    The location never moves, so a mechanism that buries the pole shows up as a falling visible-row count rather than
    as a colour figure measured somewhere else.
    """
    lum = img @ COEF
    Hh, Ww = lum.shape
    k = max(5, Hh // 40)
    box = np.lib.stride_tricks.sliding_window_view(lum, (k, k)).mean(axis=(2, 3))
    cy, _ = np.unravel_index(int(np.argmax(box)), box.shape)
    cy += k // 2
    rows = np.arange(min(Hh - 2, cy + 6), Hh)
    c0, c1 = max(0, left - off), max(1, left - 3)
    d0, d1 = min(Ww - 1, right + 3), min(Ww, right + off)
    pl = lum[rows][:, left:right + 1].mean(axis=1)
    sl = np.concatenate([lum[rows][:, c0:c1], lum[rows][:, d0:d1]], axis=1).mean(axis=1)
    keep = pl > sl + min_contrast
    n_ok, n_tot = int(keep.sum()), int(len(rows))
    if n_ok < 10:
        return None, n_ok, n_tot
    sel = rows[keep]
    ph = float(np.median([float((img[q, left:right + 1, 2] - img[q, left:right + 1, 0]).mean())
                          for q in sel]))
    sh = float(np.median([float(np.concatenate([
        img[q, c0:c1, 2] - img[q, c0:c1, 0],
        img[q, d0:d1, 2] - img[q, d0:d1, 0]]).mean()) for q in sel]))
    return dict(pole=ph, surround=sh, excess=ph - sh), n_ok, n_tot


def stats(img):
    lum = img @ COEF
    q, n_ok, n_tot = locked(img)
    return dict(mean=float(lum.mean()), blown=float((lum > 0.97).mean()),
                glow=float((lum > 0.85).mean()),
                white=float(np.all(img >= 0.9999, axis=-1).mean()),
                excess=q["excess"] if q else None, rows=(n_ok, n_tot))


s_src = stats(pic)
ctrl, _ = run(ABERR)
s_ctrl = stats(ctrl)
print("   geometry LOCKED to the source's pole: columns %d..%d, surround +/-%d" % (POLE_L, POLE_R, OFF))
print()
print("   %-38s %-10s %-10s %-10s %-10s %-12s %s"
      % ("", "mean", "blown", "glow", "white", "pole excess", "visible rows"))
print("   %-38s %-10.5f %-10.5f %-10.5f %-10.5f %-12s %d of %d"
      % ("THE SOURCE", s_src["mean"], s_src["blown"], s_src["glow"], s_src["white"],
         ("%+.4f" % s_src["excess"]) if s_src["excess"] else "n/a", s_src["rows"][0], s_src["rows"][1]))
print("   %-38s %-10.5f %-10.5f %-10.5f %-10.5f %-12s %d of %d"
      % ("control (light at zero)", s_ctrl["mean"], s_ctrl["blown"], s_ctrl["glow"], s_ctrl["white"],
         ("%+.4f" % s_ctrl["excess"]) if s_ctrl["excess"] else "not visible",
         s_ctrl["rows"][0], s_ctrl["rows"][1]))
print()

CANDIDATES = [
    ("reproduction's light group", dict(source_glow=0.04, halation=0.25, ccd_smear=0.20,
                                        flare_defocus=0.35, lens_flare=0.30, spike_spread=0.40,
                                        overflow_drain=0.60, **ABERR)),
    ("a quarter of it", dict(source_glow=0.01, halation=0.06, ccd_smear=0.05, flare_defocus=0.09,
                             lens_flare=0.07, spike_spread=0.10, overflow_drain=0.80, **ABERR)),
    ("an eighth", dict(source_glow=0.005, halation=0.03, ccd_smear=0.025, flare_defocus=0.045,
                       lens_flare=0.035, spike_spread=0.05, overflow_drain=0.85, **ABERR)),
    ("a sixteenth", dict(source_glow=0.0025, halation=0.015, ccd_smear=0.012, flare_defocus=0.022,
                         lens_flare=0.018, spike_spread=0.025, overflow_drain=0.9, **ABERR)),
    ("aberrations only (light at zero)", dict(ABERR)),
]
best = None
for label, cam in CANDIDATES:
    img, _ = run(cam)
    s = stats(img)
    print("   %-38s %-10.5f %-10.5f %-10.5f %-10.5f %-12s %d of %d"
          % (label, s["mean"], s["blown"], s["glow"], s["white"],
             ("%+.4f" % s["excess"]) if s["excess"] is not None else "not visible",
             s["rows"][0], s["rows"][1]))
    if s["excess"] is not None and s["excess"] > 0.10 and s["white"] < 4 * max(1e-5, s_ctrl["white"]) + 0.002:
        cost = abs(s["blown"] - s_ctrl["blown"]) + 3.0 * abs(s["white"] - s_ctrl["white"])
        if best is None or cost < best[1]:
            best = (label, cost, cam, s)

print()
if best is None:
    print("   no candidate keeps the pole visible with the blowout near the control's; the conservative one is the")
    print("   control itself, and the light group should be raised by eye from there.")
    best = ("aberrations only (light at zero)", 0.0, dict(ABERR), s_ctrl)
print("   the chosen setting: %s" % best[0])
print("      mean %.5f, blown %.5f (control %.5f), white %.5f (control %.5f), pole excess %s"
      % (best[3]["mean"], best[3]["blown"], s_ctrl["blown"], best[3]["white"], s_ctrl["white"],
         ("%+.4f" % best[3]["excess"]) if best[3]["excess"] is not None else "not visible"))
img, p = run(best[2], "newscene-effect.png")
run(ABERR, "newscene-control.png")
diff = {k: {kk: vv for kk, vv in v.items() if BASE.get(k, {}).get(kk) != vv} for k, v in p.items()}
diff = {k: v for k, v in diff.items() if v}
with open(os.path.join(OUT, "repro-highlight.json"), "w", encoding="utf-8") as fh:
    json.dump({"params": diff, "param_count": sum(len(v) for v in diff.values()),
               "seed": 20240101, "standard": "ntsc",
               "note": "the source's pole must carry more colour contrast against its own surround than the chain "
                       "shifts that surround (about 0.06): measured, +0.0363 inverts and +0.1459 survives. "
                       "The light group is kept low on this subject because it drives the frame to white."},
              fh, ensure_ascii=False, indent=2)
print("   wrote repro-highlight.json: %d parameter(s) across %d layer(s)"
      % (sum(len(v) for v in diff.values()), len(diff)))
for k, v in sorted(diff.items()):
    print("      %-9s %s" % (k, {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                                for kk, vv in sorted(v.items())}))
