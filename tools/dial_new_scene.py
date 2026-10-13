"""
Dial the mechanisms for this subject, since the diagnosis is settled and this render is too strong.

**What is established and no longer in question.** The pole survives because its source excess (**+0.1459**) exceeds the
chain's own shift to the surround (about **0.06**); at the reproduction's settings it comes out at **+0.2176**. The
mechanisms are the same ones that were calibrated on other material, so this is now a question of strength on **this**
subject -- and at the current values the render is over-driven: the blown fraction goes from **0.79% to 3.07%** and the
fully-white fraction from **0.096% to 2.32%**, which swallows the composition.

**So the light group is swept against two guards**, and both matter:

  * the **pole's excess** must stay well above zero, since that is the property the whole line of enquiry was about;
  * and the **blown** and **fully-white** fractions must stay near the control's, because a picture driven to white has
    no colour left to show -- the lesson from the last several turns, where exactly that erased the measurement.

`source_glow` is swept first because its magnitude is proportional to the source's own bright area, and this subject's
fixture is a large multi-bulb cluster -- the scale relation that was wrong by two orders of magnitude earlier in this
work.
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


def find_pole(img, window, min_rows=30):
    lum = img @ COEF
    Hh, Ww = lum.shape
    k = max(5, Hh // 40)
    box = np.lib.stride_tricks.sliding_window_view(lum, (k, k)).mean(axis=(2, 3))
    cy, _ = np.unravel_index(int(np.argmax(box)), box.shape)
    cy += k // 2
    start = min(Hh - 2, cy + 6)
    band = lum[start:Hh]
    if band.shape[0] < min_rows:
        return None
    wide = np.apply_along_axis(lambda r: np.convolve(r, np.ones(41) / 41, mode="same"), 1, band)
    score = (band - wide).mean(axis=0)
    lo, hi = window
    j = lo + int(np.argmax(score[lo:hi]))
    if score[j] <= 0.01:
        return None
    left = right = j
    while left - 1 >= 0 and score[left - 1] > 0.3 * score[j]:
        left -= 1
    while right + 1 < Ww and score[right + 1] > 0.3 * score[j]:
        right += 1
    off = max(10, (right - left) * 4)
    c0, c1 = max(0, left - off), max(1, left - 3)
    d0, d1 = min(Ww - 1, right + 3), min(Ww, right + off)
    pl = band[:, left:right + 1].mean(axis=1)
    sl = np.concatenate([band[:, c0:c1], band[:, d0:d1]], axis=1).mean(axis=1)
    keep = pl > sl + 0.02
    if keep.sum() < min_rows:
        return None
    rows = np.arange(start, start + len(keep))
    sel = rows[keep]
    ph = float(np.median([float((img[q, left:right + 1, 2] - img[q, left:right + 1, 0]).mean())
                          for q in sel]))
    sh = float(np.median([float(np.concatenate([
        img[q, c0:c1, 2] - img[q, c0:c1, 0],
        img[q, d0:d1, 2] - img[q, d0:d1, 0]]).mean()) for q in sel]))
    return dict(pole=ph, surround=sh, excess=ph - sh, n=int(len(sel)))


WIN = (182, 271)


def report(img):
    lum = img @ COEF
    q = find_pole(img, WIN)
    return dict(mean=float(lum.mean()), blown=float((lum > 0.97).mean()),
                glow=float((lum > 0.85).mean()),
                white=float(np.all(img >= 0.9999, axis=-1).mean()),
                excess=q["excess"] if q else None)


base = report(pic)
ctrl, _ = run(ABERR)
c = report(ctrl)
print("   the source:                     mean %.5f  blown %.5f  glow %.5f  white %.5f  pole excess %+.4f"
      % (base["mean"], base["blown"], base["glow"], base["white"], 0.1459))
print("   control (light at zero):        mean %.5f  blown %.5f  glow %.5f  white %.5f  pole excess %+.4f"
      % (c["mean"], c["blown"], c["glow"], c["white"], c["excess"]))
print()
print("   %-40s %-10s %-10s %-10s %-10s %s"
      % ("setting", "mean", "blown", "glow", "white", "pole excess"))
CANDIDATES = [
    ("the reproduction's values (too strong)", dict(source_glow=0.04, halation=0.25, ccd_smear=0.20,
                                                    flare_defocus=0.35, lens_flare=0.30,
                                                    spike_spread=0.40, overflow_drain=0.60, **ABERR)),
    ("half the light group", dict(source_glow=0.02, halation=0.12, ccd_smear=0.10,
                                  flare_defocus=0.20, lens_flare=0.15, spike_spread=0.20,
                                  overflow_drain=0.70, **ABERR)),
    ("a third", dict(source_glow=0.015, halation=0.08, ccd_smear=0.07, flare_defocus=0.12,
                     lens_flare=0.10, spike_spread=0.13, overflow_drain=0.75, **ABERR)),
    ("a quarter", dict(source_glow=0.01, halation=0.06, ccd_smear=0.05, flare_defocus=0.09,
                       lens_flare=0.07, spike_spread=0.10, overflow_drain=0.80, **ABERR)),
    ("light group minimal, aberrations only", dict(ABERR)),
]
best = None
for label, cam in CANDIDATES:
    img, _ = run(cam)
    s = report(img)
    print("   %-40s %-10.5f %-10.5f %-10.5f %-10.5f %s"
          % (label, s["mean"], s["blown"], s["glow"], s["white"],
             ("%+.4f" % s["excess"]) if s["excess"] is not None else "no pole"))
    # the guard: blown and white near the control's, with the pole's excess comfortably positive
    score = abs(s["blown"] - c["blown"]) + 3.0 * abs(s["white"] - c["white"])
    if s["excess"] is not None and s["excess"] > 0.08 and (best is None or score < best[1]):
        best = (label, score, cam, s)

print()
if best:
    print("   the best balance: %s" % best[0])
    print("      mean %.5f, blown %.5f (control %.5f), white %.5f (control %.5f), pole excess %+.4f"
          % (best[3]["mean"], best[3]["blown"], c["blown"], best[3]["white"], c["white"], best[3]["excess"]))
    img, p = run(best[2], "newscene-effect.png")
    run(ABERR, "newscene-control.png")
    diff = {k: {kk: vv for kk, vv in v.items() if BASE.get(k, {}).get(kk) != vv} for k, v in p.items()}
    diff = {k: v for k, v in diff.items() if v}
    with open(os.path.join(OUT, "repro-highlight.json"), "w", encoding="utf-8") as fh:
        json.dump({"params": diff, "param_count": sum(len(v) for v in diff.values()),
                   "seed": 20240101, "standard": "ntsc",
                   "note": "the source's pole must carry more colour contrast against its surround than the chain "
                           "shifts the surround (about 0.06): +0.0363 inverts, +0.1459 survives"},
                  fh, ensure_ascii=False, indent=2)
    print("   wrote repro-highlight.json: %d parameter(s) across %d layer(s)"
          % (sum(len(v) for v in diff.values()), len(diff)))
    for k, v in sorted(diff.items()):
        print("      %-9s %s" % (k, {kk: (round(vv, 3) if isinstance(vv, float) else vv)
                                    for kk, vv in sorted(v.items())}))
