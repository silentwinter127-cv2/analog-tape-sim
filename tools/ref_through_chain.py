"""
The decisive experiment: run the reference photograph itself through the chain and see whether its pole inverts too.

**Why this separates the two possibilities cleanly.** The pole's excess is **+0.0363** on the reference scene
(`refscene.png`) and **negative in every render**, across eleven excluded candidates. Two explanations remain and they
predict different outcomes here:

  * **if the reference photograph's own pole also inverts**, then the chain does this to poles of this kind as a general
    behaviour, and the fault is in the chain's handling of a bright thin structure against a darker surround;
  * **if it does not invert**, then the chain is faithful and the difference lies **between the two pictures** -- their
    exposure, the pole's brightness relative to its surround, or how close to white the pole sits.

**And there is a specific reason to expect the second.** The render's pole is **10.8% fully white** and the source's is
essentially never white, so if the reference photograph's pole likewise stays below saturation through the chain, the
mechanism at work is the chain pushing this particular subject past white -- not a general inversion.

**The measurement is the same validated one**, with the geometry taken from **each picture's own pole**, since the
reference photograph frames its lamp differently and the fixed columns would not land on it. That means the pole
detector runs on both -- and it has been through the positive control, so its output is evidence: it is fooled by a
bright edge if unconstrained, so the search is bounded, and it returns "not found" rather than inventing a structure.
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
BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])

REF_PHOTO = (r"C:\Users\Administrator\.dsh\attachments\v1\objects\03"
             r"\039d9af408f07923375d9fae0a7fb6839668f20b2c0508face5b0a2a14b56eb9")


class Still:
    def __init__(self, pic):
        self.height, self.width = pic.shape[:2]
        self.fps, self.frame_count = 25.0, 3
        self._pic = pic

    def frame(self, index):
        return self._pic


RASTER_OFF = {"scanlines": 0.0, "mask_depth": 0.0, "interlace_flicker": 0.0, "scan_pitch": 0.0,
              "spot_sigma": 0.0, "defocus": 0.0, "vignette": 0.0, "reflection": 0.0,
              "bloom": 0.0, "geometry_warp": 0.0, "overscan": 0.0, "curvature": 0.0,
              "breathing": 0.0, "crt_age": 0.0, "convergence": 0.0, "crt_moisture": 0.0,
              "crt_corrosion": 0.0}
LIGHT = dict(halation=0.25, ccd_smear=0.20, flare_defocus=0.35, lens_flare=0.30,
             spike_spread=0.40, overflow_drain=0.60, chroma_aberration=0.6,
             longitudinal_ca=0.4, axis_offset=0.5, halation_source_colour=0.0,
             shadow_depth=0.0, source_glow=0.04, exposure=0.0, color_cast=0.0,
             lens_fog=0.0, lens_wear=0.0, bloom_amount=0.0, dead_sites=0.0)


def render(pic, cam=None):
    layers = {"camera": dict(LIGHT, **(cam or {})),
              "format": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0, "chroma_gain": 1.0,
                         "color_noise": 0.0, "cross_color": 0.0, "cross_luma": 0.0,
                         "edge_boost": 0.0},
              "player": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0, "color_noise": 0.0,
                         "cross_color": 0.0, "cross_luma": 0.0},
              "codec": {"dv_enabled": 1.0}, "tape": {"head_switching": 0.0},
              "display": dict(RASTER_OFF, enabled=1.0, gamma=1.0, saturation=0.62)}
    p = merge_values(BASE, layers)
    r = Renderer(Still(pic), pic.shape[1], pic.shape[0], 25.0, p, seed=20240101)
    return np.clip(np.asarray(r.render_frame(0), np.float32), 0, 1)


def find_pole(img, min_rows=30, window=None, min_score=0.01):
    """
    The pole: the column band standing above its horizontal neighbours **consistently down the frame**.

    Bounded by `window` when given, because the positive control showed the unconstrained rule is defeated by a bright
    vertical edge. Returns None when no band holds up, so a frame with no pole is reported rather than measured.
    """
    lum = img @ COEF
    Hh, Ww = lum.shape
    k = max(5, Hh // 40)
    box = np.lib.stride_tricks.sliding_window_view(lum, (k, k)).mean(axis=(2, 3))
    cy, cx = np.unravel_index(int(np.argmax(box)), box.shape)
    cy += k // 2
    cx += k // 2
    start = min(Hh - 2, cy + 6)
    band = lum[start:Hh]
    if band.shape[0] < min_rows:
        return None
    wide = np.apply_along_axis(lambda r: np.convolve(r, np.ones(41) / 41, mode="same"), 1, band)
    score = (band - wide).mean(axis=0)
    if window is None:
        lo, hi = max(0, cx - Ww // 3), min(Ww, cx + Ww // 3)
    else:
        lo, hi = window
    j = lo + int(np.argmax(score[lo:hi]))
    if score[j] <= min_score:
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
    return dict(start=start, left=int(left), right=int(right), c0=int(c0), c1=int(c1),
                d0=int(d0), d1=int(d1), keep=keep, width=int(right - left + 1))


def measure(img, window=None):
    r = find_pole(img, window=window)
    if r is None:
        return None
    rows = np.arange(r["start"], r["start"] + len(r["keep"]))
    sel = rows[r["keep"]]
    lum = img @ COEF
    ph = float(np.median([float((img[q, r["left"]:r["right"] + 1, 2]
                                 - img[q, r["left"]:r["right"] + 1, 0]).mean()) for q in sel]))
    sh = float(np.median([float(np.concatenate([
        img[q, r["c0"]:r["c1"], 2] - img[q, r["c0"]:r["c1"], 0],
        img[q, r["d0"]:r["d1"], 2] - img[q, r["d0"]:r["d1"], 0]]).mean()) for q in sel]))
    return dict(band=(r["left"], r["right"]), width=r["width"], pole=ph, surround=sh,
                excess=ph - sh, n=int(len(sel)),
                band_white=float(np.all(img[rows][:, r["left"]:r["right"] + 1] >= 0.9999,
                                        axis=-1).mean()),
                frame_white=float(np.all(img >= 0.9999, axis=-1).mean()))


ref = np.asarray(Image.open(REF_PHOTO).convert("RGB"), np.float32) / 255.0
print("   the reference PHOTOGRAPH: %dx%d, mean luma %.4f, fully white %.5f"
      % (ref.shape[1], ref.shape[0], float((ref @ COEF).mean()),
         float(np.all(ref >= 0.9999, axis=-1).mean())))
q0 = measure(ref)
if q0 is None:
    print("   no pole found on the photograph itself -- nothing to compare")
    raise SystemExit(1)
print("   its pole, located on itself:  band %d..%d (%d px), %d rows"
      % (q0["band"][0], q0["band"][1], q0["width"], q0["n"]))
print("      pole %+.4f, surround %+.4f, EXCESS %+.4f" % (q0["pole"], q0["surround"], q0["excess"]))
print()

# the geometry from the photograph's own pole, so the render is measured on the same structure
L, R2 = q0["band"]
WIN = (max(0, L - 40), min(ref.shape[1], R2 + 40))

print("   THE EXPERIMENT: the reference photograph through the chain, measured on its own pole geometry")
print("   %-24s %-11s %-11s %-11s %-12s %s"
      % ("", "pole", "surround", "EXCESS", "band white", "sign"))
print("   %-24s %-+11.4f %-+11.4f %-+11.4f %-12.5f %s"
      % ("the photograph", q0["pole"], q0["surround"], q0["excess"], q0["band_white"],
         "cooler" if q0["excess"] > 0 else "warmer"))

for label, cam in (("through the chain", {}),
                   ("chain, light mechanisms at zero",
                    {k: 0.0 for k in ("halation", "ccd_smear", "flare_defocus", "lens_flare",
                                      "spike_spread", "overflow_drain", "chroma_aberration",
                                      "longitudinal_ca", "axis_offset", "source_glow")}),
                   ("chain, exposure -0.6", {"exposure": -0.6})):
    out = render(ref, cam)
    q = measure(out, window=WIN)
    if q is None:
        print("   %-24s not found (no pole stands out in the render)" % label)
        continue
    print("   %-24s %-+11.4f %-+11.4f %-+11.4f %-12.5f %s"
          % (label, q["pole"], q["surround"], q["excess"], q["band_white"],
             ("cooler" if q["excess"] > 0 else "WARMER -- INVERTED")))
    print("   %-24s (band %d..%d, %d px, %d rows)"
          % ("", q["band"][0], q["band"][1], q["width"], q["n"]))

print()
print("   the two outcomes and what each means:")
print("      if the photograph's pole inverts too -> the chain does this to poles generally;")
print("      if it stays cooler                     -> the chain is faithful and the difference is between the")
print("                                                two pictures -- their exposure and how near white the pole is.")
