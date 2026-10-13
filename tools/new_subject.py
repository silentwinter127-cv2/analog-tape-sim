"""
Measure the new photograph's pole, which is the property the last experiment identified as decisive.

**What the previous experiment established.** The chain does **not** invert poles: the reference photograph's pole keeps
an excess of **+0.2016** through it, while the reproduction's photograph has only **+0.0363** to begin with and the
chain's own additions shift the **surround** by about **0.06** -- larger than the whole difference it was supposed to
preserve. **A difference smaller than the perturbation cannot survive it**, so the remedy is material whose pole
carries a **larger colour contrast against its own surround.**

**And this photograph looks like it does.** A multiple-bulb chandelier whose bulbs read blue-white against a dark sky,
on a pole, with a **warm orange building** behind -- which is a much larger pole-to-surround colour separation than the
previous subject's blank blue-grey wall.

**Measured rather than assumed**, with the detector that passed the positive control: its pole is located, its width and
row count reported, and its **excess** compared against the two figures already known -- the previous photograph's
**+0.0363** and the reference's **+0.2142**. The instrument is run bounded, and it returns "not found" rather than
inventing a structure, so a figure it produces here is evidence.

**And then the same experiment as last time**: the photograph through the chain, to see whether the larger difference
survives where the smaller one did not. **That is the test of the diagnosis**, not just of the picture.
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
NEW = (r"C:\Users\Administrator\.dsh\attachments\v1\objects\1e"
       r"\1e772675f0c138e382f524eba8e0f9a91ee8b04e9622b001a611521f3048e2c1")


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
    lo, hi = window if window else (max(0, cx - Ww // 3), min(Ww, cx + Ww // 3))
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
                d0=int(d0), d1=int(d1), keep=keep, width=int(right - left + 1), cx=cx)


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
                excess=ph - sh, n=int(len(sel)), cx=int(r["cx"]),
                band_white=float(np.all(img[rows][:, r["left"]:r["right"] + 1] >= 0.9999,
                                        axis=-1).mean()),
                frame_white=float(np.all(img >= 0.9999, axis=-1).mean()))


pic = np.asarray(Image.open(NEW).convert("RGB"), np.float32) / 255.0
sl = pic @ COEF
print("   the new photograph: %dx%d, mean luma %.4f, blown(>0.97) %.5f, fully white %.5f, R-B %+.4f"
      % (pic.shape[1], pic.shape[0], float(sl.mean()), float((sl > 0.97).mean()),
         float(np.all(pic >= 0.9999, axis=-1).mean()),
         float(pic[..., 0].mean() - pic[..., 2].mean())))

q0 = measure(pic)
if q0 is None:
    print("   no pole found -- nothing to measure")
    raise SystemExit(1)
print()
print("   its pole, located on itself:  band %d..%d (%d px), %d rows"
      % (q0["band"][0], q0["band"][1], q0["width"], q0["n"]))
print("      pole %+.4f, surround %+.4f, EXCESS %+.4f" % (q0["pole"], q0["surround"], q0["excess"]))

print()
print("   THE THREE SUBJECTS SIDE BY SIDE -- the quantity the diagnosis says matters:")
print("   %-34s %-11s %-11s %s" % ("photograph", "pole", "surround", "EXCESS"))
print("   %-34s %-+11.4f %-+11.4f %+.4f" % ("refscene.png (the earlier one)", 0.0062, -0.0301, 0.0363))
print("   %-34s %-+11.4f %-+11.4f %+.4f" % ("the reference photograph", 0.1049, -0.1093, 0.2142))
print("   %-34s %-+11.4f %-+11.4f %+.4f" % ("this new one", q0["pole"], q0["surround"], q0["excess"]))

L, R2 = q0["band"]
WIN = (max(0, L - 40), min(pic.shape[1], R2 + 40))
print()
print("   AND THE TEST OF THE DIAGNOSIS -- does the larger difference survive the chain?")
print("   %-30s %-11s %-11s %-11s %-12s %s"
      % ("", "pole", "surround", "EXCESS", "band white", "sign"))
print("   %-30s %-+11.4f %-+11.4f %-+11.4f %-12.5f %s"
      % ("the photograph", q0["pole"], q0["surround"], q0["excess"], q0["band_white"],
         "cooler" if q0["excess"] > 0 else "warmer"))
for label, cam in (("through the chain", {}),
                   ("chain, exposure -0.6", {"exposure": -0.6}),
                   ("chain, light mechanisms at zero",
                    {k: 0.0 for k in ("halation", "ccd_smear", "flare_defocus", "lens_flare",
                                      "spike_spread", "overflow_drain", "chroma_aberration",
                                      "longitudinal_ca", "axis_offset", "source_glow")})):
    out = render(pic, cam)
    q = measure(out, window=WIN)
    if q is None:
        print("   %-30s not found" % label)
        continue
    print("   %-30s %-+11.4f %-+11.4f %-+11.4f %-12.5f %s"
          % (label, q["pole"], q["surround"], q["excess"], q["band_white"],
             "COOLER -- the difference survived" if q["excess"] > 0 else "warmer -- inverted"))
print()
print("   the diagnosis predicts: a pole whose source excess is near the reference's 0.2142 keeps its sign,")
print("   and the earlier photograph's 0.0363 -- smaller than the chain's own ~0.06 shift -- does not.")
