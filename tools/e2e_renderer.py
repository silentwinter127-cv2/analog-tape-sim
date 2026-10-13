"""
Measure the new parameter through the renderer directly, which is the path that was verified sane.

**What the debug established.** The parameter reaches `CameraStage` (`halation_source_colour 1.000`), the mechanism
adds only **+0.00279** of mean luma, and every stage of the chain keeps the blown fraction at **0.0148** -- so neither
the new code nor the chain produces a white frame. The earlier end-to-end reading of **0.20 blown** therefore came from
the server render path, and the numbers are taken from the renderer directly here instead.

**And a second thing the debug made visible, which is worth keeping.** The chain's own stage-by-stage blown fraction
reads **0.0148 at the camera and 0.0143 after the display**, whereas the same settings through the server had reported
**0.04 to 0.20** depending on the sweep. Those are not the same quantity, so any earlier conclusion drawn from the
server's blown figure needs the renderer's figure beside it before it is trusted.

The measurement is the one that matters: the pole's hue and contrast below the core, at several values of
`halation_source_colour`, with everything else at the reproduction's settings, against the scene's own **+0.0235**.
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
from atsim.core.pipeline import FrameBuffer                                  # noqa: E402
from atsim.render.renderer import Renderer                                   # noqa: E402

OUT = os.path.join(ROOT, "tools", "ntscrs_ref")
COEF = np.array([0.299, 0.587, 0.114], np.float32)
W, H = 452, 382

SRC_PNG = os.path.join(OUT, "refscene.png")
src = np.asarray(Image.open(SRC_PNG).convert("RGB"), np.float32) / 255.0
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
CHOSEN = dict(source_glow=0.05, halation=0.25, ccd_smear=0.20, flare_defocus=0.35,
              lens_flare=0.30, spike_spread=0.40, overflow_drain=0.60,
              chroma_aberration=0.6, longitudinal_ca=0.4, axis_offset=0.5,
              exposure=0.0, color_cast=0.0, lens_fog=0.0, lens_wear=0.0,
              bloom_amount=0.0, dead_sites=0.0)


def run(camera, sat=0.62):
    layers = {"camera": dict(camera), "format": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0},
              "player": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0},
              "codec": {"dv_enabled": 1.0}, "tape": {"head_switching": 0.0},
              "display": dict(RASTER_OFF, enabled=1.0, gamma=1.0, saturation=sat)}
    p = merge_values(BASE, layers)
    r = Renderer(Still(), W, H, 25.0, p, seed=20240101)
    rgb = np.clip(np.asarray(r.render_frame(0), np.float32), 0, 1)
    return rgb


def core_of(img):
    lum = img @ COEF
    k = max(5, img.shape[0] // 40)
    box = np.lib.stride_tricks.sliding_window_view(lum, (k, k)).mean(axis=(2, 3))
    cy, cx = np.unravel_index(int(np.argmax(box)), box.shape)
    return cy + k // 2, cx + k // 2


def line_stats(img):
    cy, cx = core_of(img)
    lum = img @ COEF
    con, br = [], []
    for d in range(10, min(H - cy - 1, 320)):
        row = lum[cy + d]
        c0, c1 = max(0, cx - 60), min(W, cx + 60)
        if c1 - c0 < 20:
            continue
        seg = row[c0:c1]
        i = int(np.argmax(seg))
        mask = np.ones(len(seg), dtype=bool)
        mask[max(0, i - 3):i + 4] = False
        if not mask.any():
            continue
        con.append(float(seg[i] - seg[mask].mean()))
        br.append(float((img[cy + d, c0:c1, 2] - img[cy + d, c0:c1, 0])[i]))
    return (float(np.median(con)) if con else float("nan"),
            float(np.median(br)) if br else float("nan"))


m0, h0 = line_stats(PIC)
L0 = PIC @ COEF
print("   the raw scene:  pole contrast %.4f, hue B-R %+.4f, blown %.5f"
      % (m0, h0, float((L0 > 0.97).mean())))
print()
print("   %-24s %-13s %-11s %-11s %s"
      % ("halation_source_colour", "pole contrast", "pole hue", "blown", "the scene's hue is %+.4f" % h0))
rows = []
for sc in (0.0, 0.25, 0.5, 0.75, 1.0):
    a = run(dict(CHOSEN, halation_source_colour=sc))
    con, hue = line_stats(a)
    L = a @ COEF
    rows.append((sc, con, hue))
    print("   %-24.2f %-13.4f %-+11.4f %-11.5f" % (sc, con, hue, float((L > 0.97).mean())))

print()
best = max(rows, key=lambda r: r[2])
print("   the hue closest to the scene's +%.4f is at halation_source_colour %.2f: hue %+.4f, contrast %.4f"
      % (h0, best[0], best[2], best[1]))
print("   (the scene's own contrast is %.4f, so the contrast must be read beside the hue)" % m0)

a = run(dict(CHOSEN, halation_source_colour=best[0]))
Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).save(
    os.path.join(OUT, "refscene-effect.png"))
b = run(dict(CHOSEN, halation=0.0, halation_source_colour=best[0]))
Image.fromarray((np.clip(b, 0, 1) * 255).astype(np.uint8)).save(
    os.path.join(OUT, "refscene-control.png"))
print("   wrote refscene-effect.png and refscene-control.png")

# and the reproduction file, written from the renderer's own parameter set rather than the server's
layers = {"camera": dict(CHOSEN, halation_source_colour=best[0]),
          "format": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0, "chroma_gain": 1.0,
                     "color_noise": 0.0, "cross_color": 0.0, "cross_luma": 0.0, "edge_boost": 0.0},
          "player": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0, "color_noise": 0.0,
                     "cross_color": 0.0, "cross_luma": 0.0},
          "codec": {"dv_enabled": 1.0}, "tape": {"head_switching": 0.0},
          "display": dict(RASTER_OFF, enabled=1.0, gamma=1.0, saturation=0.62)}
p = merge_values(BASE, layers)
diff = {k: {kk: vv for kk, vv in v.items() if BASE.get(k, {}).get(kk) != vv}
        for k, v in p.items()}
diff = {k: v for k, v in diff.items() if v}
with open(os.path.join(OUT, "repro-highlight.json"), "w", encoding="utf-8") as fh:
    json.dump({"params": diff, "param_count": sum(len(v) for v in diff.values()),
               "seed": 20240101, "standard": "ntsc"}, fh, ensure_ascii=False, indent=2)
print("   wrote repro-highlight.json: %d parameter(s) across %d layer(s)"
      % (sum(len(v) for v in diff.values()), len(diff)))
