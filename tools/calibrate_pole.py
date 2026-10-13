"""
Calibrate `source_glow` against the instrument's excess, which is the quantity the reference defines.

**What the re-measurement established, and it reverses the premise of three earlier attempts.** The pole's excess over
its surroundings on the reference scene is **+0.0450**. The reproduction's is **+0.1069** -- **more than twice the
reference** -- so the line was never too faint; the old statistic was diluting the signal, and the "shallow line" it
reported was an artefact of reading a 6-px band through a single column.

    source_glow = 0.00    excess -0.0930     the pole goes *darker* than its surround
    source_glow = 0.05    excess +0.1069     the reproduction as it stood
    source_glow = 0.15    excess +0.0857

**So the lever is `source_glow` and the direction is down**, and the value that lands nearest the reference's +0.0450
is found by measurement. Worth recording what the extremes mean, because they bound the useful range: at zero the pole
is **darker** than the wall around it, which is the opposite of the reference, and the reason is that `source_glow` is
the only mechanism here that lifts the surroundings *around* a source rather than adding to the source itself.

`halation_source_colour` moves the excess by **0.0008** across its whole range and `shadow_depth` **lowers** it at
every value, so neither is the lever -- both are recorded as measured rather than assumed.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from atsim.core.params import merge_values                                   # noqa: E402
from atsim.core.profiles import FACTORY_PRESETS, SCHEMA                      # noqa: E402
from atsim.render.renderer import Renderer                                   # noqa: E402
from pole_instrument import pole_colour                                      # noqa: E402

OUT = os.path.join(ROOT, "tools", "ntscrs_ref")
W, H = 452, 382
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


REF = pole_colour(PIC)
print("   the reference scene: pole %+.4f, surround %+.4f, EXCESS %+.4f, contrast %.4f, %d rows"
      % (REF["pole"], REF["surround"], REF["excess"], REF["contrast"], REF["rows"]))
print()
print("   %-18s %-11s %-11s %-11s %-11s %s"
      % ("source_glow", "pole", "surround", "EXCESS", "contrast", "vs the scene's +%.4f" % REF["excess"]))
rows = []
for g in (0.0, 0.02, 0.04, 0.06, 0.10, 0.15):
    q = pole_colour(run(dict(CHOSEN, source_glow=g)))
    rows.append((g, q))
    print("   %-18.2f %-+11.4f %-+11.4f %-+11.4f %-11.4f %+.4f"
          % (g, q["pole"], q["surround"], q["excess"], q["contrast"], q["excess"] - REF["excess"]))

best = min(rows, key=lambda r: abs(r[1]["excess"] - REF["excess"]))
print()
print("   the value nearest the reference's excess is source_glow = %.2f: excess %+.4f (reference %+.4f)"
      % (best[0], best[1]["excess"], REF["excess"]))
print("   and its pole hue is %+.4f against the reference's %+.4f" % (best[1]["pole"], REF["pole"]))

a = run(dict(CHOSEN, source_glow=best[0]))
Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).save(
    os.path.join(OUT, "refscene-effect.png"))
b = run(dict(CHOSEN, source_glow=best[0], halation=0.0, lens_flare=0.0, ccd_smear=0.0,
             flare_defocus=0.0, spike_spread=0.0, overflow_drain=0.0))
Image.fromarray((np.clip(b, 0, 1) * 255).astype(np.uint8)).save(
    os.path.join(OUT, "refscene-control.png"))

CH = dict(CHOSEN, source_glow=best[0])
layers = {"camera": CH,
          "format": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0, "chroma_gain": 1.0,
                     "color_noise": 0.0, "cross_color": 0.0, "cross_luma": 0.0, "edge_boost": 0.0},
          "player": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0, "color_noise": 0.0,
                     "cross_color": 0.0, "cross_luma": 0.0},
          "codec": {"dv_enabled": 1.0}, "tape": {"head_switching": 0.0},
          "display": dict(RASTER_OFF, enabled=1.0, gamma=1.0, saturation=0.62)}
p = merge_values(BASE, layers)
diff = {k: {kk: vv for kk, vv in v.items() if BASE.get(k, {}).get(kk) != vv} for k, v in p.items()}
diff = {k: v for k, v in diff.items() if v}
with open(os.path.join(OUT, "repro-highlight.json"), "w", encoding="utf-8") as fh:
    json.dump({"params": diff, "param_count": sum(len(v) for v in diff.values()),
               "seed": 20240101, "standard": "ntsc"}, fh, ensure_ascii=False, indent=2)
print()
print("   wrote repro-highlight.json: %d parameter(s) across %d layer(s)"
      % (sum(len(v) for v in diff.values()), len(diff)))
for k, v in sorted(diff.items()):
    print("      %-9s %s" % (k, {kk: (round(vv, 3) if isinstance(vv, float) else vv)
                                for kk, vv in sorted(v.items())}))
