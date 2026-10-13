"""
Render the three modes using the calibrated defaults, and write the deliverables.

**Why the previous run still showed a 0.247 blown fraction.** That test passed `nv_exposure_steps = 0.8` explicitly,
which overrode the calibrated default of **0.0** -- so it was measuring the old setting rather than the new one. **A
test that supplies its own value cannot report on a default**, which is the same "the instrument was measuring
something else" fault that recurred earlier in this work, and it is fixed by letting the mode's own parameters come from
the schema.

**Then each mode is reported against the clip's segmented targets:**

    NightShot        G share 0.44-0.58, hue 0.06-0.18, ratio 0.29:0.46:0.25
    colour modes     G share 0.28-0.30, hue 0.24-0.37

and the difference between the two green modes is read from the **high-frequency energy**, which is where accumulation
shows: Super NightShot sums frames so its noise is **lower**, and the reference clip's own green segments differ by a
factor of about four for exactly that reason.
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
NIGHT = os.path.join(ROOT, "tools", "nightshot")
COEF = np.array([0.299, 0.587, 0.114], np.float32)
src = np.asarray(Image.open(os.path.join(OUT, "newscene.png")).convert("RGB"),
                 np.float32) / 255.0
H, W = src.shape[:2]
# a sequence that differs between frames, so accumulation has something to average
SEQ = [src, np.roll(src, 2, axis=1), np.roll(src, -3, axis=0),
       np.roll(np.roll(src, 5, axis=1), 2, axis=0), np.roll(src, 7, axis=1),
       np.roll(src, -5, axis=0), np.roll(np.roll(src, 3, axis=1), -2, axis=0), np.roll(src, 9, axis=1)]
BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])
RASTER_OFF = {"scanlines": 0.0, "mask_depth": 0.0, "interlace_flicker": 0.0, "scan_pitch": 0.0,
              "spot_sigma": 0.0, "defocus": 0.0, "vignette": 0.0, "reflection": 0.0,
              "bloom": 0.0, "geometry_warp": 0.0, "overscan": 0.0, "curvature": 0.0,
              "breathing": 0.0, "crt_age": 0.0, "convergence": 0.0, "crt_moisture": 0.0,
              "crt_corrosion": 0.0}
OFF = dict(halation=0.0, ccd_smear=0.0, flare_defocus=0.0, lens_flare=0.0, spike_spread=0.0,
           overflow_drain=0.0, chroma_aberration=0.0, longitudinal_ca=0.0, axis_offset=0.0,
           source_glow=0.0, halation_source_colour=0.0, shadow_depth=0.0, exposure=0.0)


class Seq:
    def __init__(self, pics):
        self.height, self.width = pics[0].shape[:2]
        self.fps, self.frame_count = 25.0, len(pics)
        self._p = pics

    def frame(self, index):
        return self._p[index % len(self._p)]


def render(cam, nframes=1):
    """Only the mode and its own values are passed; everything else comes from the schema's defaults."""
    layers = {"camera": dict(OFF, **cam),
              "format": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0},
              "player": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0},
              "codec": {"dv_enabled": 1.0}, "tape": {"head_switching": 0.0},
              "display": dict(RASTER_OFF, enabled=1.0, gamma=1.0, saturation=0.62)}
    p = merge_values(BASE, layers)
    r = Renderer(Seq(SEQ), W, H, 25.0, p, seed=20240101)
    return np.clip(np.asarray(r.render_frame(nframes - 1), np.float32), 0, 1), p


def hue_spread(a):
    x = (a[..., 0] - a[..., 1]).ravel()
    y = (a[..., 1] - a[..., 2]).ravel()
    m = (np.abs(x) + np.abs(y)) > 0.02
    if m.sum() < 50:
        return float("nan")
    ang = np.arctan2(y[m], x[m])
    r = np.sqrt(np.mean(np.cos(ang)) ** 2 + np.mean(np.sin(ang)) ** 2)
    return float(np.sqrt(max(0.0, -2.0 * np.log(max(r, 1e-9)))))


def report(a):
    lum = a @ COEF
    tot = max(1e-9, float(a.sum(axis=-1).mean()))
    m = a.reshape(-1, 3).mean(axis=0) / tot
    return dict(mean=float(lum.mean()), blown=float((lum > 0.97).mean()), g=float(m[1]),
                hue=hue_spread(a), hf=float(np.abs(np.diff(lum, axis=1)).mean()),
                ratio=(m[0], m[1], m[2]))


bl = report(render({})[0])
print("   the baseline (no night vision):  mean %.4f  blown %.5f  G %.4f  hue %.4f  hf %.5f"
      % (bl["mean"], bl["blown"], bl["g"], bl["hue"], bl["hf"]))
print("   the clip's green targets:        G 0.44-0.58,  hue 0.06-0.18,  ratio 0.29:0.46:0.25")
print()
print("   %-36s %-8s %-9s %-8s %-8s %-9s %s"
      % ("mode (schema defaults)", "mean", "blown", "G share", "hue", "hf", "ratio"))
for label, cam, nf in (
    ("off", {}, 1),
    ("1 NightShot", {"nv_mode": 1.0}, 1),
    ("2 Super NightShot (8 frames)", {"nv_mode": 2.0, "nv_accumulate": 8.0}, 8),
    ("3 Color Slow Shutter (8 frames)", {"nv_mode": 3.0, "nv_green_mix": 0.0, "nv_accumulate": 8.0}, 8),
):
    out, p = render(cam, nf)
    q = report(out)
    print("   %-36s %-8.4f %-9.5f %-8.4f %-8.4f %-9.5f %.3f:%.3f:%.3f"
          % (label, q["mean"], q["blown"], q["g"], q["hue"], q["hf"], *q["ratio"]))
    Image.fromarray((out * 255).astype(np.uint8)).save(
        os.path.join(NIGHT, "mode_%s.png" % label.split()[0]))

# the settings file, so the modes can be imported
CHOSEN = {"camera": dict(OFF, nv_mode=1.0), "format": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0},
          "player": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0},
          "codec": {"dv_enabled": 1.0}, "tape": {"head_switching": 0.0},
          "display": dict(RASTER_OFF, enabled=1.0, gamma=1.0, saturation=0.62)}
p = merge_values(BASE, CHOSEN)
diff = {k: {kk: vv for kk, vv in v.items() if BASE.get(k, {}).get(kk) != vv} for k, v in p.items()}
diff = {k: v for k, v in diff.items() if v}
import json                                                                  # noqa: E402
with open(os.path.join(NIGHT, "nightshot.json"), "w", encoding="utf-8") as fh:
    json.dump({"params": diff, "param_count": sum(len(v) for v in diff.values()),
               "seed": 20240101, "standard": "ntsc",
               "note": "nv_mode selects the function: 1 NightShot, 2 Super NightShot, 3 Color Slow Shutter. "
                       "The clip's green segments measure G share 0.44-0.58 and hue spread 0.06-0.18."},
              fh, ensure_ascii=False, indent=2)
print()
print("   wrote nightshot.json: %d parameter(s) across %d layer(s)"
      % (sum(len(v) for v in diff.values()), len(diff)))
for k, v in sorted(diff.items()):
    print("      %-9s %s" % (k, {kk: (round(vv, 3) if isinstance(vv, float) else vv)
                                for kk, vv in sorted(v.items())}))
