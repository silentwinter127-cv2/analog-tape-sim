"""Render the final settings at the calibrated value with the raster off, and report it against its control."""
from __future__ import annotations

import io
import json
import os
import urllib.request

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "tools", "ntscrs_ref")
BASE = "http://127.0.0.1:8941"
COEF = np.array([0.299, 0.587, 0.114], np.float32)


def get_raw(path, timeout=180):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return r.read()


def post(path, obj, timeout=900):
    req = urllib.request.Request(BASE + path, data=json.dumps(obj).encode(),
                                headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


COMMON = {
    "format": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0, "chroma_gain": 1.0,
               "color_noise": 0.0, "cross_color": 0.0, "cross_luma": 0.0, "edge_boost": 0.0},
    "player": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0,
               "color_noise": 0.0, "cross_color": 0.0, "cross_luma": 0.0},
    "codec": {"dv_enabled": 1.0},
    "tape": {"head_switching": 0.0},
    "display": {"enabled": 1.0, "gamma": 1.0, "saturation": 0.62,
                "scanlines": 0.0, "mask_depth": 0.0, "interlace_flicker": 0.0,
                "scan_pitch": 0.0, "spot_sigma": 0.0, "defocus": 0.0, "vignette": 0.0,
                "reflection": 0.0, "bloom": 0.0, "geometry_warp": 0.0, "overscan": 0.0,
                "curvature": 0.0, "breathing": 0.0, "crt_age": 0.0, "convergence": 0.0,
                "crt_moisture": 0.0, "crt_corrosion": 0.0},
}
BASE_CAM = {"exposure": 0.0, "color_cast": 0.0, "halation": 0.0, "ccd_smear": 0.0,
            "chroma_aberration": 0.0, "lens_flare": 0.0, "spike_spread": 0.0,
            "flare_defocus": 0.0, "longitudinal_ca": 0.0, "axis_offset": 0.0,
            "overflow_drain": 0.0, "source_glow": 0.0, "lens_fog": 0.0, "lens_wear": 0.0,
            "bloom_amount": 0.0, "dead_sites": 0.0}
CHOSEN = dict(source_glow=0.02, halation=0.18, ccd_smear=0.18, flare_defocus=0.30,
              lens_flare=0.12, spike_spread=0.18, overflow_drain=0.70,
              chroma_aberration=0.20, longitudinal_ca=0.12, axis_offset=0.20)
CONTROL = dict(chroma_aberration=0.20, longitudinal_ca=0.12, axis_offset=0.20)


def render(cam, name, scale=0.55):
    params = {k: dict(v) for k, v in COMMON.items()}
    params["camera"] = dict(BASE_CAM, **cam)
    post("/api/project", {"reset_params": True})
    post("/api/project", {"params": params, "seed": 20240101, "standard": "ntsc"})
    with urllib.request.urlopen(BASE + "/api/frame?t=0.0&scale=%s&q=96" % scale, timeout=900) as r:
        im = Image.open(io.BytesIO(r.read())).convert("RGB")
    im.save(os.path.join(OUT, name))
    x = np.asarray(im, np.float32) / 255.0
    L = x @ COEF
    return im, dict(mean=round(float(L.mean()), 4), blown=round(float((L > 0.97).mean()), 5),
                    glow=round(float((L > 0.85).mean()), 5),
                    rows=round(float(np.abs(np.diff(L.mean(axis=1))).mean()), 5),
                    fine=round(float(np.abs(np.diff(L, axis=0)).mean()), 5))


def near_br(img):
    x = np.asarray(img, np.float32)
    if x.max() > 1.5:
        x = x / 255.0
    L = x @ COEF
    k = max(5, L.shape[0] // 40)
    box = np.lib.stride_tricks.sliding_window_view(L, (k, k)).mean(axis=(2, 3))
    cy, cx = np.unravel_index(int(np.argmax(box)), box.shape)
    cy += k // 2
    cx += k // 2
    sc = L.shape[0] / 900.0
    r0, r1 = min(L.shape[0] - 1, cy + int(60 * sc)), min(L.shape[0], cy + int(140 * sc))
    c0, c1 = max(0, cx - int(30 * sc)), min(L.shape[1], cx + int(30 * sc))
    if r1 <= r0 or c1 <= c0:
        return float("nan")
    p = x[r0:r1, c0:c1].reshape(-1, 3).mean(axis=0)
    return float(p[2] - p[0])


im_c, m_c = render(CONTROL, "final-control.png")
im_e, m_e = render(CHOSEN, "final-effect.png")
print("   %-28s %-10s %-10s %-10s %-11s %s"
      % ("", "mean", "blown", "glow", "row var", "near B-R"))
print("   %-28s %-10.4f %-10.5f %-10.5f %-11.5f %+.4f"
      % ("control (light at zero)", m_c["mean"], m_c["blown"], m_c["glow"], m_c["rows"],
         near_br(im_c)))
print("   %-28s %-10.4f %-10.5f %-10.5f %-11.5f %+.4f"
      % ("effect (calibrated)", m_e["mean"], m_e["blown"], m_e["glow"], m_e["rows"],
         near_br(im_e)))

src = (r"C:\Users\Administrator\.dsh\attachments\v1\objects\20"
       r"\204363c0164fe605ff4b85d7d7f3b1d8d4940f98d85d8b0a2e401a0f2b703bd8")
print()
if os.path.exists(src):
    s = np.asarray(Image.open(src).convert("RGB"), np.float32) / 255.0
    sl = s @ COEF
    print("   the raw source:  mean %.4f  blown %.5f  glow %.5f  near B-R %+.4f"
          % (float(sl.mean()), float((sl > 0.97).mean()), float((sl > 0.85).mean()), near_br(s)))

print()
print("   the raster is off, so the per-row figures are the picture's own:")
print("      row-to-row of the row means: %.5f (it was 0.04268 with the raster on)" % m_e["rows"])
print("      per-row difference:         %.5f" % m_e["fine"])

params = {k: dict(v) for k, v in COMMON.items()}
params["camera"] = dict(BASE_CAM, **CHOSEN)
post("/api/project", {"reset_params": True})
post("/api/project", {"params": params, "seed": 20240101, "standard": "ntsc"})
cfg = json.loads(get_raw("/api/export"))
with open(os.path.join(OUT, "repro-highlight.json"), "w", encoding="utf-8") as fh:
    json.dump(cfg, fh, ensure_ascii=False, indent=2)
print()
print("   wrote repro-highlight.json: %d parameter(s) across %d layer(s)"
      % (cfg["param_count"], len(cfg["params"])))
for layer, vals in sorted(cfg["params"].items()):
    print("      %-9s %s" % (layer, {k: (round(v, 3) if isinstance(v, float) else v)
                                     for k, v in sorted(vals.items())}))
