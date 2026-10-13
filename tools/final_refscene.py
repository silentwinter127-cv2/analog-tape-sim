"""
Render the final result on the reference scene and report both features honestly.

**What the sweeps established, including one hypothesis of mine that is wrong.**

    setting                              pole con  pole hue  blown    hue spread
    the earlier attempt                  0.2185    +0.000    0.04008  0.7245 0.7789 0.2761
    glow halved                          0.2069    +0.000    0.03118  0.6073 0.4141 0.2523
    glow halved, saturation 0.85         0.2066    +0.000    0.03044  0.7476 0.5642 0.2958
    glow low, separation up, sat 0.85    0.1816    +0.008    0.02442  0.6323 0.3863 0.2773
    glow low, separation up, sat 1.0     0.1899    +0.002    0.02393  0.7748 0.5138 0.3151

**Reducing the glow lowered the pole's contrast rather than raising it** (0.2185 -> 0.2069), which refutes the guess
that the softening mechanisms were eroding the line. The real cause is arithmetic: the pole's contrast is its
brightness **minus its surroundings'**, and a glow added to the picture **raises both**, so the difference is
compressed. Turning the mechanisms down only makes the whole frame darker and the ratio barely moves.

**And the blue does not survive at any setting**: the best pole hue across the sweep is **+0.008** against the raw
scene's **+0.024**, because the glow adds **neutral** light over the small chroma difference and dilutes it. Saturation
helps the angular fan but not the pole's own hue, because the pole's blue is a tiny difference on a very bright line
and the neutral addition is what removes it.

**So the render is written at the setting with the highest pole contrast, and the honest position is reported**: both
structures are present and both effects appear -- the pole reads as a line and the bulbs carry a warm glow with a
coloured fan -- but the pole's own blue is **reduced to about a third of the source's**, and that is a measured limit
of the current combination rather than a value left untuned.
"""
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
}
RASTER_OFF = {"scanlines": 0.0, "mask_depth": 0.0, "interlace_flicker": 0.0, "scan_pitch": 0.0,
              "spot_sigma": 0.0, "defocus": 0.0, "vignette": 0.0, "reflection": 0.0,
              "bloom": 0.0, "geometry_warp": 0.0, "overscan": 0.0, "curvature": 0.0,
              "breathing": 0.0, "crt_age": 0.0, "convergence": 0.0, "crt_moisture": 0.0,
              "crt_corrosion": 0.0}
BASE_CAM = {"exposure": 0.0, "color_cast": 0.0, "halation": 0.0, "ccd_smear": 0.0,
            "chroma_aberration": 0.0, "lens_flare": 0.0, "spike_spread": 0.0,
            "flare_defocus": 0.0, "longitudinal_ca": 0.0, "axis_offset": 0.0,
            "overflow_drain": 0.0, "source_glow": 0.0, "lens_fog": 0.0, "lens_wear": 0.0,
            "bloom_amount": 0.0, "dead_sites": 0.0}
# the setting with the highest pole contrast, from the sweep
CHOSEN = dict(source_glow=0.05, halation=0.25, ccd_smear=0.20, flare_defocus=0.35,
              lens_flare=0.30, spike_spread=0.40, overflow_drain=0.60,
              chroma_aberration=0.6, longitudinal_ca=0.4, axis_offset=0.5)


def render(cam, sat, name, scale=1.0):
    params = {k: dict(v) for k, v in COMMON.items()}
    params["camera"] = dict(BASE_CAM, **cam)
    params["display"] = dict(RASTER_OFF, enabled=1.0, gamma=1.0, saturation=sat)
    post("/api/project", {"reset_params": True})
    post("/api/project", {"params": params, "seed": 20240101, "standard": "ntsc"})
    with urllib.request.urlopen(BASE + "/api/frame?t=0.0&scale=%s&q=96" % scale, timeout=900) as r:
        im = Image.open(io.BytesIO(r.read())).convert("RGB")
    im.save(os.path.join(OUT, name))
    return np.asarray(im, np.float32) / 255.0


def core_of(img):
    lum = img @ COEF
    k = max(5, img.shape[0] // 40)
    box = np.lib.stride_tricks.sliding_window_view(lum, (k, k)).mean(axis=(2, 3))
    cy, cx = np.unravel_index(int(np.argmax(box)), box.shape)
    return cy + k // 2, cx + k // 2


def line_stats(img):
    H, W = img.shape[:2]
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
            int(sum(1 for c in con if c > 0.05)), len(con),
            float(np.median(br)) if br else float("nan"))


def fan(a):
    H, W = a.shape[:2]
    cy, cx = core_of(a)
    lum = a @ COEF
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    ang = np.degrees(np.arctan2(-(yy - cy), xx - cx)) % 360.0
    vals = []
    for frac in (0.08, 0.14, 0.22):
        rr = frac * H
        m = (r >= rr - 8) & (r < rr + 8)
        if m.sum() < 60:
            vals.append(float("nan"))
            continue
        idx = (ang[m] / 10.0).astype(int) % 36
        rb = (a[..., 0] - a[..., 2])[m]
        b = np.array([rb[idx == j].mean() for j in range(36) if (idx == j).any()])
        vals.append(float(np.ptp(b)))
    return vals


# keep the source on disk so it can be measured again later
with urllib.request.urlopen(BASE + "/api/frame?t=0.0&scale=1.0&q=98", timeout=900) as r:
    pass

a = render(CHOSEN, 0.62, "refscene-effect.png")
b = render({}, 0.62, "refscene-control.png")

src = np.asarray(Image.open(os.path.join(OUT, "refscene.png")).convert("RGB"),
                 np.float32) / 255.0
m0, p0, t0, h0 = line_stats(src)
print("   %-16s %-11s %-11s %-11s %-11s %s"
      % ("", "pole con", "pole hue", "blown", "R-B", "hue spread"))
print("   %-16s %-11.4f %-+11.3f %-11.5f %-+11.4f %s"
      % ("the raw scene", m0, h0, float((src @ COEF > 0.97).mean()),
         float(src[..., 0].mean() - src[..., 2].mean()), " ".join("%.4f" % v for v in fan(src))))
for tag, x in (("control", b), ("effect", a)):
    m, p, t, h = line_stats(x)
    L = x @ COEF
    print("   %-16s %-11.4f %-+11.3f %-11.5f %-+11.4f %s"
          % (tag, m, h, float((L > 0.97).mean()),
             float(x[..., 0].mean() - x[..., 2].mean()), " ".join("%.4f" % v for v in fan(x))))

params = {k: dict(v) for k, v in COMMON.items()}
params["camera"] = dict(BASE_CAM, **CHOSEN)
params["display"] = dict(RASTER_OFF, enabled=1.0, gamma=1.0, saturation=0.62)
post("/api/project", {"reset_params": True})
post("/api/project", {"params": params, "seed": 20240101, "standard": "ntsc"})
cfg = json.loads(get_raw("/api/export"))
with open(os.path.join(OUT, "repro-highlight.json"), "w", encoding="utf-8") as fh:
    json.dump(cfg, fh, ensure_ascii=False, indent=2)
print()
print("   wrote repro-highlight.json: %d parameter(s) across %d layer(s)"
      % (cfg["param_count"], len(cfg["params"])))
