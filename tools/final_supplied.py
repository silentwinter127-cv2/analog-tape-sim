"""
Upload the supplied photograph explicitly and render the tuned settings, so the output is of the right subject.

**A mistake worth recording.** The previous script relied on whatever source the server already had, and the fan test
had uploaded its own designed image last -- so the "final effect" it wrote was a render of **the fan test's green bar**,
not of the photograph. It happened to be informative, because at `chroma_aberration 1.0` that bar shows strong coloured
edges, which is the rainbow the report asked for. But it is not the requested output, and the fix is to upload the
source in the same script that renders it rather than assuming any state.

The settings carried over are the tuned ones, from the sweep on the photograph:

    chroma_aberration 1.0, longitudinal_ca 0.6, axis_offset 1.0

which raised the angular hue spread from **0.0245 to 0.2509** -- a factor of ten -- while the frame-wide per-pixel
|R-B| stayed at **0.00544**, i.e. small. That is the combination the earlier reduction had destroyed: the rainbow is
present **at the highlights** without the whole picture wearing red-green edges.
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
SRC = (r"C:\Users\Administrator\.dsh\attachments\v1\objects\20"
       r"\204363c0164fe605ff4b85d7d7f3b1d8d4940f98d85d8b0a2e401a0f2b703bd8")
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


def upload_raw(path):
    with open(path, "rb") as fh:
        data = fh.read()
    req = urllib.request.Request(
        BASE + "/api/upload", data=data, method="POST",
        headers={"Content-Type": "application/octet-stream",
                 "X-Filename": "supplied.jpg", "Content-Length": str(len(data))})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


print("   uploading the supplied photograph, in the same script that renders it")
print("      %s" % {k: v for k, v in upload_raw(SRC).items() if k in ("ok", "still")})

COMMON = {
    "format": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0, "chroma_gain": 1.0,
               "color_noise": 0.0, "cross_color": 0.0, "cross_luma": 0.0, "edge_boost": 0.0},
    "player": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0,
               "color_noise": 0.0, "cross_color": 0.0, "cross_luma": 0.0},
    "codec": {"dv_enabled": 1.0},
    "tape": {"head_switching": 0.0},
    "display": {"enabled": 1.0, "gamma": 1.0, "saturation": 0.62, "scanlines": 0.0,
                "mask_depth": 0.0, "interlace_flicker": 0.0, "scan_pitch": 0.0,
                "spot_sigma": 0.0, "defocus": 0.0, "vignette": 0.0, "reflection": 0.0,
                "bloom": 0.0, "geometry_warp": 0.0, "overscan": 0.0, "curvature": 0.0,
                "breathing": 0.0, "crt_age": 0.0, "convergence": 0.0, "crt_moisture": 0.0,
                "crt_corrosion": 0.0},
}
BASE_CAM = {"exposure": 0.0, "color_cast": 0.0, "halation": 0.0, "ccd_smear": 0.0,
            "chroma_aberration": 0.0, "lens_flare": 0.0, "spike_spread": 0.0,
            "flare_defocus": 0.0, "longitudinal_ca": 0.0, "axis_offset": 0.0,
            "overflow_drain": 0.0, "source_glow": 0.0, "lens_fog": 0.0, "lens_wear": 0.0,
            "bloom_amount": 0.0, "dead_sites": 0.0}
EFFECT = dict(source_glow=0.02, halation=0.18, ccd_smear=0.18, flare_defocus=0.30,
              lens_flare=0.12, spike_spread=0.18, overflow_drain=0.70,
              chroma_aberration=1.0, longitudinal_ca=0.6, axis_offset=1.0)
CONTROL = dict(chroma_aberration=1.0, longitudinal_ca=0.6, axis_offset=1.0)


def render(cam, name, scale=0.55):
    params = {k: dict(v) for k, v in COMMON.items()}
    params["camera"] = dict(BASE_CAM, **cam)
    post("/api/project", {"reset_params": True})
    post("/api/project", {"params": params, "seed": 20240101, "standard": "ntsc"})
    with urllib.request.urlopen(BASE + "/api/frame?t=0.0&scale=%s&q=96" % scale, timeout=900) as r:
        im = Image.open(io.BytesIO(r.read())).convert("RGB")
    im.save(os.path.join(OUT, name))
    a = np.asarray(im, np.float32) / 255.0
    L = a @ COEF
    return a, dict(mean=round(float(L.mean()), 4), blown=round(float((L > 0.97).mean()), 5),
                   glow=round(float((L > 0.85).mean()), 5),
                   fringe=round(float(np.abs(a[..., 0] - a[..., 2]).mean()), 5))


def fan(a):
    H, W = a.shape[:2]
    lum = a @ COEF
    k = max(5, H // 20)
    box = np.lib.stride_tricks.sliding_window_view(lum, (k, k)).mean(axis=(2, 3))
    cy, cx = np.unravel_index(int(np.argmax(box)), box.shape)
    cy += k // 2
    cx += k // 2
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    ang = np.degrees(np.arctan2(-(yy - cy), xx - cx)) % 360.0
    vals = []
    for rad in (80, 140, 220):
        rr = rad * (H / 900.0)
        m = (r >= rr - 10) & (r < rr + 10)
        if m.sum() < 60:
            vals.append(float("nan"))
            continue
        idx = (ang[m] / 10.0).astype(int) % 36
        rb = (a[..., 0] - a[..., 2])[m]
        b = np.array([rb[idx == j].mean() for j in range(36) if (idx == j).any()])
        vals.append(float(np.ptp(b)))
    return vals


ac, mc = render(CONTROL, "final-control.png")
ae, me = render(EFFECT, "final-effect.png")
print()
print("   %-24s %-9s %-9s %-9s %-10s %s"
      % ("", "mean", "blown", "glow", "|R-B|", "hue spread r=80/140/220"))
print("   %-24s %-9.4f %-9.5f %-9.5f %-10.5f %s"
      % ("control (light at zero)", mc["mean"], mc["blown"], mc["glow"], mc["fringe"],
         " ".join("%.4f" % v for v in fan(ac))))
print("   %-24s %-9.4f %-9.5f %-9.5f %-10.5f %s"
      % ("effect (tuned)", me["mean"], me["blown"], me["glow"], me["fringe"],
         " ".join("%.4f" % v for v in fan(ae))))

src = np.asarray(Image.open(SRC).convert("RGB"), np.float32) / 255.0
sl = src @ COEF
print()
print("   the raw source:          mean %.4f, blown %.5f, glow %.5f, |R-B| %.5f"
      % (float(sl.mean()), float((sl > 0.97).mean()), float((sl > 0.85).mean()),
         float(np.abs(src[..., 0] - src[..., 2]).mean())))

params = {k: dict(v) for k, v in COMMON.items()}
params["camera"] = dict(BASE_CAM, **EFFECT)
post("/api/project", {"reset_params": True})
post("/api/project", {"params": params, "seed": 20240101, "standard": "ntsc"})
cfg = json.loads(get_raw("/api/export"))
with open(os.path.join(OUT, "repro-highlight.json"), "w", encoding="utf-8") as fh:
    json.dump(cfg, fh, ensure_ascii=False, indent=2)
print()
print("   wrote repro-highlight.json: %d parameter(s)" % cfg["param_count"])
for layer, vals in sorted(cfg["params"].items()):
    print("      %-9s %s" % (layer, {k: (round(v, 3) if isinstance(v, float) else v)
                                     for k, v in sorted(vals.items())}))
