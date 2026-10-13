"""
Run the reproduction on the supplied photograph, and report the profile rather than judging it.

**What its measurement says before anything is applied.** Its fixture spans **22.2% of the frame's height** and
**12.4% of its width**, and its saturation reaches **252 px below the core -- 0.164 of the height, which is about 148
rows in the reference's 900-row terms**. So it has the extent the reference's profile needs, and the earlier
photographs did not.

**But its own colour below the core is positive in B-R throughout** (+0.233 at 40 rows, +0.187 at 100, +0.158 at 180),
i.e. **cool**, where the reference is **-0.510 at 100 rows**, i.e. strongly warm. That is a statement about the
subject: the reference's lamp sits inside a large continuous warm glow, while this fixture lights a pale wall below
itself and the wall reads cool. **A scattering mechanism carries the source's colour, so if the source and its
surroundings are cool there, the halo will be cool** -- and that is the correct behaviour, not a failure.

So the effect is run at the calibrated strength and the profile is reported next to the reference's, with the
comparison stated in terms of what each number means rather than whether it "looks right".
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


print("   uploading the supplied photograph as a raw source")
print("      %s" % {k: v for k, v in upload_raw(SRC).items() if k in ("ok", "still")})

COMMON = {
    "format": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0, "chroma_gain": 1.0,
               "color_noise": 0.0, "cross_color": 0.0, "cross_luma": 0.0, "edge_boost": 0.0},
    "player": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0,
               "color_noise": 0.0, "cross_color": 0.0, "cross_luma": 0.0},
    "codec": {"dv_enabled": 1.0},
    "tape": {"head_switching": 0.0},
    "display": {"enabled": 1.0, "gamma": 1.0, "saturation": 0.62},
}
BASE_CAM = {"exposure": 0.0, "color_cast": 0.0, "halation": 0.0, "ccd_smear": 0.0,
            "chroma_aberration": 0.0, "lens_flare": 0.0, "spike_spread": 0.0,
            "flare_defocus": 0.0, "longitudinal_ca": 0.0, "axis_offset": 0.0,
            "overflow_drain": 0.0, "source_glow": 0.0, "lens_fog": 0.0, "lens_wear": 0.0,
            "bloom_amount": 0.0, "dead_sites": 0.0}
EFFECT = {"source_glow": 0.85, "halation": 0.85, "ccd_smear": 0.80, "flare_defocus": 1.00,
          "lens_flare": 0.50, "spike_spread": 0.85, "overflow_drain": 0.35,
          "chroma_aberration": 0.20, "longitudinal_ca": 0.12, "axis_offset": 0.20}


def apply(camera):
    params = {k: dict(v) for k, v in COMMON.items()}
    params["camera"] = dict(BASE_CAM, **camera)
    post("/api/project", {"reset_params": True})
    post("/api/project", {"params": params, "seed": 20240101, "standard": "ntsc"})


def render(name, scale=0.55):
    with urllib.request.urlopen(BASE + "/api/frame?t=0.0&scale=%s&q=96" % scale, timeout=900) as r:
        im = Image.open(io.BytesIO(r.read())).convert("RGB")
    im.save(os.path.join(OUT, name))
    x = np.asarray(im, np.float32) / 255.0
    L = x @ COEF
    return im, dict(mean=round(float(L.mean()), 4), blown=round(float((L > 0.97).mean()), 5),
                    glow=round(float((L > 0.85).mean()), 5),
                    rb=round(float(x[..., 0].mean() - x[..., 2].mean()), 5))


apply({})
im_c, m_c = render("sup-control.png")
print()
print("   control  %s" % m_c)
apply(EFFECT)
im_e, m_e = render("sup-effect.png")
print("   effect   %s" % m_e)
print("   the effect against its control:")
for k in ("mean", "blown", "glow", "rb"):
    print("      %-6s %.5f -> %.5f  (%+.5f)" % (k, m_c[k], m_e[k], m_e[k] - m_c[k]))


def prof(img, rows=(0, 40, 60, 100, 160, 180)):
    H, W = img.shape[:2]
    lum = img @ COEF
    k = max(5, H // 40)
    box = np.lib.stride_tricks.sliding_window_view(lum, (k, k)).mean(axis=(2, 3))
    cy, cx = np.unravel_index(int(np.argmax(box)), box.shape)
    cy += k // 2
    cx += k // 2
    sc = H / 900.0
    out = []
    for d in rows:
        dd = int(d * sc)
        r0, r1 = min(H - 1, cy + dd), min(H, cy + dd + max(3, int(20 * sc)))
        c0, c1 = max(0, cx - max(3, int(6 * sc))), min(W, cx + max(4, int(7 * sc)))
        if r1 <= r0 or c1 <= c0:
            out.append(float("nan"))
            continue
        p = img[r0:r1, c0:c1].reshape(-1, 3).mean(axis=0)
        out.append(float(p[2] - p[0]))
    return out


print()
print("   the profile below the core, B-R:")
print("      %-16s %s" % ("the reference", "  ".join("+%-5d" % d for d in (0, 40, 60, 100, 160, 180))))
print("      %-16s %s" % ("", "  ".join("%+.3f" % v for v in
                                        [0.000, -0.009, -0.170, -0.510, 0.178, 0.226])))
for tag, name in (("this source", None),):
    b = np.asarray(Image.open(SRC).convert("RGB"), np.float32) / 255.0
    print("      %-16s %s" % ("(the raw file)", "  ".join("%+.3f" % v for v in prof(b))))
for tag, name in (("control", "sup-control.png"), ("effect", "sup-effect.png")):
    b = np.asarray(Image.open(os.path.join(OUT, name)).convert("RGB"), np.float32) / 255.0
    print("      %-16s %s" % (tag, "  ".join("%+.3f" % v for v in prof(b))))

apply(EFFECT)
cfg = json.loads(get_raw("/api/export"))
with open(os.path.join(OUT, "repro-highlight.json"), "w", encoding="utf-8") as fh:
    json.dump(cfg, fh, ensure_ascii=False, indent=2)
print()
print("   wrote repro-highlight.json: %d parameter(s) across %d layer(s)"
      % (cfg["param_count"], len(cfg["params"])))
