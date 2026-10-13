"""
Update the count in the lens-mechanism check, and measure `source_glow` against the reference's profile.

**The check's expectation, not its intent.** It asserts that the camera calls each lens mechanism exactly once, which
is the property that matters -- a duplicated call applies a mechanism twice. The "7" was the number of mechanisms when
it was written and there are now **8**, so the count is updated and the assertion itself is left alone. The relative
order of the three that existed before is asserted separately and still holds.

**And the measurement that decides whether the new mechanism closes the gap.** The target is the reference's own
profile below a blown core: at 60 and 100 rows it runs to **B-R -0.170 and -0.510**, where the reproduction reached
-0.014 and -0.034. The render's profile is read the same way, with `source_glow` at a range of strengths, so the value
that reaches the reference is found by measurement rather than chosen.
"""
from __future__ import annotations

import ast
import io
import json
import os
import urllib.request

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------- 2. the profile against the reference
BASE = "http://127.0.0.1:8941"
COEF = np.array([0.299, 0.587, 0.114], np.float32)
OUT = os.path.join(ROOT, "tools", "ntscrs_ref")
REF = (r"C:\Users\Administrator\.dsh\attachments\v1\objects\03"
       r"\039d9af408f07923375d9fae0a7fb6839668f20b2c0508face5b0a2a14b56eb9")


def post(path, obj, timeout=900):
    req = urllib.request.Request(BASE + path, data=json.dumps(obj).encode(),
                                headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def frame(scale=0.5):
    with urllib.request.urlopen(BASE + "/api/frame?t=0.0&scale=%s&q=96" % scale, timeout=900) as r:
        return np.asarray(Image.open(io.BytesIO(r.read())).convert("RGB"), np.float32) / 255.0


def prof(img, rows=(0, 40, 60, 100, 160, 180)):
    H, W = img.shape[:2]
    lum = img @ COEF
    k = max(9, H // 30)
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
            out.append((d, float("nan")))
            continue
        p = img[r0:r1, c0:c1].reshape(-1, 3).mean(axis=0)
        out.append((d, float(p[2] - p[0])))
    return out


ref = np.asarray(Image.open(REF).convert("RGB"), np.float32) / 255.0
rp = prof(ref)
print()
print("   the reference's B-R below the core: %s"
      % "  ".join("+%d:%+.3f" % (d, v) for d, v in rp))
print()
print("   %-14s %s" % ("source_glow", "  ".join("+%-5d" % d for d, _ in rp)))
CONFIG = {
    "camera": {"exposure": 0.0, "color_cast": 0.0, "halation": 0.0, "ccd_smear": 0.0,
               "chroma_aberration": 0.0, "lens_flare": 0.0, "spike_spread": 0.0,
               "flare_defocus": 0.0, "longitudinal_ca": 0.0, "axis_offset": 0.0,
               "overflow_drain": 0.0, "lens_fog": 0.0, "lens_wear": 0.0,
               "bloom_amount": 0.0, "dead_sites": 0.0},
    "format": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0, "chroma_gain": 1.0,
               "color_noise": 0.0, "cross_color": 0.0, "cross_luma": 0.0, "edge_boost": 0.0},
    "player": {"chroma_bandwidth": 2.0, "luma_bandwidth": 5.0,
               "color_noise": 0.0, "cross_color": 0.0, "cross_luma": 0.0},
    "codec": {"dv_enabled": 1.0},
    "tape": {"head_switching": 0.0},
    "display": {"enabled": 1.0, "gamma": 1.0, "saturation": 0.60},
}
for g in (0.0, 0.3, 0.6, 0.9):
    params = {k: dict(v) for k, v in CONFIG.items()}
    params["camera"] = dict(CONFIG["camera"], source_glow=g)
    post("/api/project", {"reset_params": True})
    post("/api/project", {"params": params, "seed": 20240101, "standard": "ntsc"})
    got = prof(frame())
    print("   %-14.2f %s" % (g, "  ".join("%+.3f" % v for _, v in got)))
