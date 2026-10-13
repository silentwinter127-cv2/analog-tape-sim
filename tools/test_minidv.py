"""
Upload an untouched image, put it through MiniDV, and measure the colour at its blown highlights.

MiniDV is digital component -- 4:1:1 Rec.601 -- so unlike VHS it does not have to carry chroma on a subcarrier at
0.75 MHz. The user's reference photo shows exactly what that buys: the blown street lamp has **separate red, green
and blue streaks** fanning out of it, which is a much stronger lateral chromatic aberration than anything the model
currently produces. So the question this answers is quantitative rather than a matter of taste: how much colour is at
a blown highlight's edge in MiniDV, against the same measurement on the reference.

The statistic is the **edge-against-interior** colour difference, which is a fringe rather than a tint, plus the
per-channel radial profile that a spoke pattern would show. The reference image is measured the same way first, so
the target is a number rather than an impression.
"""
from __future__ import annotations

import io
import json
import os
import sys
import urllib.request
import uuid

import numpy as np
from PIL import Image

BASE = "http://127.0.0.1:8941"
COEF = np.array([0.299, 0.587, 0.114], np.float32)
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "ntscrs_ref")
os.makedirs(OUT, exist_ok=True)

SRC = sys.argv[1] if len(sys.argv) > 1 else None
if not SRC or not os.path.exists(SRC):
    print("usage: python tools/test_minidv.py <image>")
    raise SystemExit(2)


def get(path, timeout=120):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return json.loads(r.read())


def post(path, obj, timeout=180):
    req = urllib.request.Request(BASE + path, data=json.dumps(obj).encode(),
                                headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def upload(path):
    """Multipart upload, because that is what the server accepts."""
    boundary = "----atsim" + uuid.uuid4().hex
    with open(path, "rb") as fh:
        data = fh.read()
    name = os.path.basename(path)
    body = b"".join([
        ("--%s\r\n" % boundary).encode(),
        ('Content-Disposition: form-data; name="file"; filename="%s"\r\n' % name).encode(),
        b"Content-Type: image/jpeg\r\n\r\n", data, b"\r\n",
        ("--%s--\r\n" % boundary).encode(),
    ])
    req = urllib.request.Request(BASE + "/api/upload", data=body,
                                headers={"Content-Type": "multipart/form-data; boundary=" + boundary},
                                method="POST")
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def frame(scale=1.0):
    with urllib.request.urlopen(BASE + "/api/frame?t=0.0&scale=%s&q=95" % scale, timeout=240) as r:
        return np.asarray(Image.open(io.BytesIO(r.read())).convert("RGB"), np.float32) / 255.0


def highlights(img, count=3, border=60, sep=200, thresh=0.55):
    lum = img @ COEF
    m = lum.copy()
    m[:border] = 0
    m[-border:] = 0
    m[:, :border] = 0
    m[:, -border:] = 0
    h, w = lum.shape
    picked = []
    for idx in np.argsort(m.ravel())[::-1]:
        y, x = divmod(int(idx), w)
        if lum[y, x] < thresh:
            break
        if all(np.hypot(y - py, x - px) > sep for py, px in picked):
            picked.append((y, x))
        if len(picked) >= count:
            break
    return picked


def edge_colour(img, cy, cx, r_in=8, r_ring=(12, 22)):
    """R-B at the edge minus R-B in the core: a fringe, not a tint."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    core, ring = r < r_in, (r >= r_ring[0]) & (r < r_ring[1])
    if not core.any() or not ring.any():
        return float("nan")
    rb = img[..., 0] - img[..., 2]
    return float(rb[ring].mean() - rb[core].mean())


def spoke_strength(img, cy, cx, r0=14, r1=60, bins=180):
    """
    How much the colour varies *around* the highlight, which is what separated coloured streaks look like.

    A uniform fringe is constant in angle; streaks are not. So the standard deviation of R-B around the ring,
    normalised, is the statistic that distinguishes "a coloured edge" from "coloured spokes".
    """
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    m = (r >= r0) & (r < r1)
    if not m.any():
        return float("nan")
    ang = np.arctan2(yy - cy, xx - cx)[m]
    rb = (img[..., 0] - img[..., 2])[m]
    prof = np.full(bins, np.nan)
    idx = ((ang + np.pi) / (2 * np.pi) * bins).astype(int) % bins
    for b in range(bins):
        sel = idx == b
        if sel.any():
            prof[b] = rb[sel].mean()
    good = ~np.isnan(prof)
    if good.sum() < 8:
        return float("nan")
    return float(np.nanstd(np.interp(np.arange(bins), np.arange(bins)[good], prof[good], period=bins)))


print("=== the reference photo's own highlights, as the target ===")
ref = np.asarray(Image.open(SRC).convert("RGB"), np.float32) / 255.0
picked_ref = highlights(ref)
print("   its highlights: %s" % [(int(y), int(x)) for y, x in picked_ref])
for y, x in picked_ref:
    print("      at (%4d,%4d): edge colour %+.5f, spoke strength %.5f"
          % (y, x, edge_colour(ref, y, x), spoke_strength(ref, y, x)))

print()
print("=== uploading it untouched and selecting MiniDV ===")
up = upload(SRC)
print("   upload: %s" % {k: up.get(k) for k in ("ok", "name", "width", "height") if k in up})

cat = get("/api/catalog")
names = []
for key in ("presets", "factory_presets", "names", "items"):
    v = cat.get(key)
    if isinstance(v, list):
        names = [x.get("name") if isinstance(x, dict) else x for x in v]
        if names:
            print("   the catalog's %r holds %d entries" % (key, len(names)))
            break
    if isinstance(v, dict):
        names = list(v)
        print("   the catalog's %r holds %d entries" % (key, len(names)))
        break
if not names:
    print("   the catalog keys are: %s" % list(cat)[:10])
print("   the MiniDV entries offered: %s" % [n for n in names if n and "dv" in n.lower()])

for preset in ("minidv_clean", "minidv_shuttle"):
    if names and preset not in names:
        print("   %s is not offered by the server" % preset)
        continue
    post("/api/preset", {"name": preset})
    im = frame()
    picked = highlights(im)
    print()
    print("   --- %s, rendered %dx%d ---" % (preset, im.shape[1], im.shape[0]))
    if not picked:
        print("      no blown highlight found above 0.55")
        continue
    for y, x in picked:
        print("      at (%4d,%4d): edge colour %+.5f, spoke strength %.5f"
              % (y, x, edge_colour(im, y, x), spoke_strength(im, y, x)))
    Image.fromarray((np.clip(im, 0, 1) * 255).astype(np.uint8)).save(
        os.path.join(OUT, "minidv_%s.png" % preset))
