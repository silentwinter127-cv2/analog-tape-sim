"""Download the clean CC0 source with a proper User-Agent, and verify what arrived."""
from __future__ import annotations

import os
import urllib.request

UA = {"User-Agent": "analog-tape-sim/1.0 (research; commons thumbnail fetch)"}
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "ntscrs_ref")
os.makedirs(OUT, exist_ok=True)
DST = os.path.join(OUT, "clean_source.jpg")

CANDIDATES = [
    "https://upload.wikimedia.org/wikipedia/commons/thumb/0/03/"
    "Car_tire_tracks_in_snow_under_sodium_light.jpg/2200px-"
    "Car_tire_tracks_in_snow_under_sodium_light.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/0/03/"
    "Car_tire_tracks_in_snow_under_sodium_light.jpg",
]

ok = False
for url in CANDIDATES:
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=180) as r:
            data = r.read()
        if len(data) < 50000:
            print("   too small (%d bytes) from %s" % (len(data), url[:70]))
            continue
        with open(DST, "wb") as fh:
            fh.write(data)
        print("   downloaded %d bytes" % len(data))
        print("   from %s" % url[:100])
        ok = True
        break
    except Exception as exc:                                            # noqa: BLE001
        print("   failed on %s: %s" % (url[:70], exc))

if not ok:
    raise SystemExit("could not download the source")

from PIL import Image                                                    # noqa: E402
im = Image.open(DST)
print("   saved %s: %dx%d %s" % (os.path.relpath(DST), im.width, im.height, im.mode))

# and report the source's own statistics, which is the target the render is compared against
import numpy as np                                                       # noqa: E402
a = np.asarray(im.convert("RGB"), np.float32) / 255.0
COEF = np.array([0.299, 0.587, 0.114], np.float32)
lum = a @ COEF
print("   its own statistics, as the target:")
print("      mean luma %.4f, blown (>0.97) %.5f, dark (<0.02) %.4f"
      % (float(lum.mean()), float((lum > 0.97).mean()), float((lum < 0.02).mean())))
print("      mean colour %s (R-B %+.4f)" % (np.round(a.reshape(-1, 3).mean(axis=0), 4),
                                            float(a[..., 0].mean() - a[..., 2].mean())))
