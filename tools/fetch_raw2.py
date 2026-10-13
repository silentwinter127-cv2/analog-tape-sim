"""
Fetch the raw candidates with spacing and retries, and evaluate whether each is usable.

Commons rate-limited the previous run after the hundred search requests that preceded it, so this one waits between
requests and backs off on a 429. Nothing about the goal changes: a **raw** photograph that actually **contains a light
source**, and the measurement that decides it is the fraction of pixels at or near white -- because a light source is
what that looks like when the exposure is set for the scene around it, and a raw frame with no saturated pixels gives
every highlight mechanism in the model nothing to act on.
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "tools", "ntscrs_ref")
UA = {"User-Agent": "analog-tape-sim/1.0 (research; raw night source; contact via repository)"}
COEF = np.array([0.299, 0.587, 0.114], np.float32)

WANT = [
    ("File:Snow under a street lamp in Tuntorp.jpg", "raw_lamp_snow.jpg"),
    ("File:Lamp at night at Brastad Arena.jpg", "raw_lamp_arena.jpg"),
    ("File:Street lights along a road in Tuntorp.jpg", "raw_lights_road.jpg"),
    ("File:Siena Streets at Night.jpg", "raw_siena.jpg"),
]


def get(url, tries=4, timeout=300):
    """A GET that waits and backs off, because Commons rate-limits a burst."""
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < tries - 1:
                wait = 6.0 * (attempt + 1)
                print("      rate-limited, waiting %.0fs" % wait)
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("gave up on %s" % url[:70])


def thumb_url(title, width=2400):
    api = "https://commons.wikimedia.org/w/api.php"
    p = {"action": "query", "titles": title, "prop": "imageinfo",
         "iiprop": "url|size|extmetadata", "iiurlwidth": width, "format": "json"}
    d = json.loads(get(api + "?" + urllib.parse.urlencode(p), timeout=90))
    page = list(d["query"]["pages"].values())[0]
    ii = page["imageinfo"][0]
    md = ii.get("extmetadata") or {}
    return (ii.get("thumburl") or ii.get("url"),
            (md.get("LicenseShortName") or {}).get("value", "?"),
            (md.get("Artist") or {}).get("value", "?"))


print("   %-22s %-11s %-12s %-11s %-10s %s"
      % ("file", "size", "blown>0.97", "glow>0.85", "mean", "verdict"))
saved = []
for title, name in WANT:
    dest = os.path.join(OUT, name)
    try:
        if os.path.exists(dest) and os.path.getsize(dest) > 50000:
            n = os.path.getsize(dest)
            url, lic, artist = thumb_url(title)
            time.sleep(2.0)
        else:
            url, lic, artist = thumb_url(title)
            time.sleep(2.0)
            data = get(url)
            with open(dest, "wb") as fh:
                fh.write(data)
            n = len(data)
        im = Image.open(dest).convert("RGB")
        a = np.asarray(im, np.float32) / 255.0
        lum = a @ COEF
        blown = float((lum > 0.97).mean())
        glow = float((lum > 0.85).mean())
        mean = float(lum.mean())
        ok = blown > 0.0002 and 0.005 < mean < 0.7
        print("   %-22s %-11s %-12.5f %-11.5f %-10.4f %s"
              % (name, "%dx%d" % (im.width, im.height), blown, glow, mean,
                 "USABLE" if ok else ("too dark" if mean <= 0.005 else
                                      ("no blowout" if blown <= 0.0002 else "too bright"))))
        saved.append(dict(name=name, title=title, licence=lic, artist=artist,
                          w=im.width, h=im.height, blown=blown, glow=glow, mean=mean, ok=ok))
    except Exception as exc:                                                # noqa: BLE001
        print("   %-22s FAILED: %s" % (name, exc))
    time.sleep(3.0)

usable = [s for s in saved if s["ok"]]
print()
print("   %d of %d usable" % (len(usable), len(saved)))
for s in usable:
    print("      %-22s %-14s %-30s blown %.5f glow %.5f mean %.4f"
          % (s["name"], s["licence"], s["artist"][:30], s["blown"], s["glow"], s["mean"]))
with open(os.path.join(OUT, "raw_sources.json"), "w", encoding="utf-8") as fh:
    json.dump(saved, fh, ensure_ascii=False, indent=1)
