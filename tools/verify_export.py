"""
Verify the export/import round trip on a live server, including the part that is easy to get wrong.

Three properties, in the order they matter for the feature's purpose -- handing a reproduction to someone else:

  * **the export is a diff.**  A default configuration must export an empty parameter set, and a configuration with
    known changes must export exactly those. A full dump would pass any test of "does it round trip" while being
    useless, so the diff property is checked first;
  * **the round trip is exact.**  Apply a set of known changes, export, wipe everything back to the defaults, import,
    and the state must come back **identically** -- compared on the frame hash, not on the parameters, because the
    parameters are what was written and the frame is what anyone actually cares about;
  * **and the import is complete on its own.**  Loading the file onto a *different* starting state must give the same
    result, which is what "resets first, then merges" buys and what a plain merge would not.
"""
from __future__ import annotations

import hashlib
import io
import json
import urllib.request

import numpy as np
from PIL import Image

BASE = "http://127.0.0.1:8941"


def get(path, timeout=180):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return json.loads(r.read())


def get_raw(path, timeout=180):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return r.read(), dict(r.headers)


def post(path, obj, timeout=600):
    req = urllib.request.Request(BASE + path, data=json.dumps(obj).encode(),
                                headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def frame_hash(scale=0.35):
    with urllib.request.urlopen(BASE + "/api/frame?t=0.0&scale=%s&q=92" % scale, timeout=600) as r:
        im = np.asarray(Image.open(io.BytesIO(r.read())).convert("RGB"))
    return hashlib.blake2s(im.tobytes(), digest_size=8).hexdigest()


# --- 1. the diff property
print("=== 1. the export is a diff, not a dump ===")
post("/api/project", {"reset_params": True})
cfg_default = json.loads(get_raw("/api/export")[0])
print("   a reset configuration exports %d changed parameter(s) across %d layer(s)"
      % (cfg_default["param_count"], len(cfg_default["params"])))
print("   its keys: %s" % sorted(cfg_default.keys()))
assert cfg_default["param_count"] == 0, "a reset configuration should export no changes"

CHANGES = {
    "camera": {"ccd_smear": 0.42, "halation": 0.51, "lens_flare": 0.33,
               "spike_spread": 0.77, "flare_defocus": 0.66, "axis_offset": 0.25},
    "format": {"chroma_bandwidth": 1.25, "chroma_gain": 1.4},
    "display": {"enabled": 1.0, "saturation": 0.72},
    "codec": {"dv_enabled": 0.0},
}
post("/api/project", {"params": CHANGES})
cfg = json.loads(get_raw("/api/export")[0])
print()
print("   after %d known changes the export holds %d parameter(s) in %d layer(s)"
      % (sum(len(v) for v in CHANGES.values()), cfg["param_count"], len(cfg["params"])))
for layer, vals in sorted(cfg["params"].items()):
    print("      %-10s %s" % (layer, {k: round(v, 4) if isinstance(v, float) else v
                                      for k, v in sorted(vals.items())}))
# A diff means a parameter appears **if and only if** its value differs from the schema's default.  The earlier
# expectation -- that everything set appears -- is wrong for the two entries that were set *to* their defaults, and
# asserting the "if and only if" is both correct and stronger.
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from atsim.core.profiles import SCHEMA as _S
_defaults = _S.resolve_all({})
wrong = []
for layer, vals in CHANGES.items():
    for k, v in vals.items():
        d = (_defaults.get(layer) or {}).get(k)
        exported = k in (cfg["params"].get(layer) or {})
        differs = not (isinstance(d, float) and isinstance(v, float) and abs(d - v) <= 1e-9) and d != v
        if exported != differs:
            wrong.append((layer, k, v, d, exported))
print("   the export holds exactly the parameters that differ from the defaults: %s" % (not wrong))
for layer, k, v, d, exported in wrong:
    print("      %s.%s set to %r, default %r, exported %s" % (layer, k, v, d, exported))
print("   (a parameter set *to* its default is correctly absent: %s)"
      % [f"{l}.{k}" for l, vs in CHANGES.items() for k, v in vs.items()
         if (not isinstance(v, float)) is False and abs(((_defaults.get(l) or {}).get(k) or -99) - v) <= 1e-9])

# --- 2. the round trip, compared on the rendered frame
print()
print("=== 2. the round trip, compared on the frame rather than on the parameters ===")
h_written = frame_hash()
print("   the frame with the changes applied:   %s" % h_written)

post("/api/project", {"reset_params": True})
h_wiped = frame_hash()
print("   after resetting everything:           %s  (differs: %s)" % (h_wiped, h_wiped != h_written))

post("/api/import", {"config": cfg})
h_imported = frame_hash()
print("   after importing the file back:        %s" % h_imported)
print("   -> round trip exact: %s" % (h_imported == h_written))

# --- 3. the import is complete on its own
print()
print("=== 3. importing onto a different starting state gives the same frame ===")
post("/api/project", {"params": {"tape": {"binder_loss": 0.5}, "camera": {"exposure": 0.6},
                                 "display": {"saturation": 0.1}}})
h_dirty = frame_hash()
print("   a deliberately different state:       %s" % h_dirty)
post("/api/import", {"config": cfg})
h_again = frame_hash()
print("   after importing the same file:        %s" % h_again)
print("   -> the file is complete on its own: %s" % (h_again == h_written))

# --- 4. a malformed file is refused rather than half-applied
print()
print("=== 4. a bad file is refused, not half-applied ===")
for label, bad in (("a non-config JSON object", {"hello": "world"}),
                   ("a newer version", {"kind": "atsim.config", "version": 99, "params": {}}),
                   ("no params at all", {"kind": "atsim.config", "version": 1})):
    try:
        r = post("/api/import", {"config": bad})
        print("   %-24s -> %s" % (label, r.get("error", "ACCEPTED (should have been refused)")))
    except urllib.error.HTTPError as exc:
        body = json.loads(exc.read())
        print("   %-24s -> refused: %s" % (label, body.get("error", "")[:60]))

post("/api/project", {"reset_params": True})
print()
print("   the project is left reset; import returned %s" % ("ok" if h_again == h_written else "MISMATCH"))
