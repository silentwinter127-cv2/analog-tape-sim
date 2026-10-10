import sys, time
sys.path.insert(0, ".")
import numpy as np
from atsim.core.profiles import FACTORY_PRESETS, PROFILE_LIBRARY, SCHEMA, resolve_preset
from atsim.render.renderer import Renderer
from atsim.io.source import SyntheticSource

src = SyntheticSource(width=360, height=240, frames=30)
print("rendering every factory preset end to end:")
ok = 0
for name, spec in FACTORY_PRESETS.items():
    params = resolve_preset(spec["layers"])
    try:
        r = Renderer(src, 360, 240, src.fps, params, seed=7)
        for i in range(6):
            out = r.render_frame(i)
        g = out @ np.array([0.299,0.587,0.114], np.float32)
        mx, mn = out.max(axis=2), out.min(axis=2)
        sat = float(np.mean(np.where(mx>1e-6,(mx-mn)/np.maximum(mx,1e-6),0)))
        print("  [OK] %-22s mean %.4f sat %.4f" % (name, out.mean(), sat))
        ok += 1
    except Exception as e:
        print("  [FAIL] %-22s %s" % (name, str(e)[:70]))
print()
print("%d/%d presets render" % (ok, len(FACTORY_PRESETS)))
# and the device profile libraries must resolve too
bad = []
for lib, items in PROFILE_LIBRARY.items():
    for pname, prof in items.items():
        try:
            resolve_preset(prof["layers"])
        except Exception as e:
            bad.append("%s.%s: %s" % (lib, pname, e))
print("profile libraries resolve:", "yes" if not bad else bad)
