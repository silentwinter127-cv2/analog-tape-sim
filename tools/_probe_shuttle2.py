import sys
sys.path.insert(0, ".")
import numpy as np
import tools.testimage as ti
from atsim.core.profiles import FACTORY_PRESETS
from atsim.core.params import merge_values
from atsim.render.renderer import Renderer
from scipy.ndimage import median_filter

src = ti.SceneSource()
base = FACTORY_PRESETS["family_archive_1992"]["layers"]
# shuttle concealment lives on the DV branch: enable the codec layer
dv = merge_values(base, {"codec": {"dv_enabled": 1.0}})

def stale_fraction(rgb, block=16):
    H, W = rgb.shape[:2]
    bh, bw = H//block, W//block
    if bh < 3 or bw < 3: return 0.0
    a = rgb[:bh*block, :bw*block]
    tiles = a.reshape(bh, block, bw, block, 3).transpose(0,2,1,3,4).reshape(bh*bw, -1)
    G = tiles.mean(axis=1).reshape(bh, bw)
    med = median_filter(G, size=3, mode="nearest")
    odd = np.abs(G-med)
    thr = np.median(odd) + 4.0*np.median(np.abs(odd-np.median(odd))) + 1e-6
    return float((odd > thr).mean())

ref = None
print("%-26s %12s %12s" % ("setting", "stale blocks", "mean|d| vs off"))
for label, ov in (("off (normal play)", {"shuttle_speed": 0.0}),
                  ("2x, 19% stale", {"shuttle_speed": 2.0, "shuttle_stale": 0.19}),
                  ("5x, 19% stale", {"shuttle_speed": 5.0, "shuttle_stale": 0.19}),
                  ("5x, 35% stale", {"shuttle_speed": 5.0, "shuttle_stale": 0.35}),
                  ("5x, no clustering", {"shuttle_speed": 5.0, "shuttle_stale": 0.19,
                                          "shuttle_cluster": 0.0}),
                  ("5x, no carry", {"shuttle_speed": 5.0, "shuttle_stale": 0.19,
                                     "shuttle_carry": 0.0})):
    r = Renderer(src, src.width, src.height, src.fps,
                 merge_values(dv, {"tape": ov}), seed=7)
    r._build_chain()
    ts = r.stage["tape"]
    out = r.render_frame(4)
    if label.startswith("off"):
        print("   [digital branch = %s, params reach stage: speed=%.1f stale=%.2f]" % (
          ts.digital, ts.shuttle_speed, ts.shuttle_fraction))
    d = 0.0 if ref is None else float(np.abs(out-ref).mean())
    if ref is None: ref = out
    print("%-26s %11.1f%% %12.5f" % (label, stale_fraction(out)*100, d))
