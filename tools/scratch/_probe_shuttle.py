import sys
sys.path.insert(0, ".")
import numpy as np
import tools.testimage as ti
from atsim.core.profiles import FACTORY_PRESETS, SCHEMA
from atsim.core.params import merge_values
from atsim.render.renderer import Renderer
from PIL import Image
from scipy.ndimage import median_filter

src = ti.SceneSource()
base = FACTORY_PRESETS["family_archive_1992"]["layers"]

def stale_fraction(rgb, block=16):
    a = rgb
    H, W = a.shape[:2]
    bh, bw = H//block, W//block
    if bh < 3 or bw < 3: return 0.0
    a = a[:bh*block, :bw*block]
    tiles = a.reshape(bh, block, bw, block, 3).transpose(0,2,1,3,4).reshape(bh*bw, -1)
    G = tiles.mean(axis=1).reshape(bh, bw)
    med = median_filter(G, size=3, mode="nearest")
    odd = np.abs(G - med)
    thr = np.median(odd) + 4.0*np.median(np.abs(odd-np.median(odd))) + 1e-6
    return float((odd > thr).mean())

print("=== shuttle: measured stale fraction vs the control ===")
print("%-26s %12s %10s" % ("setting", "stale blocks", "mean|Δ| vs off"))
ref = None
for label, ov in (("off (normal play)", {"shuttle_speed": 0.0}),
                  ("2x shuttle, 19% stale", {"shuttle_speed": 2.0, "shuttle_stale": 0.19}),
                  ("5x shuttle, 19% stale", {"shuttle_speed": 5.0, "shuttle_stale": 0.19}),
                  ("5x, 35% stale", {"shuttle_speed": 5.0, "shuttle_stale": 0.35}),
                  ("5x, no clustering", {"shuttle_speed": 5.0, "shuttle_stale": 0.19,
                                         "shuttle_cluster": 0.0})):
    lay = merge_values(base, {"tape": ov})
    r = Renderer(src, src.width, src.height, src.fps, lay, seed=7)
    out = r.render_frame(4)
    if ref is None:
        ref = out
        d = 0.0
    else:
        d = float(np.abs(out-ref).mean())
    print("%-26s %11.1f%% %10.5f" % (label, stale_fraction(out)*100, d))
Image.fromarray((np.clip(np.concatenate([
  Renderer(src, src.width, src.height, src.fps,
           merge_values(base, {"tape": {"shuttle_speed": 0.0}}), seed=7).render_frame(4),
  Renderer(src, src.width, src.height, src.fps,
           merge_values(base, {"tape": {"shuttle_speed": 5.0, "shuttle_stale": 0.19}}), seed=7).render_frame(4),
], axis=1),0,1)*255).astype(np.uint8)).save("tools/out/shuttle_ab.png")
print("wrote tools/out/shuttle_ab.png (left = normal, right = shuttle)")
