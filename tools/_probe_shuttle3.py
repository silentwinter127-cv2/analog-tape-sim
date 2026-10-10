import sys
sys.path.insert(0, ".")
import numpy as np
import tools.testimage as ti
from atsim.core.profiles import FACTORY_PRESETS
from atsim.core.params import merge_values
from atsim.render.renderer import Renderer
from PIL import Image

src = ti.SceneSource()
base = FACTORY_PRESETS["family_archive_1992"]["layers"]
dv = merge_values(base, {"codec": {"dv_enabled": 1.0}})
tiles = []
for label, ov in (("normal", {"shuttle_speed": 0.0}),
                  ("2x 19%", {"shuttle_speed": 2.0, "shuttle_stale": 0.19}),
                  ("5x 19%", {"shuttle_speed": 5.0, "shuttle_stale": 0.19}),
                  ("5x 35% clustered", {"shuttle_speed": 5.0, "shuttle_stale": 0.35}),
                  ("5x 19% rewind", {"shuttle_speed": 5.0, "shuttle_stale": 0.19,
                                     "shuttle_reverse": 1.0})):
    r = Renderer(src, src.width, src.height, src.fps,
                 merge_values(dv, {"tape": ov}), seed=7)
    tiles.append(r.render_frame(4).copy())
Image.fromarray((np.clip(np.concatenate(tiles, axis=1),0,1)*255).astype(np.uint8)).save(
  "tools/out/shuttle_sheet.png")
print("wrote tools/out/shuttle_sheet.png (normal | 2x19 | 5x19 | 5x35 | rewind)")
# also confirm the FIVE panels differ from each other
for i in range(1, len(tiles)):
    print("  panel %d vs normal: mean|d| = %.5f" % (i, np.abs(tiles[i]-tiles[0]).mean()))
