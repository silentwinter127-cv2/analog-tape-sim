import sys
sys.path.insert(0, ".")
import numpy as np
from atsim.core.profiles import FACTORY_PRESETS
from atsim.core.params import merge_values
from atsim.core.standards import get_standard
from atsim.render.renderer import Renderer
from PIL import Image

class MovingSource:
    """A clip that actually moves, so a substituted block is visibly wrong."""
    def __init__(self, width=360, height=240, frames=120, fps=25.0):
        self.width, self.height, self.frame_count, self.fps = width, height, frames, fps
    def frame(self, i):
        h, w = self.height, self.width
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        x = xx + i * 9.0          # fast horizontal pan
        y = yy + i * 4.0
        # high-contrast moving texture: vertical bars + a diagonal, so a stale
        # block is obviously offset from its neighbours
        r = 0.5 + 0.5*np.cos(x/17.0)
        g = 0.5 + 0.5*np.cos(y/13.0)
        b = 0.5 + 0.5*np.cos((x+y)/21.0)
        out = np.stack([r, g, b], axis=2).astype(np.float32)
        # a moving bright disc for a recognisable landmark
        cx, cy = (i*11.0) % w, h*0.5 + 40*np.sin(i/9.0)
        m = ((xx-cx)**2 + (yy-cy)**2) < 32**2
        out[m] = 1.0
        return out

src = MovingSource()
base = FACTORY_PRESETS["family_archive_1992"]["layers"]
dv = merge_values(base, {"codec": {"dv_enabled": 1.0}})
# neutralise the analog-looking knobs so the block residue is unambiguous
dv = merge_values(dv, {"tape": {"line_jitter": 0.0, "walk_amplitude": 0.0,
                                "micro_jitter": 0.0, "noise_floor": 0.0,
                                "head_switching": 0.0, "dropout_rate": 0.0},
                       "camera": {"sensor_noise": 0.0}, "display": {"enabled": 0.0}})
tiles = []
for label, ov in (("normal", {"shuttle_speed": 0.0}),
                  ("5x 25%", {"shuttle_speed": 5.0, "shuttle_stale": 0.25}),
                  ("5x 25% clustered", {"shuttle_speed": 5.0, "shuttle_stale": 0.25,
                                        "shuttle_cluster": 0.85}),
                  ("5x 45% clustered", {"shuttle_speed": 5.0, "shuttle_stale": 0.45,
                                        "shuttle_cluster": 0.85})):
    r = Renderer(src, src.width, src.height, src.fps,
                 merge_values(dv, {"tape": ov}), seed=7)
    tiles.append(r.render_frame(20).copy())
    print("%-18s mean|d| vs normal = %.5f" % (label,
      0.0 if len(tiles) == 1 else float(np.abs(tiles[-1]-tiles[0]).mean())))
Image.fromarray((np.clip(np.concatenate(tiles, axis=1),0,1)*255).astype(np.uint8)).save(
  "tools/out/shuttle_moving.png")
print("wrote tools/out/shuttle_moving.png")
