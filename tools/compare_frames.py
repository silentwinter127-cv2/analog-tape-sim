
import hashlib, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from atsim.core.params import merge_values
from atsim.core.profiles import FACTORY_PRESETS, SCHEMA
from atsim.render.renderer import Renderer

class Chart:
    def __init__(self, frames=6):
        self.width, self.height, self.fps, self.frame_count = 320, 240, 25.0, frames
    def frame(self, index):
        a = np.full((240, 320, 3), 0.35, np.float32)
        a[:] = (((np.arange(320) // 16) % 2) * 0.5 + 0.25)[None, :, None]
        a[40:56, 40:56] = 2.2
        a[180:196, 250:266] = 0.95
        a[110:130, :] = [0.70, 0.25, 0.30]
        return a

rows = []
for preset in ("broadcast_1992_weak", "family_archive_1992", "rental_tape_horror"):
    for layer, over in (("camera", {}),
                        ("camera", {"chroma_aberration": 0.6}),
                        ("camera", {"longitudinal_ca": 0.5}),
                        ("camera", {"field_curvature": 0.5}),
                        ("display", {"enabled": 1.0})):
        base = SCHEMA.resolve_all(FACTORY_PRESETS[preset]["layers"])
        p = merge_values(base, {layer: dict({"exposure": 0.0}, **over)})
        rr = Renderer(Chart(), 320, 240, 25.0, p, seed=7)
        for i in (0, 3):
            img = np.clip(rr.render_frame(i), 0, 1)
            rows.append("%s %-42s f%d %s" % (preset, over, i,
                        hashlib.blake2s(img.tobytes(), digest_size=8).hexdigest()))
print("\n".join(rows))
