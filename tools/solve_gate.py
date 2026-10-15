"""Which `nv_highlight_saturation` leaves the mid-tones on the display's own ratios?

The triple's comment says "below the gate the projection is exact, so the mid-tones keep the measured R:G:B".  With the
triple re-solved to `[0.3890, 1.0000, 0.8008]` the normalised weights are `[0.4896, 1.2585, 1.0078]` -- **green and
blue both above 1** -- so the gate that pins a channel sits *inside* the picture's range and the whole frame is being
compressed toward white.  That is a configuration fault, and the ratios say so: the frame enters `monochrome_green` at
R/G 0.3915 / B/G 0.7855 and leaves it at 0.4508 / 0.8317, i.e. **dragged toward white by the clip before anything
else happens.**

So the question is not "what triple" but "what gate", and it is measured here on both real inputs.
"""
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, '.')
sys.path.insert(0, 'tools')
from atsim.core.params import merge_values
from atsim.core.profiles import FACTORY_PRESETS, SCHEMA
from atsim.render.renderer import Renderer

W, H = 720, 540
BASE = SCHEMA.resolve_all(FACTORY_PRESETS['broadcast_1992_weak']['layers'])
LUMA = np.array([0.299, 0.587, 0.114], np.float32)


class Still:
    fps, frame_count = 25.0, 1

    def __init__(self, a):
        self.a = a
        self.height, self.width = a.shape[:2]

    def frame(self, i):
        return self.a


def load(p, frac):
    a = np.asarray(Image.open(p).convert('RGB'), np.float32) / 255.0
    a = a[int(a.shape[0] * frac):]
    return np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS),
                      np.float32) / 255.0


INPUTS = {'user NS2 a': load('tools/nightshot/ref_ns2_a.png', 0.18),
          'user NS2 b': load('tools/nightshot/ref_ns2_b.png', 0.18),
          '10月13 NS2': load('tools/nightshot/frames/nv2_1.png', 0.20)}


def stat(x):
    a = np.asarray(x, np.float32)
    l = np.clip(a @ LUMA, 0, 1)
    r, g, b = (float(a[..., i].mean()) for i in range(3))
    return r / max(1e-9, g), b / max(1e-9, g), float(l.mean()), 100 * float((l > 0.90).mean())


def run(src, **cam):
    lay = {'camera': dict(nv_mode=2.0, **cam),
           'format': {'chroma_bandwidth': 2.0, 'luma_bandwidth': 5.0},
           'player': {'chroma_bandwidth': 2.0, 'luma_bandwidth': 5.0},
           'codec': {'dv_enabled': 1.0}}
    r = Renderer(Still(src), W, H, 25.0, merge_values(BASE, lay), seed=20240101)
    return np.clip(np.asarray(r.render_frame(0), np.float32), 0, 1)


print('  THE REAL FRAMES (central crop, OSD excluded):')
for tag, src in list(INPUTS.items())[:2]:
    r, b, m, blown = stat(src)
    print('    %-14s R/G %.4f  B/G %.4f   mean %.4f  >0.90 %.2f%%' % (tag, r, b, m, blown))
print()
print('   %-12s %-34s %-34s %s' % ('gate', 'user NS2 a', 'user NS2 b', '10月13'))
for gate in (4.0, 2.0, 1.5, 1.2, 1.0, 0.0):
    out = []
    for tag, src in INPUTS.items():
        r, b, m, blown = stat(run(src, nv_highlight_saturation=gate))
        out.append('R/G %.4f B/G %.4f' % (r, b))
    print('   %-12s %-34s %-34s %s' % ('%.1f' % gate, out[0], out[1], out[2]))
print()
print('  for reference the targets are R/G 0.389/0.368 and B/G 0.797/0.805 on the two user frames')
