"""Find the projection triple that delivers the real frames' channel ratios AFTER the highlight clip.

**The two quantities are not the same thing and that is the whole difficulty.**  `monochrome_green`'s triple is the
*projection*, and above the saturation gate the strongest channel is pinned while the others keep climbing -- so the
picture **desaturates toward white** and its mean ratios move away from the triple.  The user's target (R/G 0.389,
B/G 0.801) is a measurement of a *finished picture*, which includes whatever clipping the real camcorder did.  Setting
the triple to the target therefore overshoots:

    triple [0.3890, 1.0, 0.8008]   -> normalised [0.4895, 1.2585, 1.0078]
    green pinned at lum 0.795 and blue at 0.992, both inside the picture's own range
    measured output R/G 0.4750 -- against the 0.3890 the triple asks for

**So the triple is the free variable and the OUTPUT is the observable**, and the map between them is measured here on
the real frame.  Two points give the local slope, which is enough to place the red channel; blue is re-solved the same
way afterwards.
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


def load(p, frac=0.20):
    a = np.asarray(Image.open(p).convert('RGB'), np.float32) / 255.0
    if frac:
        a = a[int(a.shape[0] * frac):]
    return np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS),
                      np.float32) / 255.0


def ratios(x):
    a = np.asarray(x, np.float32)
    r, g, b = (float(a[..., i].mean()) for i in range(3))
    return r / max(1e-9, g), b / max(1e-9, g)


def run(src, mode=2.0, **cam):
    lay = {'camera': dict(nv_mode=mode, **cam),
           'format': {'chroma_bandwidth': 2.0, 'luma_bandwidth': 5.0},
           'player': {'chroma_bandwidth': 2.0, 'luma_bandwidth': 5.0},
           'codec': {'dv_enabled': 1.0}}
    r = Renderer(Still(src), W, H, 25.0, merge_values(BASE, lay), seed=20240101)
    return np.clip(np.asarray(r.render_frame(0), np.float32), 0, 1)


#: the two inputs, so the solve is not fitted to one scene -- the mistake that put the previous triple on the palest
#: sample of its segment
INPUTS = {
    'the user real NS2 frame a': load('tools/nightshot/ref_ns2_a.png', 0.18),
    'the 10月13 NS2 frame': load('tools/nightshot/frames/nv2_1.png', 0.20),
}
TARGET = (0.3890, 0.8008)

print('  WHAT THE TRIPLE IN THE CODE DELIVERS (it is [0.3890, 1.0000, 0.8008] right now)')
for tag, src in INPUTS.items():
    r, b = ratios(run(src))
    print('    %-28s out R/G %.4f  B/G %.4f' % (tag, r, b))
print('    the target from the real frames:   R/G %.4f  B/G %.4f' % TARGET)
print()

#: **turn the highlight clip off to read the pure projection**, which is the quantity the triple sets directly
print('  THE SAME INPUTS WITH `nv_highlight_saturation = 0`, so the clip cannot move the ratios:')
for tag, src in INPUTS.items():
    r, b = ratios(run(src, nv_highlight_saturation=0.0))
    print('    %-28s out R/G %.4f  B/G %.4f   (the triple is 0.3890 / 0.8008)'
          % (tag, r, b))
print()
print('  so the clip is worth: R/G x%.4f   B/G x%.4f   on the user frame, and the triple must be')
print('  divided by that to land the OUTPUT on the target.')
r_clip, b_clip = ratios(run(INPUTS['the user real NS2 frame a']))
r_lin, b_lin = ratios(run(INPUTS['the user real NS2 frame a'], nv_highlight_saturation=0.0))
kr, kb = r_clip / max(1e-9, r_lin), b_clip / max(1e-9, b_lin)
print('    clip factor R/G %.4f   B/G %.4f' % (kr, kb))
print('    a projection of [%.4f, 1.0, %.4f] should then land on the target'
      % (TARGET[0] / kr, TARGET[1] / kb))
