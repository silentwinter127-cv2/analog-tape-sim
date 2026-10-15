"""Does the chain discard chroma, and at which point?

**The user's point is not about the hue of the green, it is that there is nothing BUT green.**  Those are different
physical statements and only one of them is a fault:

  * a **VHS or MiniDV chroma subcarrier carries real colour**, so a picture through those formats and out to a display
    is *coloured* -- the colour is the signal;
  * a **NightShot green display sends a monochrome signal to the green phosphor**, so the colour is *gone by
    construction* at the mono conversion.  That is `note.txt`'s "黑白信号 → 绿色显示矩阵" and it is correct for that
    mode.

So the question is which of the two the user is looking at, and it is answered by measuring how much chroma survives
each step -- with `green_mix` as the control, since that is the parameter that decides mono-versus-colour.
"""
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, '.')
sys.path.insert(0, 'tools')
from atsim.codecs import nightvision as nv
from atsim.codecs.nightvision import NightVisionSpec, mode_spec
from atsim.core.params import merge_values
from atsim.core.profiles import FACTORY_PRESETS, SCHEMA
from atsim.render.renderer import Renderer

BASE = SCHEMA.resolve_all(FACTORY_PRESETS['broadcast_1992_weak']['layers'])
LUMA = np.array([0.299, 0.587, 0.114], np.float32)


class Still:
    fps, frame_count = 25.0, 1

    def __init__(self, a):
        self.a = a
        self.height, self.width = a.shape[:2]

    def frame(self, i):
        return self.a


def chroma(x):
    """How much colour a picture has: the mean distance of each pixel from the grey of its own luma."""
    a = np.asarray(x, np.float32)
    l = (a @ LUMA)[..., None]
    return float(np.abs(a - l).mean())


def ratios(x):
    a = np.asarray(x, np.float32)
    r, g, b = (float(a[..., i].mean()) for i in range(3))
    return r / max(1e-9, g), b / max(1e-9, g)


def render(src, mode, **cam):
    lay = {'camera': dict(nv_mode=mode, **cam),
           'format': {'chroma_bandwidth': 2.0, 'luma_bandwidth': 5.0},
           'player': {'chroma_bandwidth': 2.0, 'luma_bandwidth': 5.0},
           'codec': {'dv_enabled': 1.0}}
    r = Renderer(Still(src), src.shape[1], src.shape[0], 25.0, merge_values(BASE, lay), seed=20240101)
    return np.clip(np.asarray(r.render_frame(0), np.float32), 0, 1)


a = np.asarray(Image.open('tools/nightshot/sc1515/d_nv3.png').convert('RGB'), np.float32) / 255.0
a = a[int(a.shape[0] * 0.20):]
src = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize((720, 540), Image.LANCZOS),
                 np.float32) / 255.0
src = src * np.array([1.15, 0.95, 0.80], np.float32)      # a warm cast, so colour is unmistakable
src = np.clip(src, 0, 1)
print('  the input: chroma %.5f   R/G %.4f  B/G %.4f' % (chroma(src), *ratios(src)))
print()
print('   %-34s %-10s %-9s %-9s' % ('through the chain', 'chroma', 'R/G', 'B/G'))
for tag, mode, cam in (
    ('nv_mode 0  (no night vision)', 0.0, {}),
    ('nv_mode 3  (colour slow shutter)', 3.0, {}),
    ('nv_mode 2, green_mix 0', 2.0, {'nv_green_mix': 0.0}),
    ('nv_mode 2, green_mix 0.5', 2.0, {'nv_green_mix': 0.5}),
    ('nv_mode 2, green_mix 1 (as shipped)', 2.0, {}),
):
    o = render(src, mode, **cam)
    print('   %-34s %-10.5f %-9.4f %-9.4f' % (tag, chroma(o), *ratios(o)))
print()
print('  and the mono conversion on its own, at the three mixes:')
for mix in (0.0, 0.5, 1.0):
    spec = mode_spec(NightVisionSpec.from_params(dict(BASE['camera'], nv_mode=2.0, nv_green_mix=mix)))
    o = np.clip(np.asarray(nv.monochrome_green(src, spec), np.float32), 0, 1)
    print('    green_mix %.1f -> chroma %.5f   R/G %.4f  B/G %.4f' % (mix, chroma(o), *ratios(o)))
print()
print('  green_mix is a camera key: %s' % ('"nv_green_mix"' in BASE['camera']))
print('  and `monochrome_green` reads it from the spec as `spec.green_mix`')
