"""P1/P2 and P3/P4 compared with the resolution difference removed.

The four panels are preview and export of two formats, and the export is at twice the linear size.  Comparing them
naively conflates two things:

  * **resolution**, which changes every high-frequency statistic on its own -- a downscale is a low-pass; and
  * **the chain**, which is what the user is actually looking at when they say the two do not match.

So the export is brought DOWN to the preview's size first, which is the same operation the viewer performs when it
scales one to fit the other, and only then are the two compared.  Anything left after that is the chain.
"""
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, '.')
sys.path.insert(0, 'tools')
LUMA = np.array([0.299, 0.587, 0.114], np.float32)


def load(p):
    return np.asarray(Image.open(p).convert('RGB'), np.float32) / 255.0


def down_to(a, w, h):
    return np.asarray(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).resize((w, h), Image.LANCZOS),
                      np.float32) / 255.0


def stat(x, tag):
    a = np.asarray(x, np.float32)
    l = np.clip(a @ LUMA, 0, 1)
    r, g, b = (float(a[..., i].mean()) for i in range(3))
    row = float(np.abs(np.diff(l, axis=0)).mean())
    col = float(np.abs(np.diff(l, axis=1)).mean())
    return dict(tag=tag, mean=float(l.mean()), rg=r / max(1e-9, g), bg=b / max(1e-9, g),
                p1=float(np.quantile(l, 0.01)), p99=float(np.quantile(l, 0.99)),
                std=float(l.std()), row=row, col=col)


def show(s):
    print('  %-26s mean %.4f  R/G %.4f  B/G %.4f  p1 %.4f  p99 %.4f  std %.4f  row %.5f  col %.5f'
          % (s['tag'], s['mean'], s['rg'], s['bg'], s['p1'], s['p99'], s['std'], s['row'], s['col']))


PAIRS = [('VHS', 'tools/nightshot/p1_vhs_preview.png', 'tools/nightshot/p2_vhs_export.png'),
         ('MiniDV', 'tools/nightshot/p3_minidv_preview.png', 'tools/nightshot/p4_minidv_export.png')]

for name, pp, pe in PAIRS:
    pv = load(pp)
    ex = load(pe)
    h, w = pv.shape[:2]
    ex_d = down_to(ex, w, h)
    print('=' * 118)
    print('  %s -- preview %dx%d, export %dx%d, brought down to the preview size for comparison'
          % (name, w, h, ex.shape[1], ex.shape[0]))
    print('=' * 118)
    show(stat(pv, 'P%s preview' % ('1' if name == 'VHS' else '3')))
    show(stat(ex_d, 'export, downscaled'))
    d = np.abs(pv - ex_d).mean(axis=2)
    print('  %-26s mean |difference| %.5f   worst 1%% %.5f   the picture mean is %.4f'
          % ('the two, compared', float(d.mean()), float(np.quantile(d, 0.99)), float(pv.mean())))
    sp, se = stat(pv, ''), stat(ex_d, '')
    print('  %-26s R/G %+.4f   B/G %+.4f   mean %+.4f   std %+.4f'
          % ('chain difference', se['rg'] - sp['rg'], se['bg'] - sp['bg'], se['mean'] - sp['mean'],
             se['std'] - sp['std']))
    print()
