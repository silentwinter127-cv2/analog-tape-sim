"""Re-solve the display projection from the two real NIGHTSHOT 2 frames the user supplied.

**Why the existing triple has to be re-solved rather than nudged.**  `monochrome_green` maps the mono signal onto the
display's primary ratios, and the triple currently in the code was solved from eight flat fields of the `10月13` clip:

    solved then      R/G 0.5434   B/G 0.8525      (10月13, NIGHTSHOT 1 material)
    the user's now   R/G 0.450    B/G 0.832       (two NIGHTSHOT 2 frames, native 1440x1080)

so the red channel is carrying **0.093 more than the target asks for**, which is the whole of the yellow-green cast:
measured hue medians are 156 and 154 degrees for the real frames against 161 for the render.

The solve is a ratio by construction -- `lum * w` then divided by `w @ LUMA` gives exactly `w` as the output's channel
ratios -- so the new triple is the target itself, and the divisor is only there to keep the level.
"""
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, '.')
sys.path.insert(0, 'tools')
LUMA = np.array([0.299, 0.587, 0.114], np.float32)


def chan_means(path, margin=0.02):
    """Channel means over the picture only, with the OSD band and the letterbox border removed.

    **The OSD has to be excluded or it solves the triple against itself.**  A NIGHTSHOT frame carries `STBY`, a battery
    indicator, the mode glyph and a running clock, all in **neutral white and a saturated green** -- so including them
    drags the measured R/G toward white's 1.0 and biases the very ratio being solved.  The band is the top ~14% and the
    glyph sits mid-frame; a central crop avoids both and is what a viewer judges the colour on anyway.
    """
    a = np.asarray(Image.open(path).convert('RGB'), np.float32) / 255.0
    h, w = a.shape[:2]
    b = a[int(h * 0.18):int(h * 0.94), int(w * (margin + 0.06)):int(w * (1.0 - margin - 0.06))]
    r, g, bb = (float(b[..., i].mean()) for i in range(3))
    return r, g, bb, b


rows = []
for p in ('tools/nightshot/ref_ns2_a.png', 'tools/nightshot/ref_ns2_b.png'):
    r, g, b, patch = chan_means(p)
    rows.append((r / g, b / g))
    print('%-34s central crop R/G %.4f  B/G %.4f   (mean level %.4f)'
          % (p.split('/')[-1], r / g, b / g, float(np.clip(patch @ LUMA, 0, 1).mean())))

rg = float(np.mean([x[0] for x in rows]))
bg = float(np.mean([x[1] for x in rows]))
print()
print('  the target the display has to project:  R/G %.4f   B/G %.4f' % (rg, bg))
old = np.array([0.5434, 1.0, 0.8525], np.float32)
print('  the triple in the code now:             R/G %.4f   B/G %.4f' % (old[0], old[2]))
print('  -> the red channel is %.4f too high, the blue %.4f' % (old[0] - rg, old[2] - bg))
print()

#: the projection is `lum * w` normalised by `w @ LUMA`, so the OUTPUT's ratios are `w` itself
new = np.array([rg, 1.0, bg], np.float32)
norm = new / float(new @ LUMA)
print('  the new triple               [%.4f, %.4f, %.4f]' % tuple(new))
print('  after the level normalisation [%.4f, %.4f, %.4f]' % tuple(norm))
print('  and its ratios, which is what the picture shows: R/G %.4f  B/G %.4f'
      % (norm[0] / norm[1], norm[2] / norm[1]))
print()
import colorsys
for tag, w in (('old', old), ('new', new)):
    h, s, v = colorsys.rgb_to_hsv(*(w / w.max()))
    print('  %s triple -> hue %.1f deg, saturation %.3f' % (tag, h * 360, s))
