"""Audit every camera parameter for a dead range: pull it to each end of its span and see whether anything changes.

**Why a full sweep rather than another single hunt.**  "A value that reaches the stage and is then replaced, or whose
effect is nil at every setting a user can reach" has now appeared in this work for `nv_ir_response`, `nv_ir_albedo`,
`nv_ir_depth`, `nv_exposure_trim`, `dsp_gain`, `dsp_black`, `nv_chroma_noise`, `nv_saturation_stops`,
`nv_nir_grid`, `ir_highlights`'s high-pass, `read_noise`, `noise_floor` and `ir_focus_shift` -- record enough of them
and the question stops being "which parameter is broken" and becomes "how many are".  This measures all of them the
same way, on one picture, and reports only the ones whose entire range is inert.

The picture is a **flat mid-grey with a fine sinusoidal grid**, so every class of effect has something to act on:
a blur has the grid, a gain has the level, a noise term has the flat field.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, '.')
from atsim.core.params import merge_values
from atsim.core.profiles import FACTORY_PRESETS, SCHEMA
from atsim.render.renderer import Renderer

W, H = 240, 180
BASE = SCHEMA.resolve_all(FACTORY_PRESETS['broadcast_1992_weak']['layers'])
LUMA = np.array([0.299, 0.587, 0.114], np.float32)


class Grid:
    fps, frame_count = 25.0, 2
    width, height = W, H

    def frame(self, index):
        y = np.arange(H, dtype=np.float32)[:, None]
        x = np.arange(W, dtype=np.float32)[None, :]
        a = np.zeros((H, W, 3), np.float32)
        g = 0.5 + 0.25 * np.sin(x / 1.7) * np.sin(y / 2.3)
        a[..., 0] = g
        a[..., 1] = 0.5 + 0.25 * np.sin((x + y) / 2.1)
        a[..., 2] = 0.5 + 0.25 * np.cos(x / 1.9)
        return np.clip(a, 0, 1)


def render(cam_over, mode=1.0):
    lay = {'camera': dict(nv_mode=mode, **cam_over),
           'format': {'chroma_bandwidth': 2.0, 'luma_bandwidth': 5.0},
           'player': {'chroma_bandwidth': 2.0, 'luma_bandwidth': 5.0},
           'codec': {'dv_enabled': 1.0}}
    r = Renderer(Grid(), W, H, 25.0, merge_values(BASE, lay), seed=9)
    return np.clip(np.asarray(r.render_frame(1), np.float32), 0, 1)


SPECS = SCHEMA.layer('camera').specs
SKIP = {'nv_mode', 'nv_nir_grid', 'nv_nir_cell_px', 'nv_nir_grid_max', 'nv_nir_margin_floor',
        'nv_nir_material', 'nv_nir_depth'}
print('  %d camera parameters; sweeping each to both ends of its declared span' % len(SPECS))
print('  (the night-vision material analysis is excluded: it needs the vision model and a real picture)')
print()
base = render({})
rows = []
for k in sorted(SPECS):
    sp = SPECS[k]
    if k in SKIP or not isinstance(sp.default, (int, float)):
        continue
    lo, hi = float(sp.minimum), float(sp.maximum)
    if abs(hi - lo) < 1e-9:
        continue
    try:
        a, b = render({k: lo}), render({k: hi})
    except Exception as exc:                                  # noqa: BLE001
        rows.append((k, sp.default, lo, hi, float('nan'), 'ERROR %s' % type(exc).__name__))
        continue
    dl = float(np.abs(a - base).max())
    dh = float(np.abs(b - base).max())
    rows.append((k, sp.default, lo, hi, max(dl, dh), ''))

dead = [r for r in rows if not (r[4] != r[4]) and r[4] < 1e-6]
weak = [r for r in rows if r[4] == r[4] and 1e-6 <= r[4] < 1e-4]
live = [r for r in rows if r[4] == r[4] and r[4] >= 1e-4]
print('  INERT over the whole range (%d) -- every setting a user can reach gives the same picture:' % len(dead))
for k, d, lo, hi, v, note in dead:
    print('     %-24s default %-8s range [%s, %s]   max change %s %s' % (k, d, lo, hi, v, note))
print()
print('  NEARLY inert (max change under 1e-4) (%d):' % len(weak))
for k, d, lo, hi, v, note in weak:
    print('     %-24s default %-8s   max change %.6f' % (k, d, v))
print()
print('  live: %d of %d' % (len(live), len(rows)))
print()
print('  the ten that move the picture most, for scale:')
for k, d, lo, hi, v, note in sorted(live, key=lambda r: -r[4])[:10]:
    print('     %-24s default %-8s   max change %.5f' % (k, d, v))
