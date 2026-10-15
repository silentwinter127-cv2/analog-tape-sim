"""Re-run the parameter audit with a source that CHANGES EVERY FRAME.

**Why the first audit could not see fourteen of its own subjects.**  That sweep used a static grid, which is the right
fixture for a spatial control and useless for a temporal one: an auto-exposure loop has nothing to converge to, an
accumulation has identical frames to average, and noise that is re-drawn per frame cancels in a way it never does in
motion.  Fourteen parameters were therefore reported "inert" without the test having any ability to see them -- **and
that is the same measurement error this work keeps recording, made by the instrument rather than by the code.**

This version drives the camera with a source that varies frame to frame, and compares **the whole rendered sequence**
rather than one frame, so a term that only shows up in the differences between frames is measured by exactly those
differences.
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, '.')
from atsim.core.params import merge_values
from atsim.core.profiles import FACTORY_PRESETS, SCHEMA
from atsim.render.renderer import Renderer

W, H = 240, 180
N = 8
BASE = SCHEMA.resolve_all(FACTORY_PRESETS['broadcast_1992_weak']['layers'])


class Flicker:
    """A source that differs every frame, and differs in *level* as well as in pattern.

    A level change is what an auto-exposure or auto-white-balance loop responds to, and a pattern change is what an
    accumulation averages and a noise seed decorrelates.  Both are needed, so both are here.
    """
    fps, frame_count = 25.0, N
    width, height = W, H

    def frame(self, index):
        y = np.arange(H, dtype=np.float32)[:, None]
        x = np.arange(W, dtype=np.float32)[None, :]
        lev = 0.30 + 0.30 * ((index % 4) / 3.0)              # 4-frame level cycle
        ph = 0.7 * index                                     # a moving pattern
        a = np.zeros((H, W, 3), np.float32)
        a[..., 0] = lev + 0.22 * np.sin(x / 1.7 + ph) * np.sin(y / 2.3)
        a[..., 1] = lev + 0.22 * np.sin((x + y) / 2.1 + ph)
        a[..., 2] = lev + 0.22 * np.cos(x / 1.9 + ph * 0.6)
        return np.clip(a, 0, 1)


def sequence(cam_over, mode=0.0):
    """**`mode` is a parameter because the first version of this audit forgot it.**

    The camera layer's `nv_mode` defaults to 0.0 and `mode_of` only returns a night-vision mode at 1.0 or above, so
    every `nv_*` parameter is inert while the mode is off -- **which is what the first run of this file measured
    without noticing, reporting `nv_accumulate`, `nv_chroma_noise`, `nv_read_noise` and `nv_shot_noise` as dead.**  That
    is the third time in this work a measurement has been taken in a configuration that could not see its own subject,
    and the second time in this same audit.
    """
    lay = {'camera': dict(cam_over, nv_mode=mode),
           'format': {'chroma_bandwidth': 2.0, 'luma_bandwidth': 5.0},
           'player': {'chroma_bandwidth': 2.0, 'luma_bandwidth': 5.0},
           'codec': {'dv_enabled': 1.0}}
    r = Renderer(Flicker(), W, H, 25.0, merge_values(BASE, lay), seed=9)
    return np.stack([np.clip(np.asarray(f, np.float32), 0, 1) for _, f in r.render_frames(range(N))])


def spread(seq):
    """How much the sequence varies over time -- the statistic a temporal term must move."""
    flat = seq.reshape(N, -1)
    return float(flat.std(axis=0).mean())


def mean_of(seq):
    return float(seq.mean())


SPECS = SCHEMA.layer('camera').specs
#: the fourteen the static audit could not judge, plus the temporal machinery around them
SUBJECTS = ['ae_enabled', 'ae_limit_stops', 'ae_target', 'ae_tau_s', 'af_enabled', 'af_tau_s',
            'autopresettle_s', 'awb_enabled', 'awb_limit', 'awb_tau_s', 'nv_accumulate', 'nv_chroma_noise',
            'nv_read_noise', 'nv_shot_noise', 'bump_amount', 'bump_rate_hz', 'bloom_threshold', 'flare_blades',
            'nv_green_mix', 'nv_chroma_residue', 'overflow_width', 'spike_spread', 'stuck_fraction', 'sensor_noise']

base_seq = sequence({}, 1.0)
b0, s0 = mean_of(base_seq), spread(base_seq)
print('  THE SOURCE CHANGES EVERY FRAME, and the comparison is over the WHOLE SEQUENCE')
print('  **nv_mode = 1.0**, because every `nv_*` parameter is inert while the mode is off and the first run of this')
print('  file did not notice that it was measuring them with the mode off')
print('  baseline: mean %.5f   temporal spread %.5f      (%d frames)' % (b0, s0, N))
print()
print('   %-22s %-9s %-11s %-11s %s' % ('parameter', 'default', 'mean change', 'temporal', 'verdict'))
print('   ' + '-' * 88)
rows = []
for k in SUBJECTS:
    sp = SPECS.get(k)
    if sp is None or not isinstance(sp.default, (int, float)):
        continue
    lo, hi = float(sp.minimum), float(sp.maximum)
    best = 0.0
    best_t = 0.0
    for v in (lo, hi):
        if abs(v - float(sp.default)) < 1e-9:
            continue
        seq = sequence({k: v}, 1.0)
        best = max(best, abs(mean_of(seq) - b0))
        best_t = max(best_t, abs(spread(seq) - s0))
    verdict = 'LIVE' if max(best, best_t) > 1e-4 else '** INERT **'
    rows.append((k, float(sp.default), best, best_t, verdict))
    print('   %-22s %-9s %-11.5f %-11.5f %s' % (k, sp.default, best, best_t, verdict))
print()
inert = [r for r in rows if r[4] != 'LIVE']
print('  still inert WITH a changing source (%d of %d): %s'
      % (len(inert), len(rows), ', '.join(r[0] for r in inert) or '(none)'))
print()
print('  and the ones the static audit could not judge but this one can:')
for k, d, m, t, v in rows:
    if v == 'LIVE' and k in ('nv_accumulate', 'nv_chroma_noise', 'nv_read_noise', 'nv_shot_noise', 'sensor_noise',
                             'bump_amount', 'bump_rate_hz', 'autopresettle_s', 'ae_enabled', 'awb_enabled'):
        print('     %-22s mean moves %.5f, temporal moves %.5f' % (k, m, t))
