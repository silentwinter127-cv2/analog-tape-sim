"""For each camera parameter the audit found inert, ask the source whether it is read at all.

The audit cannot tell a parameter that is dead in the code from one that is merely invisible to a **static** test
picture -- a frame-averaging loop, a noise term or a per-frame control all need frames that differ.  This walk separates
the two: if the schema key never appears in `camera.py`, the parameter can never do anything whatever the input is.
"""
import io
import re

SRC = io.open('atsim/stages/camera.py', encoding='utf-8').read()
INERT = ['ae_limit_stops', 'ae_target', 'ae_tau_s', 'af_enabled', 'af_tau_s', 'autopresettle_s', 'awb_limit',
         'awb_tau_s', 'bloom_threshold', 'bump_amount', 'bump_rate_hz', 'flare_blades', 'flare_defocus',
         'nv_accumulate', 'nv_chroma_noise', 'nv_chroma_residue', 'nv_green_mix', 'nv_read_noise',
         'nv_shot_noise', 'overflow_width', 'spike_spread', 'stuck_fraction', 'zoom_kind']

print('%-22s %-8s %-8s %-10s %s' % ('parameter', 'key?', 'self.?', 'uses self', 'verdict'))
print('-' * 92)
never = []
for n in INERT:
    key = ('"%s"' % n) in SRC
    assigned = bool(re.search(r'self\.%s\s*=' % re.escape(n), SRC))
    #: is the attribute read anywhere other than its own assignment?
    reads = len(re.findall(r'self\.%s\b' % re.escape(n), SRC))
    uses = reads - (1 if assigned else 0)
    if not key:
        verdict = 'DEAD: the schema key never appears in camera.py'
        never.append(n)
    elif uses <= 0:
        verdict = 'DEAD: assigned but never read'
        never.append(n)
    else:
        verdict = 'inert to a STATIC test (needs frames that differ)'
    print('%-22s %-8s %-8s %-10s %s' % (n, 'yes' if key else 'NO', 'yes' if assigned else 'NO', uses, verdict))

print()
print('  parameters that cannot work whatever the input is (%d): %s' % (len(never), ', '.join(never) or '(none)'))
