"""
Test `horizontal_shift` as an operation, which is not what the earlier time-base test did.

**The distinction that was missed.** `_timebase_offset` was forced to zero, which removes the *source* of the
displacement -- and the ripple barely moved, 0.006990 to 0.006783. But `horizontal_shift` is still applied, with a
zero displacement, and the question of what that **operation** does was never asked. A shift by a fractional number
of samples is an interpolation, and interpolating a **subcarrier** is not the same as interpolating a picture: at 3.17
samples per cycle, moving by a fraction of a sample changes the phase of the carrier, so the same operation that
displaces a picture sideways can rotate its colour.

So the operation is exercised directly, on a signal whose answer is known:

    a constant I  ->  `horizontal_shift` by a range of displacements  ->  what comes back

**A constant must come back constant.** If the recovered plane acquires per-row variation as a function of the
displacement, then the operation is the mechanism, and the sizes are read off rather than argued about.
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from atsim.core.signal import horizontal_shift                             # noqa: E402

H, W = 240, 320


def ripple(a):
    x = np.asarray(a, np.float32)
    return float(np.abs(np.diff(x.mean(axis=1))).mean())


print("   a CONSTANT plane through `horizontal_shift`: a constant must come back constant")
print()
print("   %-14s %-16s %-16s %-16s %s"
      % ("displacement", "mean", "max |error|", "row-to-row ripple", "changed?"))

const = np.full((H, W), 0.30, np.float32)
for d in (0.0, 0.001, 0.01, 0.1, 0.25, 0.5, -0.5, 1.0, 1.5, -0.3876):
    # `disp` is per row in the stage; a single scalar broadcast to every row is the same operation
    disp = np.full(H, float(d), np.float32)
    out = np.asarray(horizontal_shift(const, disp), np.float32)
    print("   %-14.4f %-16.6f %-16.6f %-16.6f %s"
          % (d, float(out.mean()), float(np.abs(out - 0.30).max()), ripple(out),
             "no" if np.abs(out - const).max() < 1e-9 else "YES"))

print()
print("   and a plane that is constant down each column but ramps across it -- the known-answer signal's shape:")
ramp = np.tile(np.linspace(-0.4, 0.4, W, dtype=np.float32)[None, :], (H, 1))
print("   %-14s %-16s %-16s %-16s %s"
      % ("displacement", "mean", "max |error|", "row-to-row ripple", "changed?"))
for d in (0.0, 0.25, 0.5, -0.5, 1.5, -0.3876):
    disp = np.full(H, float(d), np.float32)
    out = np.asarray(horizontal_shift(ramp, disp), np.float32)
    print("   %-14.4f %-16.6f %-16.6f %-16.6f %s"
          % (d, float(out.mean()), float(np.abs(out - ramp).max()), ripple(out),
             "no" if np.abs(out - ramp).max() < 1e-9 else "YES"))

print()
print("   and with a displacement that VARIES per row, which is what the stage actually passes:")
rng = np.random.default_rng(5)
for scale in (0.0, 0.1, 0.5, 1.0, 2.0):
    disp = (rng.standard_normal(H) * scale).astype(np.float32)
    out = np.asarray(horizontal_shift(const, disp), np.float32)
    print("      per-row disp std %.2f -> constant plane: max |error| %.6f, ripple %.6f"
          % (scale, float(np.abs(out - 0.30).max()), ripple(out)))
    out2 = np.asarray(horizontal_shift(ramp, disp), np.float32)
    print("      %-28s ramp: max |error| %.6f, ripple %.6f"
          % ("", float(np.abs(out2 - ramp).max()), ripple(out2)))
