"""
Derive the exposure loop's damping from the measured overshoot, and report honestly how well one value fits.

**The measurement being modelled.** Across the seven mode changes the level overshoots its settled target by
**28.60% to 81.51% of the level**, against a control noise floor of **2.65% to 6.22%**. The model's loop is
**first-order** -- a single time constant `tau_s = 1.1` -- and **a first-order lag cannot overshoot at all**. So the
loop is the wrong *order*, not merely mistuned.

**And a second-order lag's overshoot fixes its damping analytically**:

    OS = exp(-pi * zeta / sqrt(1 - zeta^2))        ->        zeta from the measured OS

**Five values come out, and the first thing to report is that they are spread.** An earlier draft of this script
asserted they agreed "to within a factor of two" before the numbers had been computed; **the measured spread is a
factor of 5.7** (0.065 to 0.370), so that claim was written ahead of its evidence and is retracted. **The table below is
what the data says.**

**What the spread does and does not support.** It **does** support a second order rather than a first, because a
first-order lag predicts exactly zero overshoot for all five and the footage shows five non-zero values against a 3-6%
noise floor. It **does not** support a single damping value as a precise calibration -- and the values are ordered
(NIGHTSHOT 2 -> after overshoots most, NIGHTSHOT 1 -> 2 least), which is a **pattern rather than scatter** and suggests
the transitions are not all the same kind of event. So the number reported is the **median**, with the spread stated,
and the ordering noted as something the single-damping model cannot reproduce.

**The natural frequency is a separate question and the two disagree.** An underdamped second-order system's period is
`T = 2*pi*tau_s/sqrt(1-zeta^2)`; at `tau_s = 1.1 s` and the median `zeta` it comes to several seconds, while the luma
series was measured to oscillate at **1.77 s** (r = 0.515). **The 1.77 s was measured on the green stretch's own luma,
which includes the operator panning**, so it bounds the loop's period rather than measuring it. The overshoot is what
the seven transitions constrain, each against its own before-and-after levels, so `zeta` comes from it and the period
stays at the loop's existing constant.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

OVERSHOOTS = [("NIGHTSHOT 1 -> NIGHTSHOT 2", 0.2860), ("NightShot -> normal", 0.3837),
              ("? -> NIGHTSHOT 1", 0.4397), ("NightShot -> ?", 0.4899),
              ("NIGHTSHOT 2 -> after", 0.8151)]
#: the control's noise floor, which decides whether an overshoot is significant at all
FLOOR = 0.0622
TAU = 1.1


def zeta_from_os(os_):
    lo, hi = 1e-6, 0.999
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if math.exp(-math.pi * mid / math.sqrt(1 - mid * mid)) > os_:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


print("   === the damping each transition implies, ordered by overshoot ===")
print("   %-32s %-11s %-11s %s" % ("transition", "overshoot", "zeta", "against the %.1f%% floor" % (100 * FLOOR)))
zs = []
for name, os_ in OVERSHOOTS:
    z = zeta_from_os(os_)
    zs.append(z)
    print("   %-32s %-11.2f%% %-11.4f %.1fx the floor"
          % (name, 100 * os_, z, os_ / FLOOR))
print()
print("   the five zetas: %s" % ", ".join("%.3f" % z for z in zs))
print("   spread: %.3f to %.3f, a factor of %.1f" % (min(zs), max(zs), max(zs) / min(zs)))
print("   median %.4f" % float(np.median(zs)))
print()
print("   **RETRACTION**: an earlier draft of this script asserted the five agreed 'to within a factor of two'")
print("   before they had been computed.  The measured spread is a factor of %.1f, so that claim was unsupported" % (max(zs) / min(zs)))
print("   and is withdrawn.")
print()
print("   what the spread DOES support: a second order rather than a first, because a first-order lag predicts")
print("   exactly ZERO overshoot for all five, and all five are non-zero against a %.1f%% noise floor." % (100 * FLOOR))
print("   what it does NOT support: a single damping as a precise calibration.")
print()
print("   and the values are ORDERED, which is a pattern rather than scatter:")
for name, os_ in OVERSHOOTS:
    print("      %-32s %.1f%%" % (name, 100 * os_))
print("      so the transitions are not all the same kind of event -- a single damping cannot reproduce that,")
print("      and saying so is more useful than averaging it away.")

Z = float(np.median(zs))
print()
print("   === the natural frequency, and a disagreement stated rather than smoothed over ===")
T_pred = 2 * math.pi * TAU / math.sqrt(1 - Z * Z)
print("      tau_s = %.2f s with zeta = %.3f gives an underdamped period of %.2f s" % (TAU, Z, T_pred))
print("      the luma series was measured to oscillate at 1.77 s (r = 0.515) -- THOSE DISAGREE")
print("      and the 1.77 s came from the green stretch's own luma, which includes the operator panning,")
print("      so it bounds the loop's period rather than measuring it.")
print()
print("   === the step response at the median damping, which is what the implementation would produce ===")
wn = 1.0 / TAU
t = np.arange(0, 8 * TAU, 1 / 30.0)
# the standard underdamped second-order step response
wd = wn * math.sqrt(1 - Z * Z)
step = 1 - np.exp(-Z * wn * t) * (np.cos(wd * t) + (Z * wn / wd) * np.sin(wd * t))
print("      overshoot of the fitted response: %.2f%%   (the median of the footage's five: %.2f%%)"
      % (100 * float(np.max(step) - 1.0), 100 * OVERSHOOTS[2][1]))
print("      time to the first peak: %.2f s" % float(t[int(np.argmax(step))]))
print("      time to 63%%: %.2f s   (the model's first-order tau_s is %.2f s, so the order change barely moves it)"
      % (float(t[np.argmax(step >= 0.632)]), TAU))
print()
print("   so the loop's damping would be zeta = %.3f, giving the footage's median overshoot." % Z)
print("   The spread (0.065-0.370) and the ordering are the model's remaining error, and are recorded as such.")
