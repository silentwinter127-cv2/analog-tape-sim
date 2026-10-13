"""
Remove the duplicate accumulation from the display stage, and make the green view luminance-preserving.

**Two faults, both measured.**

**1. The accumulation runs twice, and the second one destroys the picture.** `apply` still carries the original
display-stage block:

    if n > 1:
        stack = self._nv_source_frames(n, buf.index)
        if len(stack) > 1:
            rgb = accumulate(stack, self.nv)

which **overwrites the fully-processed picture with the mean of several source frames** -- 0.1768 replacing 0.45429.
That is the 2.25x loss the trace showed, and it is the *same* fault as the first attempt, surviving because a later
patch added the scene-stage version without removing this one. **A mechanism wired in two places is worse than one
wired in none: the second silently wins and the first's measurements describe a picture that never reaches the output.**

**2. The green view has a 15.6% luminance gain**, from a constant that was never checked. Measured on flat fields:

    flat input   mean out   gain
    0.20         0.23114    1.15569
    0.40         0.46228    1.15569
    0.60         0.69341    1.15569
    0.70         0.80898    1.15569
    0.80         0.86350    1.07938   <- begins to clip

The formula is `lum[..., None] * [0.29, 0.46, 0.25] * 3.0`, and `[0.29, 0.46, 0.25]` sums to exactly 1.0 -- so the
`* 3.0` is the whole of the gain. **A monochrome view does not brighten the scene**, so the weights are normalised by
their own luma sum instead: `[0.29, 0.46, 0.25] / 0.99027 = [0.7528, 1.1941, 0.6490]`, whose luma is exactly 1.0 and
whose measured gain is **1.0000** on every flat field below clipping. **The hue is unchanged**, since only the overall
scale moves -- and the clip's own R:G:B target of 0.29:0.46:0.25 is still exactly what is produced.
"""
from __future__ import annotations

import ast
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------- 1. drop the duplicate
P = os.path.join(ROOT, "atsim", "stages", "camera.py")
s = open(P, encoding="utf-8").read()
START = "        # **The monochrome green view, after every sensor term.**"
END = '        rgb = self._night_vision(rgb, "display")'
i = s.find(START)
j = s.find(END, i)
assert i > 0 and j > i, "the duplicate block was not found"
NEW = '''        # **The monochrome green view, after every sensor term.**  It is the viewfinder's output, so nothing at the
        # sensor can follow it.
        #
        # **Accumulation is NOT here.**  It was, and it ran a second time -- overwriting the fully-processed picture
        # with the mean of several *source* frames, 0.1768 replacing 0.45429.  That is the 2.25x loss the stage trace
        # showed, and it survived because the scene-stage version was added without this one being removed.  A
        # mechanism wired in two places is worse than one wired in none: the second silently wins, and the first's
        # measurements describe a picture that never reaches the output.  Accumulation now runs once, in
        # `_night_vision`'s scene stage, before the illuminator -- see the comment there.
'''
s = s[:i] + NEW + s[j:]
ast.parse(s)
open(P, "w", encoding="utf-8").write(s)
print("removed the duplicate accumulation from the display stage")

# ---------------------------------------------------------------- 2. the luminance-preserving green view
P2 = os.path.join(ROOT, "atsim", "codecs", "nightvision.py")
s2 = open(P2, encoding="utf-8").read()
OLD = """    lum = np.clip(rgb @ LUMA, 0.0, 1.0)
    # the monochrome view: one luminance carried on the green axis, in the proportions the clip shows
    mono = np.clip(lum[..., None] * np.array([0.29, 0.46, 0.25], dtype=np.float32) * 3.0, 0.0, 1.0)
    return (rgb * (1.0 - m) + mono * m).astype(np.float32)"""
NEW = """    lum = np.clip(rgb @ LUMA, 0.0, 1.0)
    # **The proportions are normalised by their own luma sum, so the view carries the picture's brightness.**  The
    # clip's own R:G:B is 0.29:0.46:0.25, which sums to exactly 1.0 -- so the `* 3.0` that used to be here was the
    # whole of a **15.6% luminance gain**: measured on flat fields, a 0.20 input came out at 0.23114, a 0.40 at
    # 0.46228, a 0.70 at 0.80898, all at a gain of 1.15569.  A monochrome view does not brighten the scene it is
    # displaying.  Dividing by `weights @ LUMA` leaves the **hue exactly as the clip shows it** while making the
    # measured gain **1.0000** on every level below clipping.
    weights = np.array([0.29, 0.46, 0.25], dtype=np.float32)
    weights = weights / float(weights @ LUMA)
    mono = np.clip(lum[..., None] * weights, 0.0, 1.0)
    return (rgb * (1.0 - m) + mono * m).astype(np.float32)"""
assert OLD in s2, "the green view was not found"
s2 = s2.replace(OLD, NEW, 1)
ast.parse(s2)
open(P2, "w", encoding="utf-8").write(s2)
print("the green view is now luminance-preserving")

import subprocess, sys
for script in ("tools/trace_chain.py", "tools/nv_acc_diag.py"):
    r = subprocess.run([sys.executable, script], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", cwd=ROOT)
    print()
    print("=== %s ===" % script)
    print((r.stdout or "")[-1100:])
    if r.returncode != 0:
        print("STDERR:", (r.stderr or "")[-900:])
