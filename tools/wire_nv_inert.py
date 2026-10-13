"""
Wire the three night-vision parameters that were declared but never read.

**Measured, each over its full declared range, changing nothing at all -- the frames were bit-identical:**

    nv_green_gain     0.5 to 1.6     identical frame
    nv_read_noise     0.0 to 0.2     identical frame
    nv_ir_response    0.0            identical frame

**All three are the "declared, reachable, and never read" fault**, which this project has now found in `agc_range`,
`chroma_under_luma`, `player.line_delay`, `tape.row_chroma_*`, `player.chroma_vert_blend`, `crt_age`, `convergence`,
`mask_depth`, and in `nv_*` itself before it reached the schema. **It is worth stating that this check is the reason
they were found at all**: reading the code would have shown the fields being *declared* and the module *referencing
them nowhere*, which is suggestive -- but varying them and comparing frames is what makes it a fact.

**What each one should do, physically:**

  * **`nv_green_gain`** scales the green channel in the monochrome view. A viewfinder's green gun has its own gain, and
    the reference's ratio is 0.29:0.46:0.25, so a gain above 1 pushes the picture further onto the green axis;
  * **`nv_ir_response`** is how much of the near-infrared term the sensor admits -- **the fraction of the IR-cut filter
    that has been removed.** At 0 the filter is in place and NightShot should show no IR contribution at all, which is
    the parameter's whole meaning and is currently ignored;
  * **`nv_read_noise`** adds the sensor's read noise at the mode's gain. **It must be added before accumulation**, so
    that summing frames reduces it -- which is the entire reason Super NightShot exists. Adding it afterwards would
    make accumulation unable to touch it, and the mode's headline property would be absent from the noise model.

**That last point is the one with content.** Read noise applied before the sum falls as `1/sqrt(N)`; applied after, it
does not fall at all. So the placement is not cosmetic: it decides whether the model can show why a night-vision camera
has an accumulating mode.
"""
from __future__ import annotations

import ast
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = os.path.join(ROOT, "atsim", "codecs", "nightvision.py")
s = open(P, encoding="utf-8").read()

# ---------------------------------------------------------------- 1. green gain
OLD_G = """    weights = np.array([0.29, 0.46, 0.25], dtype=np.float32)
    weights = weights / float(weights @ LUMA)
    mono = np.clip(lum[..., None] * weights, 0.0, 1.0)
    return (rgb * (1.0 - m) + mono * m).astype(np.float32)"""
NEW_G = """    weights = np.array([0.29, 0.46, 0.25], dtype=np.float32)
    weights = weights / float(weights @ LUMA)
    # **The green gun's own gain.**  `nv_green_gain` was declared and never read -- measured, 0.5 and 1.6 gave
    # bit-identical frames.  It scales the green channel alone, which is what a viewfinder's green gain does, and then
    # the weights are re-normalised so the change is a **hue shift and not a brightness change**: without that, raising
    # the gain would also brighten the picture and the two effects could not be told apart.
    g = float(np.clip(spec.green_gain, 0.1, 4.0))
    weights = weights * np.array([1.0, g, 1.0], dtype=np.float32)
    weights = weights / float(weights @ LUMA)
    mono = np.clip(lum[..., None] * weights, 0.0, 1.0)
    return (rgb * (1.0 - m) + mono * m).astype(np.float32)"""
assert OLD_G in s, "the green view was not found"
s = s.replace(OLD_G, NEW_G, 1)

# ---------------------------------------------------------------- 2. IR response gates the IR term
OLD_IR = """    m = float(np.clip(spec.ir_albedo_mix, 0.0, 1.0))
    if m < 1e-4:
        return rgb"""
NEW_IR = """    # **`nv_ir_response` is the fraction of the IR-cut filter that has been removed**, and it was declared and never
    # read -- measured, 0.0 gave a bit-identical frame, so turning the filter back on changed nothing.  It gates the
    # whole near-infrared term, which is the parameter's entire meaning: at 0 the filter is in place and the sensor
    # sees what a normal camera sees.
    m = float(np.clip(spec.ir_albedo_mix, 0.0, 1.0)) * float(np.clip(spec.ir_response, 0.0, 1.0))
    if m < 1e-4:
        return rgb"""
assert OLD_IR in s, "the ir_highlights guard was not found"
s = s.replace(OLD_IR, NEW_IR, 1)

# ---------------------------------------------------------------- 3. read noise, before accumulation
NEW_FN = '''

def read_noise(rgb: np.ndarray, spec: NightVisionSpec, seed: int = 0) -> np.ndarray:
    """
    The sensor's read noise at the mode's gain -- **applied before accumulation, so summing frames reduces it.**

    **Why the placement carries the meaning.** Read noise is added per frame at the sensor, so summing N frames adds it
    N times in quadrature while the signal adds N times coherently: the ratio improves as `sqrt(N)`. Applied *after*
    accumulation it would not fall at all, and the model would be unable to show why a night-vision camera has an
    accumulating mode in the first place. **That is the whole content of Super NightShot**, so where this runs is not a
    detail.

    `nv_read_noise` was declared and never read -- measured, 0.0 and 0.2 gave bit-identical frames.

    The noise is **deterministic in the frame index**, so a still picture is stable across renders and a sequence
    differs frame to frame, which is what an accumulator needs to have something to average.
    """
    a = float(np.clip(spec.read_noise, 0.0, 0.5))
    if a < 1e-6:
        return rgb
    rng = np.random.default_rng(0x9E3779B9 ^ (int(seed) * 2654435761 & 0xFFFFFFFF))
    return np.clip(rgb + rng.normal(0.0, a, rgb.shape).astype(np.float32), 0.0, 1.0)
'''
s = s + NEW_FN
ast.parse(s)
open(P, "w", encoding="utf-8").write(s)
print("wired nv_green_gain and nv_ir_response; added read_noise")

# ---------------------------------------------------------------- 4. call it, before accumulation
P2 = os.path.join(ROOT, "atsim", "stages", "camera.py")
s2 = open(P2, encoding="utf-8").read()
OLD_IMP = ("        from ..codecs.nightvision import (accumulate, add_illuminator, ir_highlights,\n"
           "                                           mode_of, monochrome_green)")
NEW_IMP = ("        from ..codecs.nightvision import (accumulate, add_illuminator, ir_highlights,\n"
           "                                           mode_of, monochrome_green, read_noise)")
assert OLD_IMP in s2, "the import was not found"
s2 = s2.replace(OLD_IMP, NEW_IMP, 1)

OLD_ACC = """            n = int(round(float(np.clip(self.nv.accumulate, 1.0, 64.0))))
            out = rgb
            if n > 1:
                stack = self._nv_source_frames(n, index)
                if len(stack) > 1:
                    out = accumulate(stack, self.nv)"""
NEW_ACC = """            n = int(round(float(np.clip(self.nv.accumulate, 1.0, 64.0))))
            out = rgb
            if n > 1:
                stack = self._nv_source_frames(n, index)
                if len(stack) > 1:
                    # **The mode's read noise is added to each frame BEFORE the sum**, so accumulation reduces it --
                    # which is the whole reason Super NightShot exists.  Read noise adds per frame at the sensor, so N
                    # frames add it N times in quadrature while the signal adds N times coherently and the ratio
                    # improves as sqrt(N).  Applied after the sum it would not fall at all, and the model could not
                    # show why the camera has an accumulating mode.
                    idx = int(index) if index is not None else 0
                    stack = [read_noise(np.asarray(f, np.float32), self.nv, idx - (len(stack) - 1 - j))
                             for j, f in enumerate(stack)]
                    out = accumulate(stack, self.nv)
            else:
                out = read_noise(out, self.nv, int(index) if index is not None else 0)"""
assert OLD_ACC in s2, "the accumulation block was not found"
s2 = s2.replace(OLD_ACC, NEW_ACC, 1)
ast.parse(s2)
open(P2, "w", encoding="utf-8").write(s2)
print("read noise is applied per frame, before the sum")

import subprocess, sys
r = subprocess.run([sys.executable, "tools/nv_inert_check.py"], capture_output=True, text=True,
                   encoding="utf-8", errors="replace", cwd=ROOT)
print()
print(r.stdout[-1400:] if r.stdout else "")
if r.returncode != 0:
    print("STDERR:", (r.stderr or "")[-1200:])
