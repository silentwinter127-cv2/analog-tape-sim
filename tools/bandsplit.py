"""
Measure `fft_bandpass` on a column-constant ramp, because `luma = comp - chroma` presumes it returns near zero there.

**Why this comes before touching the division again.**  Three attempts to correct the chroma gain each fixed the polarity
and broke the contrast, which means the model being corrected was wrong rather than the correction being mistuned.  **And
one number points at the deeper fault**: with the division removed, `luma = comp - chroma` came out at **0.27 of the
input's spread** on a ramp that is constant down every column.  For such a signal the band split should put almost
everything in the luma band and almost nothing in the chroma band -- so `chroma` should be near zero and `luma` should
come back at *its own* spread, not a quarter of it.

**So the band split itself is measured**, on the exact signal the stage feeds it, with no division and no comb involved.
The quantities are the chroma band's own spread, its mean, and the luma's spread as a fraction of the composite's --
and the correct values for a column-constant picture are known in advance: **chroma near zero, luma near 1.0.**

**And the same measurement is taken on a signal that *should* produce chroma** -- a picture with colour that alternates
between lines, which is what the subcarrier carries -- so the two cases can be told apart.  **If the split gives chroma
on the first, the fault is in the split; if it does not, the fault is in what the stage hands it.**
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from atsim.core.signal import fft_bandpass, split_composite, rgb_to_yiq, remod_chroma  # noqa: E402
from atsim.core.standards import STANDARDS                                     # noqa: E402

COEF = np.array([0.299, 0.587, 0.114], np.float32)
W, H = 320, 240
std = STANDARDS["ntsc"] if isinstance(STANDARDS, dict) else STANDARDS
fsc = std.subcarrier_hz
print("   the standard: %s   subcarrier %.4f MHz   pixel rate %.4f MHz"
      % (getattr(std, "name", std), fsc / 1e6, std.pixel_rate_hz / 1e6))

#: a black-to-white horizontal ramp: constant down every column, so no vertical detail
ramp = np.zeros((H, W, 3), np.float32)
ramp[:] = np.linspace(0.0, 1.0, W, dtype=np.float32)[None, :, None]
#: the same shape as a composite signal -- a luma with no chroma at all
ramp_comp = np.clip(ramp @ COEF, 0, 1)

print()
print("   === 1. the raw band split, on a column-constant picture ===")
print("      the input's own spread: %.5f" % float(ramp_comp.std()))
for bw in (0.35e6, 0.8e6, 3.0e6):
    ch = fft_bandpass(ramp_comp, max(0.10e6, fsc - 1.6 * bw), fsc + 1.6 * bw, std, unity_at_hz=fsc)
    ch = np.asarray(ch, np.float32)
    lum = ramp_comp - ch
    print("      chroma window +-%.2f MHz -> chroma spread %.5f (%.3f of input), luma spread %.5f (%.3f)"
          % (1.6 * bw / 1e6, float(ch.std()), float(ch.std()) / float(ramp_comp.std()),
             float(lum.std()), float(lum.std()) / float(ramp_comp.std())))
print("      for a column-constant picture the chroma band should be near ZERO and the luma near 1.0.")

print()
print("   === 2. and the full split, with the comb off, on the same input ===")
for cd in (0.0, 0.55):
    lum, ch = split_composite(ramp_comp, std, comb_depth=cd, chroma_bandwidth_hz=0.8e6)
    lum = np.asarray(lum, np.float32)
    ch = np.asarray(ch, np.float32)
    print("      comb_depth %.2f -> chroma spread %.5f, luma spread %.5f (%.3f of input), luma mean %.5f"
          % (cd, float(ch.std()), float(lum.std()), float(lum.std()) / float(ramp_comp.std()), float(lum.mean())))

print()
print("   === 3. a signal that SHOULD carry chroma, as the control ===")
#: colour bars: the chroma changes across the picture, which is what the subcarrier carries
bars = np.zeros((H, W, 3), np.float32)
for k in range(8):
    bars[:, k * W // 8:(k + 1) * W // 8] = [(k % 2), ((k // 2) % 2), ((k // 4) % 2)]
y, i, q = rgb_to_yiq(bars)
comp_bars = np.clip(y + remod_chroma(i, q, std, chroma_gain=1.0), 0, 2)
lum, ch = split_composite(comp_bars, std, comb_depth=0.0, chroma_bandwidth_hz=0.8e6)
print("      colour bars: composite spread %.5f -> chroma spread %.5f (%.3f), luma spread %.5f (%.3f)"
      % (float(comp_bars.std()), float(np.asarray(ch).std()),
         float(np.asarray(ch).std()) / float(comp_bars.std()),
         float(np.asarray(lum).std()), float(np.asarray(lum).std()) / float(comp_bars.std())))
print("      here the chroma band SHOULD carry energy, so a small value would mean the split is broken both ways.")

print()
print("   === 4. the subcarrier's own phase between lines, which is what the split depends on ===")
ph = np.asarray([std.subcarrier_phase(y) for y in range(6)], np.float32)
print("      the first six lines' phases: %s" % np.round(ph, 4))
print("      consecutive differences: %s" % np.round(np.diff(ph), 4))
print("      a phase step near +-pi between lines is what makes the 1H comb cancel the carrier.")
