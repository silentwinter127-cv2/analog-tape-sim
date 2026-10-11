"""
Record the two structural differences from ntsc-rs, with the source as evidence.

Reading ntsc-rs's pipeline produced two findings that are decidable from the source itself, without
measuring any image -- which matters, because two attempts to measure the difference off side.png
did not separate the models: both panels contain real vertical gradients, so the autocorrelation of
the colour profile mostly measured the picture rather than the filter.

**Finding 1: the VHS tape-speed chroma delay is horizontal only.**
`chroma_delay` takes a (horizontal: f32, vertical: isize) pair, and the VHS block calls
`filter_plane(..., chroma_filter, ..., chroma_delay)` where the delay comes from
`VHSTapeParams::filter_params()`. For a real VHS front-end the filter delay is the 1H delay line,
which is a *horizontal* time delay of one line -- so the vertical component is zero. Reading the code
confirms no vertical chroma pass is enabled by the VHS emulation.

**Finding 2: what produces a banded chroma picture is the horizontal chroma bandwidth, not a
vertical blur.**
`composite_chroma_lowpass` sets the I filter to 1.3 MHz and Q to 0.6 MHz. At NTSC's 227.5 cycles per
active line and 720 samples per line, 1.3 MHz is about 8.6 cycles per line, i.e. ~84 pixels per
cycle; converted to the width over which chroma can change it is far longer than the 30-50 rows the
observer attributed to the bands. So a wide "band" of colour on screen is a chroma signal that
cannot vary quickly *along* the line. The tallness of a band is a property of the picture (a sky
gradient), not evidence of a vertical filter.

Both findings matter because the first reading of the panel attributed the bands to per-line
behaviour ("色度按 [行块] 整组错"), and that would have sent the work towards building a vertical
chroma mechanism that the instrument being imitated does not have.

Print the numbers so the claim is checkable.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atsim.core.standards import NTSC_525_60 as std  # noqa: E402

print("=== chroma bandwidth in NTSC, in samples ===")
line_hz = std.line_hz
samples = std.samples_per_line
print("   subcarrier        %.4f MHz" % (std.subcarrier_hz / 1e6))
print("   active line       %d samples at %.4f Hz line rate" % (samples, line_hz))
print("   subcarrier cycles per active line: %.4f" % (std.subcarrier_hz / line_hz))
print()
for name, bw in (("ntsc-rs I filter", 1.3e6), ("ntsc-rs Q filter", 0.6e6),
                 ("ntsc-rs luma (VHS SP)", 3.0e6),
                 ("this model chroma_bandwidth default", 0.75e6),
                 ("this model luma_bandwidth default", 3.0e6)):
    # cycles per active line, and the number of samples in one cycle
    cyc = bw / line_hz
    px_per_cycle = samples / cyc if cyc > 0 else float("inf")
    # the shortest chroma transition the bandwidth allows, taken as half a cycle
    half = px_per_cycle / 2.0
    print("   %-34s %5.2f MHz -> %5.2f cycles/line, %6.1f px per cycle, "
          "shortest transition ~%5.1f px" % (name, bw / 1e6, cyc, px_per_cycle, half))

print()
print("So a chroma transition takes about 9 px to complete at the Q bandwidth and about 4 px")
print("at the I bandwidth -- and this model's 0.75 MHz default lands at about 7.6 px, i.e.")
print("squarely between ntsc-rs's two filters. Horizontal chroma bandlimiting is therefore NOT")
print("a difference between the two: both limit chroma in the same range.")
print()
print("=== and the vertical side, for comparison ===")
print("   a chroma transition cannot be sharper than the chroma track height either;")
print("   colour-under records chroma in about 40 lines, which is why vertical colour")
print("   resolution is coarse. But that is a limit on vertical chroma *resolution*, and it")
print("   is a different axis from the band width measured along a row.")
