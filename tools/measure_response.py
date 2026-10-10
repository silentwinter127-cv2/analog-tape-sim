"""
Measure the signal-domain frequency response of the chain (calibration check).

For each medium parameter that claims a bandwidth in MHz, this measures the MTF
the pipeline really applies, so the numbers in the UI mean something physical
rather than "some blur".

Method: render a single spatial tone, then read the amplitude at that exact
frequency with a DFT against the exact basis.  This avoids the pitfalls of
trying to compare two different bandwidths against each other.

Run:  python tools/measure_response.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atsim.core.params import merge_values
from atsim.core.pipeline import FrameBuffer
from atsim.core.profiles import CAMERA_PROFILES, DISPLAY_PROFILES
from atsim.core.signal import rgb_to_yiq
from atsim.render.renderer import Renderer

LINE_HZ = 15734.264


class ToneSource:
    """One spatial frequency, luma-only or colour-only, over a few frames."""

    def __init__(self, width=720, height=480, freq_cycles_per_line=40.0,
                 chroma=False, frames=8, fps=30000.0 / 1001.0):
        self.width, self.height = width, height
        self.frame_count = frames
        self.fps = fps
        self.freq = freq_cycles_per_line
        self.chroma = chroma
        from atsim.io.ffmpeg import VideoInfo
        self.info = VideoInfo(path="<tone>", width=width, height=height, fps=fps,
                              frame_count=frames, duration=frames / fps,
                              codec="synthetic")

    def frame(self, index: int) -> np.ndarray:
        h, w = self.height, self.width
        x = np.arange(w, dtype=np.float32)[None, :]
        c = np.cos(2.0 * np.pi * self.freq * x / w)
        img = np.empty((h, w, 3), dtype=np.float32)
        if self.chroma:
            # colour-only tone: R and B in antiphase so luma stays constant
            img[..., 0] = 0.5 + 0.25 * c
            img[..., 1] = 0.5
            img[..., 2] = 0.5 - 0.25 * c
        else:
            img[...] = (0.5 + 0.40 * c)[..., None]
        return np.clip(img, 0, 1).astype(np.float32)

    def prefetch(self, *a): pass
    def warm(self, *a): pass
    def clear(self): pass


def amp_at(plane: np.ndarray, freq: float) -> float:
    """Amplitude of the component at `freq` cycles/line, on a mid-frame line."""
    h, w = plane.shape
    x = np.arange(w, dtype=np.float64)
    basis = np.exp(-2j * np.pi * freq * x / w)
    line = plane[int(h * 0.5)].astype(np.float64)
    return float(abs(np.dot(line - line.mean(), basis)) / (w / 2.0))


DISABLED = {"tape": {"noise_floor": 0.0, "line_jitter": 0.0, "micro_jitter": 0.0,
                     "walk_amplitude": 0.0, "vertical_instability": 0.0,
                     "dropout_rate": 0.0, "wear": 0.0, "crease_rate": 0.0,
                     "head_switching": 0.0},
            "display": {"enabled": 0.0},
            "player": {"cross_color": 0.0, "cross_luma": 0.0, "line_delay": 0.0,
                       "frequency_response": 0.0, "head_wear": 0.0,
                       "tbc_enabled": 1.0, "tbc_residual": 0.0, "transport_jitter": 0.0,
                       "output_gain": 0.0, "output_bias": 0.0},
            "format": {"color_noise": 0.0, "edge_boost": 0.0,
                       "cross_color": 0.0, "cross_luma": 0.0,
                       "chroma_vertical_averaging": 0.0}}


def base_layers() -> dict:
    return merge_values(
        merge_values(CAMERA_PROFILES["clean_digital"]["layers"],
                     DISPLAY_PROFILES["none"]["layers"]),
        DISABLED)


def measure(freq_cycles_per_line: float, chroma: bool, overrides: dict) -> float:
    layers = merge_values(base_layers(), overrides)
    src = ToneSource(freq_cycles_per_line=freq_cycles_per_line, chroma=chroma)
    r = Renderer(src, src.width, src.height, src.fps, layers, seed=101)
    out = r.render_frame(3)
    if chroma:
        _, i, q = rgb_to_yiq(out)
        plane = np.hypot(i, q)
    else:
        plane = rgb_to_yiq(out)[0]
    return amp_at(plane, freq_cycles_per_line)


def main() -> int:
    print("=" * 80)
    print("LUMA response: amplitude out / amplitude in, at the test frequency")
    print("=" * 80)
    print(f"{'MHz':>6} {'cyl':>6} " + "".join(f"{n:>10}" for n in
                                              ("1.5MHz", "3.0MHz", "4.2MHz", "5.0MHz")))
    for mhz in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.2, 5.0):
        cyl = mhz * 1e6 / LINE_HZ
        if cyl > 355:
            continue
        ref = measure(cyl, False, {"format": {"luma_bandwidth": 6.0}})
        row = [measure(cyl, False, {"format": {"luma_bandwidth": bw}})
               for bw in (1.5, 3.0, 4.2, 5.0)]
        ratio = [v / ref if ref > 1e-9 else 0.0 for v in row]
        print(f"{mhz:6.1f} {cyl:6.1f} " + "".join(f"{v:10.3f}" for v in ratio))

    print()
    print("=" * 80)
    print("CHROMA response: amplitude out / amplitude in (colour tone, luma flat)")
    print("=" * 80)
    print(f"{'MHz':>6} {'cyl':>6} " + "".join(f"{n:>10}" for n in
                                              ("0.7MHz", "1.0MHz", "1.3MHz")))
    for mhz in (0.2, 0.4, 0.7, 1.0, 1.3, 1.6):
        cyl = mhz * 1e6 / LINE_HZ
        if cyl > 355:
            continue
        ref = measure(cyl, True, {"format": {"chroma_bandwidth": 2.0,
                                             "chroma_vertical_averaging": 0.0}})
        row = [measure(cyl, True, {"format": {"chroma_bandwidth": bw,
                                              "chroma_vertical_averaging": 0.0}})
               for bw in (0.7, 1.0, 1.3)]
        ratio = [v / ref if ref > 1e-9 else 0.0 for v in row]
        print(f"{mhz:6.1f} {cyl:6.1f} " + "".join(f"{v:10.3f}" for v in ratio))

    print()
    print("Reading: each column should be ~1.0 well below its corner and fall through")
    print("~0.7 near the corner (the -3 dB convention documented in core/signal.py).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
