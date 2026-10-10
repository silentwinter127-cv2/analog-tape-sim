"""
Broadcast / video standards: the physical time base that everything else is
derived from.

Every spatial filter in this engine is expressed in *pixel* units, but tape
formats are specified in *MHz* of analog bandwidth.  A standard is what lets us
convert one into the other, and therefore what makes the simulation
"mechanism driven" rather than "slider driven".

All timing values are taken from the respective broadcast specifications.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class VideoStandard:
    name: str
    # --- timing -------------------------------------------------------
    line_hz: float           # horizontal line rate [Hz]
    field_hz: float          # field rate [Hz]
    frames_per_second: float
    fields_per_frame: int
    lines_per_frame: int     # total lines incl. blanking, per frame
    active_lines: int        # lines carrying picture per frame
    samples_per_line: int    # active digital samples per active line
    # --- color --------------------------------------------------------
    subcarrier_hz: float     # chroma subcarrier [Hz]
    color_system: str        # "NTSC" | "PAL"
    # --- bandwidth ----------------------------------------------------
    luma_bandwidth_hz: float      # nominal system luma bandwidth [Hz]
    chroma_bandwidth_hz: float    # nominal system chroma bandwidth [Hz]
    # --- blanking -----------------------------------------------------
    front_porch_us: float = 1.5
    sync_us: float = 4.7
    back_porch_us: float = 4.7   # contains color burst

    # ------------------------------------------------------------------
    @property
    def subcarrier_rad_per_sample(self) -> float:
        """Subcarrier phase advance per active sample [rad]."""
        return 2.0 * 3.141592653589793 * self.subcarrier_hz / self.line_hz / self.samples_per_line

    @property
    def active_line_us(self) -> float:
        """Duration of the active part of one line [us]."""
        return (1.0 / self.line_hz) * 1e6

    @property
    def pixel_rate_hz(self) -> float:
        """Digital sampling rate of one active line [Hz]."""
        return self.line_hz * self.samples_per_line

    def pixels_for_bandwidth(self, bandwidth_hz: float, width: int | None = None) -> float:
        """
        Convert an analog bandwidth in Hz into a Gaussian sigma in pixels.

        The nominal active line is sampled at `pixel_rate_hz` by the digital
        standard; if the delivered image is narrower/wider we scale the sigma
        proportionally so the *relative* detail loss is unchanged.

        A Gaussian's -3 dB point sits at f3db = 0.1325 cycles/sample, i.e.
        sigma = 0.1325 / (bandwidth / sample_rate).
        """
        if bandwidth_hz <= 0:
            return 1e6
        f_norm = bandwidth_hz / self.pixel_rate_hz          # cycles / sample
        if f_norm <= 1e-9:
            return 1e6
        sigma = 0.1325 / f_norm
        if width is not None and width > 0:
            sigma *= self.samples_per_line / float(width)
        return float(max(0.30, min(sigma, 1e5)))

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Digital standards we actually implement.
#
# Rec.601 525/60 (NTSC): 858 total samples/line of which 720 active,
# 525 total lines of which 486 nominal / 480 digital active.
# Rec.601 625/50 (PAL): 864 total samples/line, 720 active, 625 lines,
# 576 active.
# ---------------------------------------------------------------------------
NTSC_525_60 = VideoStandard(
    name="NTSC 525/60 (Rec.601)",
    line_hz=15734.264,
    field_hz=59.94,
    frames_per_second=30000.0 / 1001.0,
    fields_per_frame=2,
    lines_per_frame=525,
    active_lines=480,
    samples_per_line=720,
    subcarrier_hz=3579545.4545,
    color_system="NTSC",
    luma_bandwidth_hz=4.2e6,
    chroma_bandwidth_hz=1.3e6,
)

PAL_625_50 = VideoStandard(
    name="PAL 625/50 (Rec.601)",
    line_hz=15625.0,
    field_hz=50.0,
    frames_per_second=25.0,
    fields_per_frame=2,
    lines_per_frame=625,
    active_lines=576,
    samples_per_line=720,
    subcarrier_hz=4433618.75,
    color_system="PAL",
    luma_bandwidth_hz=5.0e6,
    chroma_bandwidth_hz=1.3e6,
)

STANDARDS: dict[str, VideoStandard] = {
    "ntsc": NTSC_525_60,
    "pal": PAL_625_50,
}


def get_standard(key: str) -> VideoStandard:
    try:
        return STANDARDS[key.lower()]
    except KeyError:
        raise KeyError(f"unknown standard {key!r}; known: {sorted(STANDARDS)}") from None


def standard_for(key_or_standard) -> VideoStandard:
    if isinstance(key_or_standard, VideoStandard):
        return key_or_standard
    return get_standard(str(key_or_standard))
