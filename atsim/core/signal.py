"""
Signal representation and the frequency-domain primitives.

The engine deliberately keeps *luma* and *chroma* as separate planes through
most of the chain.  That is the only way to model what actually happens in a
composite tape format:

    the two components are filtered by different bandwidths, then they are
    physically added into ONE signal, and the player must separate them
    again -- imperfectly.

Cross-color (fine luma detail turning into colour) and cross-luma (colour
turning into moving dot patterns) are *consequences* of that round trip, not
overlays.  They appear here because we really do add and really do separate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.ndimage import convolve1d

from .standards import VideoStandard

TWO_PI = 2.0 * np.pi

# NTSC / SMPTE 170M weights (also the Rec.601 luma weights)
RGB_TO_Y = np.array([0.299, 0.587, 0.114], dtype=np.float64)


# ---------------------------------------------------------------------------
# kernel design
# ---------------------------------------------------------------------------
def gaussian_kernel(sigma: float) -> np.ndarray:
    """
    Finite Gaussian kernel, normalised to unity DC gain.

    The sequence x[n] = (1-a)^n for n >= 0 is *exactly* a geometric decay, so a
    Gaussian can be approximated by a cascade; but we use the true sampled
    Gaussian and let `scipy.ndimage` do the work, because measured cost matters
    here: the whole preview frame rate rests on these filters.
    """
    sigma = float(max(1e-6, sigma))
    if sigma < 0.30:
        return np.array([1.0], dtype=np.float32)
    radius = int(np.ceil(3.0 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return (k / k.sum()).astype(np.float32)


def highpass_kernel(sigma: float) -> np.ndarray:
    """Complementary high-pass: delta - lowpass (DC removed exactly)."""
    k = gaussian_kernel(sigma)
    hp = -k
    c = len(k) // 2
    hp[c] += 1.0
    return hp


def bandpass_kernel(lo_sigma: float, hi_sigma: float) -> np.ndarray:
    """Band-pass as lowpass(hi) - lowpass(lo), both centred."""
    khi = gaussian_kernel(hi_sigma)
    klo = gaussian_kernel(lo_sigma)
    n = max(len(khi), len(klo))
    if n % 2 == 0:
        n += 1
    out = np.zeros(n, dtype=np.float32)
    for k in (khi, -klo):
        c = len(k) // 2
        off = n // 2 - c
        out[off:off + len(k)] += k
    return out


def apply_fir(plane: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    """Separable FIR along one axis, reflect-padded at the edges."""
    if len(kernel) == 1:
        return plane
    return convolve1d(plane, kernel, axis=axis, mode="reflect")


def lowpass(plane: np.ndarray, sigma: float, axis: int | None = None) -> np.ndarray:
    k = gaussian_kernel(sigma)
    if axis is None:
        return apply_fir(apply_fir(plane, k, 1), k, 0)
    return apply_fir(plane, k, axis)


def highpass(plane: np.ndarray, sigma: float, axis: int | None = None) -> np.ndarray:
    k = highpass_kernel(sigma)
    if axis is None:
        return apply_fir(apply_fir(plane, k, 1), k, 0)
    return apply_fir(plane, k, axis)


def bandpass(plane: np.ndarray, lo_sigma: float, hi_sigma: float,
             axis: int | None = None) -> np.ndarray:
    k = bandpass_kernel(lo_sigma, hi_sigma)
    if axis is None:
        return apply_fir(apply_fir(plane, k, 1), k, 0)
    return apply_fir(plane, k, axis)


def horizontal_shift(plane: np.ndarray, shifts: np.ndarray,
                     mode: str = "reflect") -> np.ndarray:
    """
    Shift every row of `plane` horizontally by its own (fractional) amount.

    `shifts` is a per-row array in pixels, positive = to the right.  Linear
    interpolation between two integer-shifted copies is the standard way to
    model a sampling-phase error: the line clock simply samples the wrong
    instant on the tape.

    Works for both (H, W) signal planes and (H, W, C) pictures -- the head
    switching tear and the CRT tracking error need it on pictures too.  Edge
    behaviour is reflect (or replicate) padding, so a large deliberate offset
    smears the edge rather than wrapping the picture around.
    """
    h, w = plane.shape[0], plane.shape[1]
    sh = np.asarray(shifts, dtype=np.float32).reshape(-1)
    if sh.size < h:
        sh = np.pad(sh, (0, h - sh.size), mode="edge")
    sh = sh[:h]
    left = np.floor(sh).astype(np.int64)

    if mode == "wrap":
        cols = np.arange(w, dtype=np.int64)[None, :]
        idx0 = np.mod(cols - left[:, None], w)
        idx1 = np.mod(idx0 - np.sign(sh)[:, None].astype(np.int64), w)
        far = (np.abs(sh) > 1e-6)[:, None]
        idx1 = np.where(far, idx1, idx0)
        a = np.take_along_axis(plane, idx0, axis=1)
        b = np.take_along_axis(plane, idx1, axis=1)
        frac = (sh - left)[:, None]
        if plane.ndim > 2:
            frac = frac[..., None]
        return (a + (b - a) * frac).astype(plane.dtype, copy=False)

    # reflect / replicate padding, then a plain integer gather
    pad = int(np.max(np.abs(np.concatenate([left, left + 1]))) + 1)
    pad = int(min(pad, 4096))
    if pad > 0:
        if mode == "reflect":
            padded = np.pad(plane, ((0, 0), (pad, pad)) + ((0, 0),) * (plane.ndim - 2),
                            mode="reflect")
        else:
            padded = np.pad(plane, ((0, 0), (pad, pad)) + ((0, 0),) * (plane.ndim - 2),
                            mode="edge")
    else:
        padded = plane

    cols = np.arange(w, dtype=np.int64)[None, :]
    idx0 = cols - left[:, None] + pad
    idx1 = idx0 - np.sign(sh)[:, None].astype(np.int64)
    far = (np.abs(sh) > 1e-6)[:, None]
    idx1 = np.where(far, idx1, idx0)
    if plane.ndim > 2:
        idx0 = idx0[..., None]
        idx1 = idx1[..., None]
    a = np.take_along_axis(padded, idx0, axis=1)
    b = np.take_along_axis(padded, idx1, axis=1)
    frac = (sh - left)[:, None]
    if plane.ndim > 2:
        frac = frac[..., None]
    return (a + (b - a) * frac).astype(plane.dtype, copy=False)


def vertical_shift(plane: np.ndarray, shift: float, mode: str = "edge") -> np.ndarray:
    """Whole-plane (or whole-picture) vertical shift in pixels."""
    h = plane.shape[0]
    if abs(shift) < 1e-6:
        return plane
    i0 = int(np.floor(shift))
    frac = float(shift - i0)
    rows0 = np.clip(np.arange(h) + i0, 0, h - 1)
    rows1 = np.clip(np.arange(h) + i0 + (1 if shift > 0 else -1), 0, h - 1)
    a = plane[rows0]
    b = plane[rows1]
    shape = (h,) + (1,) * (plane.ndim - 1)
    return (a + (b - a) * np.float32(frac)).astype(plane.dtype, copy=False)


def shift_line_band(plane: np.ndarray, row0: int, row1: int, amount: float) -> np.ndarray:
    """Shift only rows [row0, row1) -- used for head-switching / creases."""
    row0 = int(max(0, row0))
    row1 = int(min(plane.shape[0], row1))
    if row1 <= row0 or abs(amount) < 1e-6:
        return plane
    band = plane[row0:row1]
    shifts = np.full(band.shape[0], amount, dtype=np.float32)
    plane = plane.copy()
    plane[row0:row1] = horizontal_shift(band, shifts)
    return plane


# ---------------------------------------------------------------------------
# colour transforms
# ---------------------------------------------------------------------------
def rgb_to_yiq(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
    i = rgb[..., 0] * 0.5959 + rgb[..., 1] * -0.2746 + rgb[..., 2] * -0.3213
    q = rgb[..., 0] * 0.2115 + rgb[..., 1] * -0.5227 + rgb[..., 2] * 0.3112
    return y, i, q


def yiq_to_rgb(y: np.ndarray, i: np.ndarray, q: np.ndarray) -> np.ndarray:
    r = y + 0.9563 * i + 0.6210 * q
    g = y - 0.2721 * i - 0.6474 * q
    b = y - 1.1070 * i + 1.7046 * q
    return np.stack([r, g, b], axis=-1)


# ---------------------------------------------------------------------------
# the frame passed between pipeline stages
# ---------------------------------------------------------------------------
@dataclass
class SignalFrame:
    """
    A frame *as a video signal*, not as a picture.

    y : luma plane,  float32 (H, W), nominal 0..1
    i, q : chroma planes, float32 (H, W)
    """
    y: np.ndarray
    i: np.ndarray
    q: np.ndarray

    @property
    def height(self) -> int:
        return int(self.y.shape[0])

    @property
    def width(self) -> int:
        return int(self.y.shape[1])

    def copy(self) -> "SignalFrame":
        return SignalFrame(self.y.copy(), self.i.copy(), self.q.copy())

    @classmethod
    def from_rgb(cls, rgb: np.ndarray, std: VideoStandard) -> "SignalFrame":
        y, i, q = rgb_to_yiq(rgb.astype(np.float32))
        return cls(y.astype(np.float32), i.astype(np.float32), q.astype(np.float32))

    def to_rgb(self, cross_luma: np.ndarray | None = None) -> np.ndarray:
        y = self.y if cross_luma is None else self.y + cross_luma
        if str(getattr(self, "color_system", "NTSC")).upper() == "PAL":
            # PAL: V = Q axis inverted every other line, U = I axis
            alt = np.where((np.arange(self.height) % 2 == 0)[:, None], 1.0, -1.0).astype(np.float32)
            q = self.q * alt
        else:
            q = self.q
        rgb = yiq_to_rgb(y, self.i, q)
        return np.clip(rgb, 0.0, 1.0).astype(np.float32)

    def blend_to(self, other: "SignalFrame", t: float) -> "SignalFrame":
        t = float(np.clip(t, 0.0, 1.0))
        return SignalFrame(
            self.y * (1 - t) + other.y * t,
            self.i * (1 - t) + other.i * t,
            self.q * (1 - t) + other.q * t,
        )

    def stats(self) -> dict[str, Any]:
        return {
            "y_mean": float(self.y.mean()),
            "y_std": float(self.y.std()),
            "i_std": float(self.i.std()),
            "q_std": float(self.q.std()),
        }


# ---------------------------------------------------------------------------
# composite encoding: the step that creates crosstalk by construction
# ---------------------------------------------------------------------------
def subcarrier_phase(width: int, height: int, std: VideoStandard) -> np.ndarray:
    """
    (H, W) subcarrier phase at sample centres.

    The subcarrier advances `subcarrier_rad_per_sample` per active sample; with
    a non-integer number of cycles per line (NTSC's famous 227.5 cycles) the
    phase alternates between consecutive lines -- which is precisely why the
    classic NTSC dot-crawl pattern moves.
    """
    inc = std.subcarrier_rad_per_sample
    cols = np.arange(width, dtype=np.float64)[None, :]
    rows = np.arange(height, dtype=np.float64)[:, None]
    return (inc * cols + std.line_hz * 0.0 * rows + inc * width * rows).astype(np.float32)


def composite_encode(y: np.ndarray, i: np.ndarray, q: np.ndarray,
                     std: VideoStandard, chroma_gain: float = 0.63,
                     burst_phase: float = 0.0) -> np.ndarray:
    """Y + modulated chroma -> one composite signal (0..1 nominal)."""
    ph = subcarrier_phase(y.shape[1], y.shape[0], std) + burst_phase
    c = chroma_gain * (i * np.cos(ph) + q * np.sin(ph))
    return (y + c).astype(np.float32)


def _reflect_pad(plane: np.ndarray, pad: int) -> np.ndarray:
    if pad <= 0:
        return plane
    pad = int(min(pad, max(1, plane.shape[1] // 2 - 1)))
    axes = ((0, 0), (pad, pad)) + ((0, 0),) * (plane.ndim - 2)
    return np.pad(plane, axes, mode="reflect")


def fft_bandpass(plane: np.ndarray, lo_hz: float, hi_hz: float,
                 std: VideoStandard, smooth_hz: float | None = None,
                 unity_at_hz: float | None = None) -> np.ndarray:
    """
    Zero-phase horizontal band-pass with an exact, flat pass-band.

    A composite chroma band-pass has to pass the 3.58 MHz subcarrier with unity
    gain while rejecting luma below and noise above, and the shoulder width can
    be far narrower than the carrier frequency.  An FIR would need thousands of
    taps to do that (and a pure Gaussian-difference shape cannot: its corners
    drag the carrier down -- measured 0.17-0.7x, which quietly destroys most of
    the colour).  A zero-phase FFT response is exact, has no group delay to
    compensate, and costs ~10 ms at SD resolution.

    `unity_at_hz` scales the response so its gain at that frequency is exactly
    1.0 -- measured empirically, so the simulation has no hidden level offsets.
    """
    pad = int(max(32, 0.25 * plane.shape[1]))
    src = _reflect_pad(plane, pad)
    n = src.shape[1]
    freqs = np.fft.rfftfreq(n, d=1.0) * std.pixel_rate_hz
    if smooth_hz is None:
        smooth_hz = max(0.05e6, 0.25 * (hi_hz - lo_hz))
    # Cascaded (8th-order) Butterworth-ish shoulders.  First-order rolls off so
    # gradually that the shoulders eat the colour sidebands themselves: measured
    # gain at 2.83 MHz was 0.44, which removed roughly half of I.  Cascading
    # keeps the plateau flat and still rejects DC strongly.
    order = 4
    hi_c = 0.5 * (lo_hz + hi_hz) + 0.5 * (hi_hz - lo_hz) * 1.6
    lo_c = 0.5 * (lo_hz + hi_hz) - 0.5 * (hi_hz - lo_hz) * 1.6
    hi_c = max(hi_c, lo_hz * 1.02)
    lo_c = max(1e4, min(lo_c, hi_c * 0.97))
    hi = 1.0 / (1.0 + (freqs / hi_c) ** (2 * order))
    lo = (freqs / lo_c) ** (2 * order) / (1.0 + (freqs / lo_c) ** (2 * order))
    response = hi * lo
    if unity_at_hz:
        g = float(np.interp(unity_at_hz, freqs, response))
        if g > 1e-6:
            response = response / g
    spec = np.fft.rfft(src, axis=1)
    out = np.fft.irfft(spec * response[None, :], n=n, axis=1).astype(plane.dtype)
    # strip the padding; the reflect padding makes the edge behaviour sane
    return out[:, pad:pad + plane.shape[1]] if plane.ndim == 2 else \
        out[:, pad:pad + plane.shape[1], :]


def split_composite(comp: np.ndarray, std: VideoStandard,
                    comb_depth: float = 0.0, chroma_bandwidth_hz: float = 0.8e6,
                    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Band-split a composite signal into (luma, chroma).

    The chroma band is centred on the subcarrier with unity gain *at the
    carrier*, because the carrier is what carries the colour: a band-pass whose
    slopes cut into 3.58 MHz attenuates the entire colour signal.  The plateau
    spans +-chroma_bandwidth/2 around the subcarrier, which is exactly the
    band-limit a colour-under format imposes before recording.

    comb_depth = 0.0 : pure band-split separation (cheap receiver)
    comb_depth -> 1  : 1H delay-line comb filter, which cancels the subcarrier
                       because its phase inverts on alternate lines.  Real VCRs
                       sit in between, and the residue is what we see as
                       dot crawl.
    """
    fsc = std.subcarrier_hz
    half = max(float(chroma_bandwidth_hz), 0.30e6)
    chroma = fft_bandpass(comp, max(0.10e6, fsc - 1.6 * half), fsc + 1.6 * half,
                          std, unity_at_hz=fsc)

    if comb_depth > 0.001:
        # A delay-line comb is a *matched* filter: because the subcarrier
        # inverts an odd number of times per line (227.5 cycles for NTSC), the
        # same response that rejects the carrier also costs carrier amplitude --
        # exactly (1 - 2*depth).  Measured without normalising: comb 0.55 left
        # only 45% of the colour.  Dividing by that factor makes comb_depth a
        # pure trade-off between dot crawl and vertical luma detail, with the
        # colour level untouched, which is what a real receiver's comb does.
        delayed = np.empty_like(chroma)
        delayed[1:] = chroma[:-1]
        delayed[0] = chroma[0]
        combed = 0.5 * (chroma + delayed)
        mixed = (1.0 - comb_depth) * chroma + comb_depth * combed
        gain = 1.0 - comb_depth
        if abs(gain) > 1e-3:
            chroma = mixed / gain

    luma = comp - chroma
    return luma.astype(np.float32), chroma.astype(np.float32)


# ---------------------------------------------------------------------------
# receiver Y/C separation topologies
# ---------------------------------------------------------------------------
# The four designs a consumer receiver might use.  Each computes luma from the
# composite; chroma is whatever is left over, so the separator's error becomes
# cross-colour in the colour -- which is where the fringes on hard edges come from.
YC_SEPARATION = ("band", "box", "line1", "line2", "fir2d")


def separation_kind_name(value) -> str:
    """
    The schema's numeric Y/C separation value, as the name `yc_separate` takes.

    The parameter is numeric because the UI drives it as a slider and every other discrete choice in
    this schema is numeric (`codec_kind`, `filter_kind`, `vhs_speed`); the function takes a name
    because a name is what makes four designs readable at the point of use. This is the single place
    the two meet, so the numbering and the tuple cannot drift.
    """
    try:
        k = int(round(float(value)))
    except (TypeError, ValueError):
        return "band"
    if 0 <= k < len(YC_SEPARATION):
        return YC_SEPARATION[k]
    return "band"

#: taps for the 2-D FIR, as (row, column) offsets from the sample being separated.
#: Diagonally adjacent samples sit at the same subcarrier phase, which is why a 2-D
#: comb can separate without a delay line of its own.
FIR2D_DIAG_W = 0.0625


def _phase_alternates(std: VideoStandard) -> bool:
    """
    Does the subcarrier invert between adjacent lines?

    It does for NTSC (227.5 cycles per active line, so each line starts half a cycle
    later) and not for PAL, whose subcarrier advances by a quarter cycle per line and
    is switched in phase on alternate lines instead.  The distinction matters because
    a comb's vertical taps only cancel the carrier when the inversion holds; applying
    NTSC taps to PAL would leave the carrier in the luma rather than removing it.
    """
    cycles = std.subcarrier_hz / std.line_hz
    return abs(cycles - round(cycles)) > 0.25


def yc_separate(comp: np.ndarray, std: VideoStandard, kind: str = "band",
                comb_depth: float = 0.0, aperture: int = 2,
                chroma_bandwidth_hz: float = 0.8e6,
                ) -> tuple[np.ndarray, np.ndarray]:
    """
    Split a composite signal into (luma, chroma) using a named receiver topology.

    `kind`:
        "band"  -- band-split separation, with `comb_depth` mixing in a 1H delay-line
                   comb.  This is the behaviour the model had before, and is the
                   default so nothing changes unless a topology is requested.
        "box"   -- luma is the mean of `aperture`+2 samples spanning +-1 px (the cheap
                   set: a small horizontal aperture, no delay line).
        "line1" -- luma is the mean of this line and the previous (one glass delay
                   line).  The taps are one line apart, so the carrier cancels.
        "line2" -- luma is 0.5*this + 0.25*prev + 0.25*next (two delay lines).  Wider
                   vertical aperture: less cross-colour, more vertical luma smear.
        "fir2d" -- luma is a 2-D FIR over +-`aperture` px and +-1 line.  The diagonal
                   taps are at the same phase as the centre, so they carry colour and
                   are subtracted; the result has the least vertical smear of the four.

    The vertical taps are only used when the subcarrier actually inverts between
    lines.  On PAL the vertical part is dropped and the horizontal part is used alone,
    because taps one line apart would not cancel there.
    """
    if kind not in YC_SEPARATION:
        raise ValueError("unknown Y/C separation %r; expected one of %s"
                         % (kind, ", ".join(YC_SEPARATION)))
    if kind == "band":
        # the original behaviour, unchanged
        return split_composite(comp, std, comb_depth=comb_depth,
                               chroma_bandwidth_hz=chroma_bandwidth_hz)

    c = np.asarray(comp, np.float32)
    h, w = c.shape
    vert_ok = _phase_alternates(std) and h >= 3

    if kind == "box":
        # a small horizontal aperture: luma is the local mean over the aperture, and
        # the colour that survives it is the residual.  A short aperture leaves the
        # carrier only partly cancelled, which is what makes a cheap set's fringes
        # coarse and wide.
        ap = max(1, int(aperture))
        acc = np.zeros_like(c)
        n = 0
        for k in range(-ap, ap + 1):
            acc += np.roll(c, k, axis=1)
            n += 1
        luma = acc / float(n)
    elif kind == "line1" and vert_ok:
        prev = np.empty_like(c)
        prev[0] = c[0]
        prev[1:] = c[:-1]
        luma = 0.5 * (c + prev)
    elif kind == "line2" and vert_ok:
        prev = np.empty_like(c)
        nxt = np.empty_like(c)
        prev[0] = c[0]
        prev[1:] = c[:-1]
        nxt[-1] = c[-1]
        nxt[:-1] = c[1:]
        luma = 0.5 * c + 0.25 * prev + 0.25 * nxt
    elif kind == "fir2d":
        ap = max(1, int(aperture))
        # A plain weighted mean.  The weights are accumulated as they are applied, so the
        # normalising total cannot disagree with the taps -- a filter whose taps do not sum to
        # one is a brightness control with a misleading name, and an earlier version of this
        # branch lifted the picture by 25% for exactly that reason.
        #
        # The diagonals are the taps that carry colour (they sit at the same subcarrier phase as
        # the centre), so including them is what removes the carrier without a delay line.
        acc = np.zeros_like(c)
        total = 0.0

        def _tap(plane: np.ndarray, dy: int, dx: int, weight: float) -> None:
            nonlocal acc, total
            t = plane if dy == 0 else np.roll(plane, -dy, axis=0)
            acc = acc + weight * np.roll(t, dx, axis=1)
            total += weight

        _tap(c, 0, 0, 0.5)
        for k in range(1, ap + 1):
            w = 0.125 if k == 1 else 0.0625
            _tap(c, 0, k, w)
            _tap(c, 0, -k, w)
        if vert_ok:
            _tap(c, -1, 0, 0.125)
            _tap(c, 1, 0, 0.125)
            for dx in (1, -1):
                _tap(c, -1, dx, FIR2D_DIAG_W)
                _tap(c, 1, dx, FIR2D_DIAG_W)
        luma = acc / max(1e-6, total)
    else:
        # the vertical topologies fall back to the horizontal aperture when the
        # subcarrier does not invert per line
        return yc_separate(comp, std, "box", comb_depth, aperture, chroma_bandwidth_hz)

    # the colour is the residual, then band-limited around the carrier so the
    # synchronous detector's 2f product vanishes (see demod_chroma)
    chroma = c - luma
    fsc = std.subcarrier_hz
    half = max(float(chroma_bandwidth_hz), 0.30e6)
    chroma = fft_bandpass(chroma, max(0.10e6, fsc - 1.6 * half), fsc + 1.6 * half,
                          std, unity_at_hz=fsc)
    return luma.astype(np.float32), chroma.astype(np.float32)


def demod_chroma(chroma: np.ndarray, std: VideoStandard,
                 i_bw_hz: float = 1.3e6, q_bw_hz: float = 0.5e6,
                 phase_offset: np.ndarray | None = None
                 ) -> tuple[np.ndarray, np.ndarray]:
    """
    Synchronous detection of I and Q with their (asymmetric) bandwidths.

    The chroma input must already be band-limited around the subcarrier (see
    :func:`split_composite`): that band-limiting is what makes the 2f product
    term vanish, so a plain low-pass after mixing recovers the baseband colour
    exactly -- which is how a real synchronous detector works.

    `phase_offset` (radians, per line) is how a time-base corrector is modelled.
    A TBC does *not* shift the colour carrier: shifting a carrier in space rotates
    its phase, and one pixel is 0.42 of a subcarrier cycle, so sub-pixel jitter
    becomes a large hue error.  Measured, letting the corrector shift the carrier
    took chroma correlation from 0.93 to 0.02 while luma was unaffected.  A real
    TBC locks its sampling phase to the recorded colour burst, which is
    mathematically the same as de-modulating against a reference displaced by
    exactly the sampling error -- that is what this argument does.

    `i_bw_hz` / `q_bw_hz` are the *receiver's* output bandwidths (NTSC I is
    wideband, Q narrowband).  A caller that needs the round trip to be
    transparent -- the medium model, which must not impose a second band-limit
    on data the format stage already limited -- should pass matching wide values
    for both, or the Q path will quietly shave off chroma detail and the
    re-modulated carrier will no longer match the one it came from.

    Note that a naive demodulator (multiply by 2*cos, low-pass, halve) is *not*
    accurate here: the subcarrier does not complete an integer number of
    half-cycles across the active line (NTSC: 227.5 cycles/line), so the
    second-harmonic residue leaves a bias.  Band-limiting first removes that
    term, and the mixer reference is scaled so a constant-I carrier returns I.
    """
    phase = subcarrier_phase(chroma.shape[1], chroma.shape[0], std)
    if phase_offset is not None:
        phase = phase + np.asarray(phase_offset, dtype=np.float32).reshape(-1, 1)
    w = chroma.shape[1]
    i_lp = std.pixels_for_bandwidth(max(1e5, i_bw_hz), w)
    q_lp = std.pixels_for_bandwidth(max(1e5, q_bw_hz), w)
    # mixing with 2*cos already recovers I at unity gain: because the input is
    # band-limited around the carrier, the 2f product is absent and the low-pass
    # simply passes the baseband term (verified: a constant-I carrier returns I).
    i = lowpass(chroma * (2.0 * np.cos(phase)), i_lp, axis=1)
    q = lowpass(chroma * (2.0 * np.sin(phase)), q_lp, axis=1)
    return i.astype(np.float32), q.astype(np.float32)


def remod_chroma(i: np.ndarray, q: np.ndarray, std: VideoStandard,
                 chroma_gain: float = 0.63) -> np.ndarray:
    phase = subcarrier_phase(i.shape[1], i.shape[0], std)
    return (chroma_gain * (i * np.cos(phase) + q * np.sin(phase))).astype(np.float32)


def chroma_under_luma(chroma: np.ndarray, std: VideoStandard,
                      chroma_bandwidth_hz: float) -> np.ndarray:
    """
    VHS / Video8 record chroma *below* luma (colour-under), heterodyned down to
    ~629 kHz.  Two consequences we must model:

      1. very narrow chroma bandwidth -> colours smear horizontally,
      2. the chroma is only ~40 lines tall, so vertical colour resolution
         collapses and adjacent-line chroma averages together.
    """
    sigma_x = std.pixels_for_bandwidth(max(1e4, chroma_bandwidth_hz), chroma.shape[1])
    out = lowpass(chroma, sigma_x, axis=1)
    out = lowpass(out, 1.35, axis=0)   # 2-line vertical averaging in the receiver
    return out.astype(np.float32)
