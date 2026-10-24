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
from scipy.fft import irfft as _sp_irfft, next_fast_len as _next_fast_len, rfft as _sp_rfft
from scipy.ndimage import convolve1d
from scipy.signal import lfilter as _lfilter

from .standards import VideoStandard

try:
    import cv2 as _cv2
    _HAVE_CV2 = True
except Exception:
    _cv2 = None  # type: ignore[assignment]
    _HAVE_CV2 = False

TWO_PI = 2.0 * np.pi
_ONE_KERNEL = np.array([1.0], dtype=np.float32)

# NTSC / SMPTE 170M weights (also the Rec.601 luma weights)
RGB_TO_Y = np.array([0.299, 0.587, 0.114], dtype=np.float64)

_GAUSS_CACHE: dict[int, np.ndarray] = {}
_SUBCARRIER_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
_FFT_BP_CACHE: dict[tuple, np.ndarray] = {}


# ---------------------------------------------------------------------------
# kernel design
# ---------------------------------------------------------------------------
def gaussian_kernel(sigma: float) -> np.ndarray:
    """
    Finite Gaussian kernel, normalised to unity DC gain.
    """
    sigma = float(max(1e-6, sigma))
    if sigma < 0.30:
        return _ONE_KERNEL
    key = int(round(sigma * 100000.0))
    hit = _GAUSS_CACHE.get(key)
    if hit is not None:
        return hit
    radius = int(np.ceil(3.0 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    out = (k / k.sum()).astype(np.float32)
    if len(_GAUSS_CACHE) > 256:
        _GAUSS_CACHE.pop(next(iter(_GAUSS_CACHE)), None)
    _GAUSS_CACHE[key] = out
    return out


def highpass_kernel(sigma: float) -> np.ndarray:
    """Complementary high-pass: delta - lowpass (DC removed exactly)."""
    k = gaussian_kernel(sigma)
    hp = -k.copy()
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


def _fft_fir_1d(plane: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    """Exact reflect-padded 1D FIR filter via float32 pocketfft for wide kernels."""
    klen = len(kernel)
    rad = klen // 2
    pad_width = [(0, 0)] * plane.ndim
    pad_width[axis] = (rad, rad)
    # scipy.ndimage mode='reflect' ('d c b a | a b c d | d c b a') equals np.pad mode='symmetric'
    padded = np.pad(plane, pad_width, mode="symmetric")
    n = padded.shape[axis]
    nfft = _next_fast_len(n + klen - 1, real=True)
    fk = _sp_rfft(kernel[::-1], n=nfft)
    shape = [1] * plane.ndim
    shape[axis] = -1
    fk = fk.reshape(shape)
    fp = _sp_rfft(padded, n=nfft, axis=axis)
    out = _sp_irfft(fp * fk, n=nfft, axis=axis)
    sl = [slice(None)] * plane.ndim
    sl[axis] = slice(2 * rad, 2 * rad + plane.shape[axis])
    return out[tuple(sl)].astype(plane.dtype, copy=False)


def apply_fir(plane: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    """Separable FIR along one axis, reflect-padded at the edges."""
    klen = len(kernel)
    if klen == 1:
        return plane
    if (
        _HAVE_CV2
        and plane.ndim in (2, 3)
        and plane.dtype == np.float32
        and (plane.ndim == 2 or plane.shape[2] <= 4)
        and klen <= 255
        and plane.shape[0] > klen
        and plane.shape[1] > klen
    ):
        k_rev = kernel[::-1]
        if axis == 1:
            return _cv2.sepFilter2D(plane, _cv2.CV_32F, k_rev, _ONE_KERNEL, borderType=_cv2.BORDER_REFLECT)
        if axis == 0:
            return _cv2.sepFilter2D(plane, _cv2.CV_32F, _ONE_KERNEL, k_rev, borderType=_cv2.BORDER_REFLECT)
    if klen >= 39 and plane.dtype == np.float32 and plane.shape[axis] >= klen:
        return _fft_fir_1d(plane, kernel, axis)
    return convolve1d(plane, kernel, axis=axis, mode="reflect")


def lowpass(plane: np.ndarray, sigma: float, axis: int | None = None) -> np.ndarray:
    k = gaussian_kernel(sigma)
    klen = len(k)
    if klen == 1:
        return plane
    if axis is None:
        if (
            _HAVE_CV2
            and plane.ndim in (2, 3)
            and plane.dtype == np.float32
            and (plane.ndim == 2 or plane.shape[2] <= 4)
            and klen <= 255
            and plane.shape[0] > klen
            and plane.shape[1] > klen
        ):
            return _cv2.sepFilter2D(plane, _cv2.CV_32F, k, k, borderType=_cv2.BORDER_REFLECT)
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
    """
    h, w = plane.shape[0], plane.shape[1]
    sh = np.asarray(shifts, dtype=np.float32).reshape(-1)
    if sh.size < h:
        sh = np.pad(sh, (0, h - sh.size), mode="edge")
    sh = sh[:h]
    max_s = float(np.max(np.abs(sh)))
    if max_s < 1e-6:
        return plane
    left = np.floor(sh).astype(np.int32)
    cols = np.arange(w, dtype=np.int32)[None, :]
    step = np.where(sh > 1e-6, -1, np.where(sh < -1e-6, 1, 0)).astype(np.int32)[:, None]
    rows = np.arange(h, dtype=np.int32)[:, None]

    if mode == "wrap":
        idx0 = np.mod(cols - left[:, None], w)
        idx1 = np.mod(idx0 + step, w)
        a = plane[rows, idx0]
        b = plane[rows, idx1]
    else:
        pad = int(max(abs(int(left.min())), abs(int(left.max()) + 1)) + 1)
        pad = int(min(pad, 4096))
        if pad > 0:
            pad_mode = "reflect" if mode == "reflect" else "edge"
            padded = np.pad(plane, ((0, 0), (pad, pad)) + ((0, 0),) * (plane.ndim - 2),
                            mode=pad_mode)
        else:
            padded = plane
        idx0 = cols - left[:, None] + pad
        idx1 = idx0 + step
        a = padded[rows, idx0]
        b = padded[rows, idx1]

    frac = (sh - left)[:, None]
    if plane.ndim > 2:
        frac = frac[..., None]
    b = b - a
    b *= frac
    b += a
    return b.astype(plane.dtype, copy=False)


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
        sf = SignalFrame(self.y.copy(), self.i.copy(), self.q.copy())
        if hasattr(self, "color_system"):
            sf.color_system = getattr(self, "color_system")  # type: ignore[attr-defined]
        if getattr(self, "_rf_direct", False):
            sf._rf_direct = True  # type: ignore[attr-defined]
        return sf

    @classmethod
    def from_rgb(cls, rgb: np.ndarray, std: VideoStandard) -> "SignalFrame":
        y, i, q = rgb_to_yiq(rgb.astype(np.float32, copy=False))
        return cls(y.astype(np.float32, copy=False),
                   i.astype(np.float32, copy=False),
                   q.astype(np.float32, copy=False))

    def to_rgb(self, cross_luma: np.ndarray | None = None) -> np.ndarray:
        y = self.y if cross_luma is None else self.y + cross_luma
        if str(getattr(self, "color_system", "NTSC")).upper() == "PAL":
            # PAL: V = Q axis inverted every other line, U = I axis
            alt = np.where((np.arange(self.height) % 2 == 0)[:, None], 1.0, -1.0).astype(np.float32)
            q = self.q * alt
        else:
            q = self.q
        rgb = yiq_to_rgb(y, self.i, q)
        return np.clip(rgb, 0.0, 1.0).astype(np.float32, copy=False)

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
def _subcarrier_tables(width: int, height: int, std: VideoStandard) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    key = (int(width), int(height), std.name,
           round(float(std.subcarrier_rad_per_sample), 7),
           round(float(std.samples_per_line), 4))
    hit = _SUBCARRIER_CACHE.get(key)
    if hit is not None:
        return hit
    inc = std.subcarrier_rad_per_sample
    cols = np.arange(width, dtype=np.float64)[None, :]
    rows = np.arange(height, dtype=np.float64)[:, None]
    ph = (inc * cols + std.line_hz * 0.0 * rows + inc * std.samples_per_line * rows).astype(np.float32)
    cos_ph = np.cos(ph).astype(np.float32)
    sin_ph = np.sin(ph).astype(np.float32)
    res = (ph, cos_ph, sin_ph)
    if len(_SUBCARRIER_CACHE) > 24:
        _SUBCARRIER_CACHE.pop(next(iter(_SUBCARRIER_CACHE)), None)
    _SUBCARRIER_CACHE[key] = res
    return res


def subcarrier_phase(width: int, height: int, std: VideoStandard) -> np.ndarray:
    """
    (H, W) subcarrier phase at sample centres.
    """
    return _subcarrier_tables(width, height, std)[0]


def composite_encode(y: np.ndarray, i: np.ndarray, q: np.ndarray,
                     std: VideoStandard, chroma_gain: float = 0.63,
                     burst_phase: float = 0.0) -> np.ndarray:
    """Y + modulated chroma -> one composite signal (0..1 nominal)."""
    _, cos_ph, sin_ph = _subcarrier_tables(y.shape[1], y.shape[0], std)
    if abs(burst_phase) < 1e-8:
        c = np.float32(chroma_gain) * (i * cos_ph + q * sin_ph)
    else:
        cb = np.float32(np.cos(burst_phase))
        sb = np.float32(np.sin(burst_phase))
        cos_tot = cos_ph * cb - sin_ph * sb
        sin_tot = sin_ph * cb + cos_ph * sb
        c = np.float32(chroma_gain) * (i * cos_tot + q * sin_tot)
    return (y + c).astype(np.float32, copy=False)


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
    """
    pad = int(max(32, 0.25 * plane.shape[1]))
    src = _reflect_pad(plane, pad)
    n = src.shape[1]
    bp_key = (
        n,
        round(float(lo_hz), 1),
        round(float(hi_hz), 1),
        round(float(std.pixel_rate_hz), 1),
        None if smooth_hz is None else round(float(smooth_hz), 1),
        None if unity_at_hz is None else round(float(unity_at_hz), 1),
    )
    response = _FFT_BP_CACHE.get(bp_key)
    if response is None:
        freqs = np.fft.rfftfreq(n, d=1.0) * std.pixel_rate_hz
        order = 4
        hi_c = 0.5 * (lo_hz + hi_hz) + 0.5 * (hi_hz - lo_hz) * 1.6
        lo_c = 0.5 * (lo_hz + hi_hz) - 0.5 * (hi_hz - lo_hz) * 1.6
        hi_c = max(hi_c, lo_hz * 1.02)
        lo_c = max(1e4, min(lo_c, hi_c * 0.97))
        hi = 1.0 / (1.0 + (freqs / hi_c) ** (2 * order))
        lo = (freqs / lo_c) ** (2 * order) / (1.0 + (freqs / lo_c) ** (2 * order))
        resp = hi * lo
        if unity_at_hz:
            g = float(np.interp(unity_at_hz, freqs, resp))
            if g > 1e-6:
                resp = resp / g
        response = resp.astype(np.float32)
        if len(_FFT_BP_CACHE) > 64:
            _FFT_BP_CACHE.pop(next(iter(_FFT_BP_CACHE)), None)
        _FFT_BP_CACHE[bp_key] = response
    spec = _sp_rfft(src, axis=1)
    resp_view = response[None, :] if plane.ndim == 2 else response[None, :, None]
    out = _sp_irfft(spec * resp_view, n=n, axis=1).astype(plane.dtype, copy=False)
    return out[:, pad:pad + plane.shape[1]] if plane.ndim == 2 else \
        out[:, pad:pad + plane.shape[1], :]


def split_composite(comp: np.ndarray, std: VideoStandard,
                    comb_depth: float = 0.0, chroma_bandwidth_hz: float = 0.8e6,
                    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Band-split a composite signal into (luma, chroma).
    """
    fsc = std.subcarrier_hz
    half = max(float(chroma_bandwidth_hz), 0.30e6)
    chroma = fft_bandpass(comp, max(0.10e6, fsc - 1.6 * half), fsc + 1.6 * half,
                          std, unity_at_hz=fsc)

    if comb_depth > 0.001:
        delayed = np.empty_like(chroma)
        delayed[1:] = chroma[:-1]
        delayed[0] = chroma[0]
        combed = 0.5 * (chroma + delayed)
        mixed = (1.0 - comb_depth) * chroma + comb_depth * combed
        gain = 1.0 - comb_depth
        if abs(gain) > 1e-3:
            chroma = mixed / gain

    luma = comp - chroma
    return luma.astype(np.float32, copy=False), chroma.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# receiver Y/C separation topologies
# ---------------------------------------------------------------------------
YC_SEPARATION = ("band", "box", "line1", "line2", "fir2d", "iir")


def separation_kind_name(value) -> str:
    try:
        k = int(round(float(value)))
    except (TypeError, ValueError):
        return "band"
    if 0 <= k < len(YC_SEPARATION):
        return YC_SEPARATION[k]
    return "band"


FIR2D_DIAG_W = 0.0625


def _phase_alternates(std: VideoStandard) -> bool:
    cycles = std.subcarrier_hz / std.line_hz
    return abs(cycles - round(cycles)) > 0.25


def chroma_vert_blend(chroma: np.ndarray, feedback: float = 0.5) -> np.ndarray:
    """
    A one-pole vertical feedback filter on the chroma, as ntsc-rs's `chroma_vert_blend` is.

        out[i] = k*out[i-1] + (1-k)*in[i]
    """
    k = float(np.clip(feedback, 0.0, 0.995))
    if k <= 1e-6 or chroma.shape[0] <= 1:
        return chroma
    out = np.empty_like(chroma)
    out[0] = chroma[0]
    inv = 1.0 - k
    out[1:], _ = _lfilter([inv], [1.0, -k], chroma[1:], axis=0, zi=k * chroma[0:1])
    return out


def chroma_vert_coherence(feedback: float) -> float:
    """The 1/e coherence length in lines, for reporting and for a check to compare against."""
    k = float(np.clip(feedback, 0.0, 0.995))
    return 0.0 if k <= 1e-6 else float(-1.0 / np.log(k))


def yc_separate(comp: np.ndarray, std: VideoStandard, kind: str = "band",
                comb_depth: float = 0.0, aperture: int = 2,
                chroma_bandwidth_hz: float = 0.8e6,
                vert_blend: float = 0.5,
                ) -> tuple[np.ndarray, np.ndarray]:
    """
    Split a composite signal into (luma, chroma) using a named receiver topology.
    """
    if kind not in YC_SEPARATION:
        raise ValueError("unknown Y/C separation %r; expected one of %s"
                         % (kind, ", ".join(YC_SEPARATION)))
    if kind == "band":
        return split_composite(comp, std, comb_depth=comb_depth,
                               chroma_bandwidth_hz=chroma_bandwidth_hz)

    c = np.asarray(comp, np.float32)
    h, w = c.shape
    vert_ok = _phase_alternates(std) and h >= 3

    if kind == "box":
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
    elif kind == "iir" and vert_ok:
        luma = chroma_vert_blend(c, vert_blend)
    elif kind == "fir2d":
        ap = max(1, int(aperture))
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
        return yc_separate(comp, std, "box", comb_depth, aperture, chroma_bandwidth_hz,
                           vert_blend)

    chroma = c - luma
    fsc = std.subcarrier_hz
    half = max(float(chroma_bandwidth_hz), 0.30e6)
    chroma = fft_bandpass(chroma, max(0.10e6, fsc - 1.6 * half), fsc + 1.6 * half,
                          std, unity_at_hz=fsc)
    return luma.astype(np.float32, copy=False), chroma.astype(np.float32, copy=False)


def demod_chroma(chroma: np.ndarray, std: VideoStandard,
                 i_bw_hz: float = 1.3e6, q_bw_hz: float = 0.5e6,
                 phase_offset: np.ndarray | None = None
                 ) -> tuple[np.ndarray, np.ndarray]:
    """
    Synchronous detection of I and Q with their (asymmetric) bandwidths.
    """
    _, cos_ph, sin_ph = _subcarrier_tables(chroma.shape[1], chroma.shape[0], std)
    if phase_offset is not None:
        po = np.asarray(phase_offset, dtype=np.float32).reshape(-1, 1)
        co = np.cos(po)
        so = np.sin(po)
        cos_tot = cos_ph * co - sin_ph * so
        sin_tot = sin_ph * co + cos_ph * so
    else:
        cos_tot = cos_ph
        sin_tot = sin_ph
    w = chroma.shape[1]
    eff_w = min(w, int(std.samples_per_line))
    i_lp = std.pixels_for_bandwidth(max(1e5, i_bw_hz), eff_w)
    q_lp = std.pixels_for_bandwidth(max(1e5, q_bw_hz), eff_w)
    i = lowpass(chroma * (np.float32(2.0) * cos_tot), i_lp, axis=1)
    q = lowpass(chroma * (np.float32(2.0) * sin_tot), q_lp, axis=1)
    return i.astype(np.float32, copy=False), q.astype(np.float32, copy=False)


def remod_chroma(i: np.ndarray, q: np.ndarray, std: VideoStandard,
                 chroma_gain: float = 0.63) -> np.ndarray:
    _, cos_ph, sin_ph = _subcarrier_tables(i.shape[1], i.shape[0], std)
    return (np.float32(chroma_gain) * (i * cos_ph + q * sin_ph)).astype(np.float32, copy=False)


def chroma_under_luma(chroma: np.ndarray, std: VideoStandard,
                      chroma_bandwidth_hz: float) -> np.ndarray:
    """
    VHS / Video8 record chroma *below* luma (colour-under), heterodyned down to
    ~629 kHz.
    """
    sigma_x = std.pixels_for_bandwidth(max(1e4, chroma_bandwidth_hz), chroma.shape[1])
    out = lowpass(chroma, sigma_x, axis=1)
    out = lowpass(out, 1.35, axis=0)   # 2-line vertical averaging in the receiver
    return out.astype(np.float32, copy=False)
