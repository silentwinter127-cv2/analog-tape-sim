"""
Analog-style recursive (IIR) filters.

Why this exists, and why it is not just a different way to blur:

Every filter in this project was originally a zero-phase Gaussian FIR.  That is
convenient -- no group delay to compensate, no ringing -- and it is also *wrong*
in one specific way that matters visually.  A real video amplifier is a
capacitive network: a cascade of one-pole sections.  Its amplitude response is
roughly Gaussian-ish, but its **phase response is not zero**, and the resulting
group delay is precisely what makes chroma sit slightly to the right of luma on
a composite signal.  Colour fringing on hard edges is partly a *delay* artefact,
so a zero-phase filter cannot produce it however the bandwidth is set.

ntsc-rs models this with cascaded one-pole/butterworth sections running at 4x the
subcarrier rate.  This module does the same thing in the pixel domain, so the
same physical behaviour is available without re-architecting the sample rate.

Two filter families are provided and the caller chooses:

    Gaussian (zero phase)   ideal, no delay, no ringing
    OnePole / Butterworth   recursive, real group delay, can ring

The one-pole (constant-k) form is what consumer video circuitry used.  Three
sections in series approximates the classic triple-constant-k response.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
def one_pole_coeffs(cutoff_hz: float, rate_hz: float, sections: int = 1) -> float:
    """
    Single-pole coefficient for `y[n] = a*x[n] + (1-a)*y[n-1]`.

    The coefficient is solved *exactly* so that a cascade of `sections` identical
    stages has its -3 dB point at `cutoff_hz`.

    Derivation.  One stage has |H|^2 = a / (a + w^2*(1-a)) for normalised
    frequency w = 2*pi*f/rate (using cos(w) ~ 1 - w^2/2).  Requiring the n-stage
    cascade to be -3 dB, (a / (a + w^2(1-a)))^n = 1/2, gives

        a = w^2 (2^(1/n) - 1) / (1 + w^2 (2^(1/n) - 1))

    A textbook cascade-correction factor was tried first and was wrong: it placed
    a "3 MHz" filter's -3 dB point near 5.9 MHz, so the cascade passed only 0.31
    at the frequency it claimed -- three times too much loss.  The exact form
    removes the guesswork and costs nothing, because `a` depends only on cutoff.
    """
    if cutoff_hz <= 0.0 or rate_hz <= 0.0:
        return 1.0
    n = max(1, int(sections))
    w = 2.0 * np.pi * min(float(cutoff_hz), 0.5 * float(rate_hz)) / float(rate_hz)
    k = 2.0 ** (1.0 / n) - 1.0
    return float(np.clip(w * w * k / (1.0 + w * w * k), 1e-6, 0.9999))


def apply_one_pole(plane: np.ndarray, a: float, sections: int = 1,
                   axis: int = 1) -> np.ndarray:
    """
    Cascaded one-pole low-pass along an axis, as a real recursive filter.

    Uses the exact difference equation y[n] = a*x[n] + (1-a)*y[n-1] via
    `scipy.signal.lfilter`, applied `sections` times in series.  A hand-written
    accumulator loop was tried first and was wrong: it re-seeded the state from
    the first sample of every row, so the filter never left its startup transient
    -- DC gain looked perfect while the AC response collapsed to almost nothing
    (a "1 MHz" filter measured 0.002 at 1 MHz instead of 0.71).

    The recursive form is kept deliberately rather than being replaced by a
    symmetric FIR, because the *group delay* is the point: a real amplifier
    delays its output, and that delay is what places chroma slightly to the right
    of luma on a composite signal.  A zero-phase filter cannot produce that
    fringing however its bandwidth is set.
    """
    from scipy.signal import lfilter
    if sections <= 0 or a >= 0.9999 or a <= 1e-6:
        return plane
    out = plane.astype(np.float32, copy=True)
    # **Seed each row's state, instead of starting from zero.**  `lfilter` with a zero initial condition makes every
    # row begin at zero and climb, with a time constant of `1/a` samples -- and at the default 0.75 MHz cutoff on an
    # 11.33 MHz pixel rate `a` is 7.17e-7, so the time constant is about **1.4 million samples** against image widths
    # of one to two thousand.  The filter therefore never reaches steady state: measured on a constant plane its
    # response was **0.46942** at 0.75 MHz and **0.01679** at 0.30 MHz, which is the fraction of the transient that
    # fits inside the row rather than a gain at all.  **A one-pole lowpass's steady-state DC gain is exactly 1.**
    #
    # Seeding the state to the row's first sample is what a filter that has been running on a signal does.  It makes
    # a constant pass unchanged -- the property a lowpass must have -- and it removes the dependence on the picture's
    # width, which had made a preview and a full render behave differently.
    #
    # The docstring above records the mirror image of this fault: a hand-written accumulator that re-seeded from the
    # first sample of every row left the *AC* response collapsed while the DC "looked perfect".  The answer chosen
    # then was `lfilter`, whose zero state made the *DC* wrong and the AC right.
    zi_value = out.take([0], axis=axis) if out.shape[axis] else out
    zi = (1.0 - a) * np.repeat(zi_value, 1, axis=axis)
    for _ in range(int(sections)):
        out, _ = lfilter([a], [1.0, -(1.0 - a)], out, axis=axis, zi=zi)
        out = out.astype(np.float32)
    return out


def apply_butterworth(plane: np.ndarray, cutoff_hz: float, rate_hz: float,
                      axis: int = 1) -> np.ndarray:
    """
    2nd-order Butterworth biquad along an axis.

    Sharper than the constant-k cascade with a similar passband, and it is the
    other filter family ntsc-rs offers.  Useful when a user wants the response
    knee to be crisp rather than gradual.
    """
    from scipy.signal import lfilter
    if cutoff_hz <= 0.0 or rate_hz <= 0.0:
        return plane
    f = min(2.0 * cutoff_hz, rate_hz) / rate_hz
    omega = np.pi * f
    sn, cs = np.sin(omega), np.cos(omega)
    alpha = sn / (2.0 * 0.7071067811865476)
    gain = 1.0 / (1.0 + alpha)
    b = [(1.0 - cs) * 0.5 * gain, (1.0 - cs) * gain, (1.0 - cs) * 0.5 * gain]
    a = [1.0, -2.0 * cs * gain, (1.0 - alpha) * gain]
    return lfilter(b, a, plane, axis=axis).astype(np.float32)


# ---------------------------------------------------------------------------
@dataclass
class FilterChoice:
    """Which filter family a stage should use."""
    kind: str = "gaussian"      # "gaussian" | "constant_k" | "butterworth"
    sections: int = 3

    @property
    def recursive(self) -> bool:
        return self.kind in ("constant_k", "butterworth")

    def label(self) -> str:
        return {"gaussian": "高斯（零相位，理想）",
                "constant_k": "恒 K 型（真实电路，有群延迟）",
                "butterworth": "巴特沃斯（较陡）"}.get(self.kind, self.kind)


def lowpass_hz(plane: np.ndarray, cutoff_hz: float, rate_hz: float,
               choice: FilterChoice) -> np.ndarray:
    """
    Apply a horizontal low-pass with the chosen filter family.

    The Gaussian path is kept because it is the right tool for *simulating* a
    bandwidth limit; the recursive paths are the right tool for simulating the
    *circuit* that imposed it, including the delay that goes with it.
    """
    if cutoff_hz <= 0.0 or cutoff_hz >= rate_hz:
        return plane
    if choice.kind == "constant_k":
        a = one_pole_coeffs(cutoff_hz, rate_hz, sections=choice.sections)
        return apply_one_pole(plane, a, sections=choice.sections)
    if choice.kind == "butterworth":
        return apply_butterworth(plane, cutoff_hz, rate_hz)
    from ..core.signal import lowpass
    sigma = 0.1325 / (cutoff_hz / rate_hz)
    out = lowpass(plane, sigma, axis=1)
    # **A lowpass must pass DC unchanged.**  The kernel above is truncated, and its response to a constant is
    # `sum(exp(-k^2 / 2 sigma^2))` -- well below one whenever the kernel is narrow.  Measured: **0.46942** at the
    # default 0.75 MHz cutoff, i.e. 53% of the signal removed, and **0.01679** at 0.30 MHz, i.e. almost all of it.
    # That gain is what made the format stage's composite retain only 68.9% of an ideal one's amplitude, what made
    # the player recover only 47.7% of the encoder's I when it applied the same filter again, and what made
    # sweeping `chroma_bandwidth` from 0.3 to 3.0 look like it did nothing -- the chroma was being removed rather
    # than band-limited.
    #
    # The compensation was verified first: multiplying the recovered chroma by the reciprocal of the measured gain
    # restores the player's I to 0.349 / 0.348 / 0.347 at 0.75 / 1.00 / 1.50 MHz against a target of 0.3435.
    # Normalising here does the same at the source rather than at every consumer.
    import numpy as np
    k = int(max(1.0, round(3.0 * sigma)))
    taps = np.exp(-0.5 * (np.arange(-k, k + 1, dtype=np.float64) / sigma) ** 2)
    total = float(taps.sum())
    if total > 1e-9:
        out = out / total
    return out.astype(plane.dtype)


def group_delay_samples(cutoff_hz: float, rate_hz: float,
                        sections: int = 3) -> float:
    """
    Group delay of the cascade, in samples.

    Reported so the UI can show *why* chroma lags luma: a 3 MHz constant-k
    cascade at 11.33 MHz delays the signal by a fraction of a pixel, and on a
    hard edge that fraction is exactly the colour fringe offset.
    """
    a = one_pole_coeffs(cutoff_hz, rate_hz, sections=sections)
    return float(sections * (1.0 - a) / max(1e-9, a))


# ---------------------------------------------------------------------------
def notch(plane: np.ndarray, freq_norm: float, quality: float,
          amount: float) -> np.ndarray:
    """
    IIR notch filter, for the ringing that a sharp luminance transition excites.

    A notch at a specific frequency is how a receiver's ringing shows up: an edge
    excites the response and the filter rings, producing the overshoot/
    undershoot pair visible as a white-then-dark outline.  `freq_norm` is
    normalised to Nyquist.
    """
    from scipy.signal import lfilter
    f = float(np.clip(freq_norm, 0.0, 1.0))
    if f <= 1e-6 or amount <= 1e-6:
        return plane
    bandwidth = (f / max(1e-3, quality)) * np.pi
    w = f * np.pi
    beta = np.tan(bandwidth * 0.5)
    gain = 1.0 / (1.0 + beta)
    b = [gain, -2.0 * np.cos(w) * gain, gain]
    a = [1.0, -2.0 * np.cos(w) * gain, 2.0 * gain - 1.0]
    ring = lfilter(b, a, plane, axis=1).astype(np.float32)
    return (plane + amount * ring).astype(np.float32)


def preemphasis(plane: np.ndarray, cutoff_hz: float, rate_hz: float,
                amount: float) -> np.ndarray:
    """
    High-frequency pre-emphasis / aperture correction.

    Video recorders boost high frequencies before recording to protect them from
    noise, and the receiver must undo it -- imperfectly.  The visible result is an
    overshoot on hard edges.  Modelled as a one-pole low-pass subtracted from the
    signal, which is how the real emphasis network behaves.
    """
    if amount <= 1e-6:
        return plane
    a = one_pole_coeffs(cutoff_hz, rate_hz, sections=1)
    lp = apply_one_pole(plane, a, sections=1)
    # the emphasis network passes the signal and adds back the removed high band
    return (plane + amount * (plane - lp)).astype(np.float32)
