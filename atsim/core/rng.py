"""
Deterministic stochastic processes with the *right temporal structure*.

The single most important engineering rule in the reference plan:

    "Do not let all random numbers be regenerated every frame."

A tape being played back is one physical object being read continuously.  Its
noise has

  * a fixed spatial texture (the same grain pattern on the same tape area),
  * a slowly varying temporal component (head/tape contact, transport drift),
  * intermittent localised defects (dropouts, creases, stains).

This module provides the building blocks: seeded streams, mean-reverting
(Ornstein-Uhlenbeck-like) processes for transport drift, and value noise with
separable spatial and temporal correlation lengths.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# deterministic seed derivation
# ---------------------------------------------------------------------------
def derive_seed(base: int, *tags: object) -> int:
    """
    Derive a stable 32-bit sub-seed from a project seed plus tags.

    Using a hash (instead of `base + counter`) means adding a new noise source
    later cannot shift the random draws of every existing source, which would
    break render reproducibility of previously saved projects.
    """
    h = hashlib.blake2s(digest_size=8)
    h.update(str(int(base)).encode())
    for t in tags:
        h.update(b"\x00")
        h.update(str(t).encode())
    return int.from_bytes(h.digest()[:4], "little")


def rng_for(base_seed: int, *tags: object) -> np.random.Generator:
    return np.random.default_rng(derive_seed(base_seed, *tags))


# ---------------------------------------------------------------------------
# mean-reverting process (transport / time-base drift)
# ---------------------------------------------------------------------------
@dataclass
class OUProcess:
    """
    Discrete Ornstein-Uhlenbeck process: x[n] = x[n-1] + theta*(mu-x) + sigma*w

    Used for quantities that are physically anchored (transport speed, head
    drum phase, tape tension) and cannot wander off to infinity, unlike a pure
    random walk.  `corr_len` is in samples (lines or fields) and is converted
    into a per-step reversion coefficient.

    A VCR's transport drift is the *same continuous process* no matter how much
    of the clip is rendered, so the process is stateful: `advance` continues
    where the previous call stopped.  That is what lets the tape plan grow
    lazily instead of being allocated for the whole clip up front -- sizing the
    plan from a "frames are unbounded" render context is exactly the bug that
    made an early build allocate 19 GB.
    """
    rng: np.random.Generator
    sigma: float
    corr_len: float = 8.0
    theta_scale: float = 1.0
    _state: float = 0.0

    @property
    def theta(self) -> float:
        corr = max(1e-6, float(self.corr_len))
        return float(np.clip(self.theta_scale / corr, 1e-6, 1.0))

    def reset(self) -> None:
        self._state = 0.0

    def advance(self, n: int) -> np.ndarray:
        """
        Return the next `n` samples, continuing from the internal state.

        Implemented as an exponential moving average over white noise with
        `lfilter`, i.e. O(n) inside compiled code.  Output is scaled to the
        requested `sigma`, which is the process's stationary standard
        deviation.
        """
        from scipy.signal import lfilter

        n = int(max(0, n))
        if n == 0:
            return np.zeros(0, dtype=np.float64)
        theta = self.theta
        w = self.rng.standard_normal(n)
        out = lfilter([theta], [1.0, -(1.0 - theta)], w).astype(np.float64)
        self._state = float(out[-1])
        if self.sigma != 1.0:
            out *= float(self.sigma)
        return out

    def sample(self, n: int, initial: float | None = None) -> np.ndarray:
        """One-shot sample of `n` fresh samples."""
        if initial is not None:
            self._state = float(initial)
        return self.advance(n)

    def sample_fast(self, n: int) -> np.ndarray:
        """Alias used at call sites that want the cheap vectorised path."""
        return self.advance(n)

    def prefix(self, n: int) -> np.ndarray:
        """
        The first `n` samples from a fresh start (does not disturb state).

        Per-frame stages that need "the value of this slow process as of frame
        k" use this: the transport really did run through those samples on the
        way here, so recomputing the prefix is physically correct, and it is
        O(n) once rather than O(n) per frame.
        """
        n = int(max(0, n))
        if n == 0:
            return np.zeros(0, dtype=np.float64)
        keep = self._state
        self._state = 0.0
        out = self.advance(n)
        self._state = keep
        return out


# ---------------------------------------------------------------------------
# coherent value noise (spatial texture + slow temporal change)
# ---------------------------------------------------------------------------
class ValueNoise3D:
    """
    Seeded, tileable-ish value noise over (frame, y, x) with independent
    spatial and temporal correlation lengths.

    Implementation: random lattice at a coarse grid, then trilinear
    interpolation.  Cost is O(N) per frame and the *lattice* is generated once
    per clip, which is exactly the "fixed spatial texture, slowly changing
    temporal component" structure we want: for tex_seed stability the lattice
    is a function of the seed only, and time only moves the sampling point.
    """

    def __init__(self, seed: int, height: int, width: int,
                 spatial_scale: float = 24.0, time_scale: float = 12.0,
                 temporal_drift: float = 1.0):
        self.seed = int(seed)
        self.height = int(height)
        self.width = int(width)
        self.spatial_scale = max(2.0, float(spatial_scale))
        self.time_scale = max(1.0, float(time_scale))
        self.temporal_drift = float(temporal_drift)
        rng = np.random.default_rng(seed)
        gy = int(np.ceil(self.height / self.spatial_scale)) + 4
        gx = int(np.ceil(self.width / self.spatial_scale)) + 4
        gt = 8  # temporal lattice period; wraps slowly
        self._lattice = rng.standard_normal((gt, gy, gx)).astype(np.float32)
        self._gt, self._gy, self._gx = gt, gy, gx
        # pre-computed fractional coordinates in x/y (they never change)
        self._fy = (np.arange(self.height, dtype=np.float32) / self.spatial_scale)
        self._fx = (np.arange(self.width, dtype=np.float32) / self.spatial_scale)

    def sample(self, frame_index: int) -> np.ndarray:
        """Return a (H, W) float32 field with zero mean, unit-ish variance."""
        t = frame_index / self.time_scale * self.temporal_drift
        t0 = int(np.floor(t)) % self._gt
        t1 = (t0 + 1) % self._gt
        ft = np.float32(t - np.floor(t))

        y0 = np.floor(self._fy).astype(np.int32)
        x0 = np.floor(self._fx).astype(np.int32)
        wy = (self._fy - y0)[:, None].astype(np.float32)
        wx = (self._fx - x0)[None, :].astype(np.float32)

        def plane(ti: int) -> np.ndarray:
            lat = self._lattice[ti]
            a = lat[np.ix_(y0, x0)]
            b = lat[np.ix_(y0, x0 + 1)]
            c = lat[np.ix_(y0 + 1, x0)]
            d = lat[np.ix_(y0 + 1, x0 + 1)]
            top = a + (b - a) * wx
            bot = c + (d - c) * wx
            return top + (bot - top) * wy

        p0 = plane(t0)
        p1 = plane(t1)
        return (p0 + (p1 - p0) * ft).astype(np.float32)

    def static_field(self, smooth: int = 1) -> np.ndarray:
        """A time-independent spatial texture (for 'fixed grain')."""
        rng = np.random.default_rng(self.seed ^ 0x5EED)
        f = rng.standard_normal((self.height, self.width)).astype(np.float32)
        for _ in range(max(0, int(smooth))):
            f = 0.5 * (f + np.roll(f, 1, axis=0))
            f = 0.5 * (f + np.roll(f, 1, axis=1))
        s = f.std()
        return f / s if s > 1e-9 else f


def bandlimited_noise(rng: np.random.Generator, shape: Sequence[int],
                      corr_y: float = 1.0, corr_x: float = 1.0) -> np.ndarray:
    """
    Gaussian white noise low-passed with a separable exponential kernel.

    corr_* = 1.0 gives white noise; larger values give a coarser grain, which
    is how spatial noise *scale* is controlled separately from noise *level*.
    Implemented with a single-pole IIR (lfilter) so cost stays O(H*W).
    """
    from scipy.signal import lfilter

    shape = tuple(int(s) for s in shape)
    n = rng.standard_normal(shape).astype(np.float32)
    if corr_x > 1.001:
        a = 1.0 - 1.0 / float(corr_x)
        n = lfilter([1.0 - a], [1.0, -a], n, axis=1).astype(np.float32)
    if corr_y > 1.001:
        a = 1.0 - 1.0 / float(corr_y)
        n = lfilter([1.0 - a], [1.0, -a], n, axis=0).astype(np.float32)
    s = float(n.std())
    if s > 1e-9:
        n /= s
    return n
